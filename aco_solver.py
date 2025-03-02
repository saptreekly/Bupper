import numpy as np
from typing import List, Tuple, Dict, Optional
import random
import math
from utils import TimeWindow
import time
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.sparse import csr_matrix
import numpy.ma as ma

@dataclass
class SolutionMetrics:
    """Track solution improvement metrics"""
    iteration: int
    cost: float
    violations: int
    computation_time: float
    improvement_rate: float

class ACO:
    """Ant Colony Optimization with parallel processing and vectorized operations"""

    def __init__(self, 
                 base_ants: int = 40,
                 base_evaporation: float = 0.15,
                 alpha: float = 1.5,
                 beta: float = 2.5,
                 q0: float = 0.1,
                 evap_increase: float = 0.05,
                 stagnation_limit: int = 5,
                 speed: float = 1.0,
                 time_penalty_factor: float = 2.0,
                 base_time_penalty: float = 2.0,
                 lateness_multiplier: float = 1.5,
                 max_parallel_ants: int = 8,
                 verbose: bool = False):
        self.base_ants = base_ants
        self.base_evaporation = base_evaporation
        self.alpha = alpha
        self.beta = beta
        self.q0 = q0
        self.evap_increase = evap_increase
        self.stagnation_limit = stagnation_limit
        self.speed = speed
        self.time_penalty_factor = time_penalty_factor
        self.base_time_penalty = base_time_penalty
        self.lateness_multiplier = lateness_multiplier
        self.max_parallel_ants = max_parallel_ants
        self.VERBOSE = verbose

        # Performance tracking
        self.metrics = []
        self.last_improvement_iteration = 0
        self.best_solution_history = []

        # Precomputed data structures
        self.nearest_neighbors = None
        self.distance_matrix = None

    def precompute_nearest_neighbors(self, points: np.ndarray) -> None:
        """Precompute and store sorted nearest neighbors for each node"""
        n_points = len(points)
        self.distance_matrix = np.sqrt(np.sum((points[:, np.newaxis, :] - 
                                             points[np.newaxis, :, :]) ** 2, axis=2))
        # Mask diagonal to exclude self-distances
        masked_distances = ma.masked_array(self.distance_matrix, 
                                        mask=np.eye(n_points, dtype=bool))
        # Get sorted indices for each node
        self.nearest_neighbors = np.argsort(masked_distances, axis=1)

    def initialize_pheromone(self, n_points: int, conductivities: Optional[Dict] = None) -> np.ndarray:
        """Initialize pheromone matrix using conductivities if available"""
        pheromone = np.ones((n_points, n_points)) * self.q0

        if conductivities:
            # Vectorized initialization from conductivities
            max_cond = max(conductivities.values())
            for (i, j), cond in conductivities.items():
                scaled_pheromone = (cond / max_cond) * (1.0 - self.q0) + self.q0
                pheromone[i, j] = scaled_pheromone
                pheromone[j, i] = scaled_pheromone

        return pheromone

    def construct_solution_parallel(self, args: Tuple) -> List[int]:
        """Construct a single ant's solution (used in parallel processing)"""
        points, pheromone, route_nodes = args
        unvisited = set(route_nodes[1:])
        current = 0
        path = [current]

        while unvisited:
            # Use precomputed nearest neighbors
            neighbors = self.nearest_neighbors[current]
            valid_neighbors = [n for n in neighbors if n in unvisited]

            if not valid_neighbors:
                break

            # Calculate move probabilities vectorized
            pher_values = pheromone[current, valid_neighbors] ** self.alpha
            dist_values = (1.0 / np.maximum(self.distance_matrix[current, valid_neighbors], 1e-6)) ** self.beta
            probs = pher_values * dist_values

            # Normalize probabilities
            total = np.sum(probs)
            if total == 0 or not np.isfinite(total):
                probs = np.ones_like(probs) / len(probs)
            else:
                probs = probs / total

            # Select next city
            next_city = np.random.choice(valid_neighbors, p=probs)
            path.append(next_city)
            unvisited.remove(next_city)
            current = next_city

        path.append(0)
        return path

    def update_pheromone_vectorized(self,
                                  pheromone: np.ndarray,
                                  all_paths: List[List[int]],
                                  all_costs: List[float]) -> np.ndarray:
        """Update pheromone levels using vectorized operations"""
        # Global evaporation
        pheromone *= (1.0 - self.base_evaporation)

        # Calculate deposits
        deposits = 1.0 / (np.array(all_costs) + 1e-10)

        # Update pheromone matrix efficiently
        for path, deposit in zip(all_paths, deposits):
            path_array = np.array(path)
            # Use advanced indexing for efficient updates
            idx_from = path_array[:-1]
            idx_to = path_array[1:]
            pheromone[idx_from, idx_to] += deposit
            pheromone[idx_to, idx_from] += deposit  # Symmetric update

        return pheromone

    def solve(self,
             points: np.ndarray,
             route_nodes: List[int],
             n_iterations: int = 100,
             time_windows: Optional[Dict[int, TimeWindow]] = None,
             conductivities: Optional[Dict] = None,
             demands: Optional[List[float]] = None,
             capacity: Optional[float] = None) -> Tuple[List[int], float, Dict[int, float]]:
        """Solve TSP/VRP with vectorized operations and parallel processing"""
        import streamlit as st

        st.write("\n=== ACO Solver Starting ===")
        st.write(f"Number of points: {len(points)}")

        # Precompute nearest neighbors
        self.precompute_nearest_neighbors(points)

        # Initialize pheromone matrix
        n_points = len(points)
        pheromone = self.initialize_pheromone(n_points, conductivities)

        # Calculate dynamic number of ants based on problem size
        self.n_ants = min(
            int(self.base_ants * np.log2(n_points / 10 + 1)),
            200  # Maximum ants cap
        )

        best_path = None
        best_cost = float('inf')
        best_arrival_times = {}
        previous_best_cost = float('inf')
        stagnation_counter = 0

        # Determine optimal chunk size for parallel processing
        chunk_size = max(1, min(self.n_ants // (self.max_parallel_ants * 2), 20))

        for iteration in range(n_iterations):
            iteration_start = time.time()

            # Parallel solution construction
            executor_class = ProcessPoolExecutor if self.n_ants > 50 else ProcessPoolExecutor
            with executor_class(max_workers=self.max_parallel_ants) as executor:
                # Prepare arguments for parallel processing
                args_list = [(points, pheromone, route_nodes)] * self.n_ants

                # Submit tasks in chunks
                futures = []
                for i in range(0, self.n_ants, chunk_size):
                    chunk_args = args_list[i:i + chunk_size]
                    futures.extend([executor.submit(self.construct_solution_parallel, args) 
                                 for args in chunk_args])

                all_paths = []
                all_costs = []
                all_arrival_times = []

                for future in as_completed(futures):
                    try:
                        path = future.result()
                        arrival_times = self.calculate_arrival_times(path, self.distance_matrix, time_windows)
                        base_cost = np.sum([self.distance_matrix[path[i]][path[i+1]] 
                                          for i in range(len(path)-1)])

                        # Apply penalties if constraints are enabled
                        penalty = 0.0
                        if time_windows:
                            for node, arrival in arrival_times.items():
                                if node in time_windows:
                                    tw = time_windows[node]
                                    if arrival > tw.latest:
                                        penalty += (arrival - tw.latest) * self.time_penalty_factor

                        if demands is not None and capacity is not None:
                            route_demand = sum(demands[i] for i in path[1:-1])
                            if route_demand > capacity:
                                penalty += (route_demand - capacity) * self.base_time_penalty

                        total_cost = base_cost + penalty

                        all_paths.append(path)
                        all_costs.append(total_cost)
                        all_arrival_times.append(arrival_times)

                    except Exception as e:
                        st.write(f"Error in ant construction: {str(e)}")
                        continue

            if all_costs:
                min_cost_idx = np.argmin(all_costs)
                if all_costs[min_cost_idx] < best_cost:
                    best_path = all_paths[min_cost_idx]
                    best_cost = all_costs[min_cost_idx]
                    best_arrival_times = all_arrival_times[min_cost_idx]
                    stagnation_counter = 0

                    improvement = ((previous_best_cost - best_cost) / previous_best_cost 
                                if previous_best_cost != float('inf') else 1.0)
                    if improvement > 0.05:  # Log only significant improvements
                        st.write(f"Cost improved by {improvement*100:.1f}%")
                else:
                    stagnation_counter += 1

                previous_best_cost = best_cost

                # Update pheromones using vectorized operations
                pheromone = self.update_pheromone_vectorized(pheromone, all_paths, all_costs)

            # Adapt parameters
            self.adapt_parameters(iteration, best_cost, previous_best_cost, len(points))

            # Early stopping check
            if stagnation_counter >= self.stagnation_limit:
                st.write(f"Stopping early due to stagnation at iteration {iteration}")
                break

        return best_path, best_cost, best_arrival_times

    def calculate_time_window_penalty(self,
                                   arrival_time: float,
                                   time_window: TimeWindow) -> float:
        """Calculate clamped penalty for time window violation."""
        if arrival_time < time_window.earliest:
            return 0
        elif arrival_time > time_window.latest:
            lateness = min(arrival_time - time_window.latest, 1000.0)  # Clamp maximum lateness
            base_penalty = lateness * self.time_penalty_factor
            exp_factor = min(1.0 + math.log2(1 + lateness), 10.0)  # Clamp exponential factor
            return base_penalty * exp_factor
        return 0

    def adapt_parameters(self, 
                        iteration: int,
                        current_cost: float,
                        previous_best: float,
                        problem_size: int) -> None:
        """Adapt ACO parameters based on solution improvement and problem size."""
        if current_cost < previous_best:
            self.last_improvement_iteration = iteration
            # Scale ant count with problem size
            self.n_ants = min(
                self.base_ants * int(math.log2(problem_size / 100 + 1)),
                200  # Maximum ants cap
            )
            current_evaporation = self.base_evaporation
        else:
            stagnation_time = iteration - self.last_improvement_iteration
            if stagnation_time >= self.stagnation_limit:
                # Increase exploration
                self.n_ants = min(self.n_ants * 2, 200)
                current_evaporation = min(
                    self.base_evaporation + (stagnation_time / 10) * self.evap_increase,
                    0.9
                )

                # Dynamic parameter adaptation
                self.q0 = max(0.01, self.q0 * 0.95)  # Reduce exploitation
                self.alpha = max(0.5, self.alpha * 0.98)  # Reduce pheromone influence
                self.beta = min(5.0, self.beta * 1.02)  # Increase heuristic influence

                # Adjust ALNS parameters
                self.time_penalty_factor = min(5.0, self.time_penalty_factor * 1.05)

    def apply_alns(self,
                   current_route: List[int], 
                   distances: np.ndarray,
                   time_windows: Dict[int, TimeWindow],
                   removal_ratio: float = 0.15,  # Reduced from 0.2
                   min_improvement: float = 0.01,  # Minimum improvement threshold
                   max_attempts: int = 3) -> Tuple[List[int], Dict[int, float]]:
        """
        Apply conservative ALNS with predictive checks and immediate reinsertion.
        """
        if len(current_route) <= 3:  # Route too small
            return current_route, self.calculate_arrival_times(
                current_route, distances, time_windows)

        # Calculate initial state
        current_arrival_times = self.calculate_arrival_times(current_route, distances, time_windows)
        current_cost = self.calculate_total_cost(current_route, distances, current_arrival_times, time_windows)
        best_route = current_route[:]
        best_cost = current_cost
        best_times = current_arrival_times

        # Identify problematic nodes
        violations = []
        total_violation = 0
        for node in current_route[1:-1]:  # Exclude depot
            if node in time_windows:
                arrival = current_arrival_times.get(node, 0)
                tw = time_windows[node]
                if arrival > tw.latest:
                    violation = arrival - tw.latest
                    violations.append((node, violation))
                    total_violation += violation

        if not violations:
            return current_route, current_arrival_times

        # Sort violations by severity
        violations.sort(key=lambda x: x[1], reverse=True)

        # Calculate adaptive threshold
        avg_violation = total_violation / len(violations) if violations else 0
        threshold = avg_violation * 0.5  # Only consider significant violations

        # Track improvements
        improvements = []
        nodes_processed = 0
        successful_moves = 0

        # Process one node at a time with immediate reinsertion
        for node, violation in violations:
            if violation < threshold:
                continue

            nodes_processed += 1

            # Predict potential benefit
            benefit_ratio = violation / current_cost
            if benefit_ratio < min_improvement:
                continue

            # Try removal and reinsertion
            temp_route = [n for n in best_route if n != node]
            insertion_attempts = []

            # Try each feasible insertion position
            for pos in range(1, len(temp_route)):
                candidate = temp_route[:pos] + [node] + temp_route[pos:]
                times = self.calculate_arrival_times(candidate, distances, time_windows)
                cost = self.calculate_total_cost(candidate, distances, times, time_windows)

                # Calculate net improvement
                improvement = best_cost - cost
                if improvement > 0:
                    insertion_attempts.append((improvement, candidate, times, cost))

            # Select best feasible insertion
            if insertion_attempts:
                insertion_attempts.sort(reverse=True)  # Sort by improvement
                improvement, new_route, new_times, new_cost = insertion_attempts[0]

                # Only accept significant improvements
                if improvement / best_cost >= min_improvement:
                    best_route = new_route
                    best_cost = new_cost
                    best_times = new_times
                    successful_moves += 1
                    improvements.append(improvement)

            # Early stopping if no progress
            if nodes_processed >= max_attempts and not improvements:
                break

        # Verify solution integrity
        if len(set(best_route[1:-1])) != len(set(current_route[1:-1])):
            return current_route, current_arrival_times

        # Log only significant improvements
        if improvements:
            total_improvement = ((current_cost - best_cost) / current_cost) * 100
            if total_improvement > 1.0:  # Only log >1% improvements
                import streamlit as st
                st.write(f"ALNS: {successful_moves}/{nodes_processed} moves, "
                        f"cost reduction: {total_improvement:.1f}%")

        return best_route, best_times

    def repair_time_windows(self,
                         route: List[int],
                         distances: np.ndarray,
                         time_windows: Dict[int, TimeWindow],
                         max_iterations: int = 50,
                         cost_threshold: float = 1.5) -> Tuple[List[int], Dict[int, float]]:
        """Repair time window violations with minimal logging."""
        if len(route) <= 2:
            return route, {0: 0.0}

        best_route = route.copy()
        best_arrival_times = self.calculate_arrival_times(route, distances, time_windows)
        best_violations = self.count_time_violations(best_arrival_times, time_windows)
        initial_cost = self.calculate_total_cost(route, distances, best_arrival_times, time_windows)
        best_cost = initial_cost

        iteration = 0
        while iteration < max_iterations and best_violations > 0:
            iteration += 1
            improvement_found = False

            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_arrival_times = self.calculate_arrival_times(new_route, distances, time_windows)
                    new_violations = self.count_time_violations(new_arrival_times, time_windows)
                    new_cost = self.calculate_total_cost(new_route, distances, new_arrival_times, time_windows)

                    if (new_violations < best_violations or 
                        (new_violations == best_violations and new_cost < best_cost * cost_threshold)):
                        improvement_found = True
                        best_route = new_route
                        best_arrival_times = new_arrival_times
                        best_violations = new_violations
                        best_cost = new_cost
                        break
                if improvement_found:
                    break
            if not improvement_found:
                break

        return best_route, best_arrival_times

    def count_time_violations(self,
                           arrival_times: Dict[int, float],
                           time_windows: Dict[int, TimeWindow]) -> int:
        """Count number of time window violations in route."""
        violations = 0
        for node, arrival in arrival_times.items():
            if node in time_windows:
                tw = time_windows[node]
                if arrival > tw.latest:
                    violations += 1
        return violations

    def calculate_arrival_times(self,
                             route: List[int],
                             distances: np.ndarray,
                             time_windows: Optional[Dict[int, TimeWindow]] = None) -> Dict[int, float]:
        """Calculate arrival times at each node."""
        arrival_times = {0: 0.0}  # Start at depot
        current_time = 0.0

        for i in range(len(route) - 1):
            current = route[i]
            next_node = route[i + 1]

            # Add travel time
            travel_time = distances[current][next_node] / self.speed

            # Add service and waiting time if time windows exist
            if time_windows and current in time_windows:
                tw = time_windows[current]
                current_time = tw.get_departure_time(current_time)

            current_time += travel_time
            arrival_times[next_node] = current_time

        return arrival_times


    def calculate_total_cost(self,
                           path: List[int],
                           distances: np.ndarray,
                           arrival_times: Dict[int, float],
                           time_windows: Optional[Dict[int, TimeWindow]] = None) -> float:
        """Calculate total cost including distance and time window penalties."""
        # Base distance cost
        distance_cost = sum(distances[path[i]][path[i+1]] 
                              for i in range(len(path)-1))

        # Time window penalties
        if not time_windows:
            return distance_cost

        time_penalties = 0
        # Arrival times already use global indices, no conversion needed
        for node, arrival_time in arrival_times.items():
            if node in time_windows:
                penalty = self.calculate_time_window_penalty(
                    arrival_time, time_windows[node])
                time_penalties += penalty

        return distance_cost + time_penalties

    def update_pheromone(self,
                        pheromone: np.ndarray,
                        all_paths: List[List[int]],
                        all_costs: List[float],
                        current_evaporation: float) -> np.ndarray:
        """Update pheromone levels using vectorized operations."""
        # Global evaporation
        pheromone *= (1.0 - current_evaporation)

        # Vectorized deposit calculation
        deposits = 1.0 / (np.array(all_costs) + 1e-10)

        # Update pheromone matrix efficiently
        for path, deposit in zip(all_paths, deposits):
            path_array = np.array(path)
            start_nodes = path_array[:-1]
            end_nodes = path_array[1:]
            pheromone[start_nodes, end_nodes] += deposit
            pheromone[end_nodes, start_nodes] += deposit  # Symmetric update

        return pheromone

    def log(self, message: str, force: bool = False) -> None:
        """Conditional logging based on verbosity"""
        if self.VERBOSE or force:
            import streamlit as st
            st.write(message)

    def verify_and_fix_routes(self, routes: List[List[int]], 
                         n_points: int,
                         distances: np.ndarray,
                         demands: List[float],
                         capacity: float,
                         time_windows: Optional[Dict[int, TimeWindow]] = None,
                         speed: float = 1.0,
                         max_repair_iterations: int = 5,
                         cost_increase_threshold: float = 0.2,
                         time_penalty_multiplier: float = 3.0) -> List[List[int]]:
        """
        Verify all nodes are included in routes and fix any missing nodes.
        Args:
            routes: List of vehicle routes
            n_points: Total number of points including depot
            distances: Distance matrix
            demands: List of demands for each node
            capacity: Vehicle capacity
            time_windows: Optional time window constraints
            speed: Travel speed
            max_repair_iterations: Maximum number of repair attempts before early stopping
            cost_increase_threshold: Maximum allowed proportional cost increase for insertions
            time_penalty_multiplier: Penalty factor for time window violations
        """
        import streamlit as st

        # Remove duplicate depot entries (keep only start/end)
        for i, route in enumerate(routes):
            if not route:
                continue
            cleaned_route = [node for j, node in enumerate(route) 
                            if node != 0 or j == 0 or j == len(route)-1]
            if cleaned_route[0] != 0:
                cleaned_route.insert(0, 0)
            if cleaned_route[-1] != 0:
                cleaned_route.append(0)
            routes[i] = cleaned_route

        # Count node occurrences
        node_counts = {i: 0 for i in range(1, n_points)}  # Exclude depot
        for route in routes:
            for node in route[1:-1]:  # Skip depot at start/end
                node_counts[node] = node_counts.get(node, 0) + 1

        # Find missing and duplicate nodes
        missing_nodes = [node for node, count in node_counts.items() if count == 0]
        duplicate_nodes = [node for node, count in node_counts.items() if count > 1]

        if missing_nodes or duplicate_nodes:
            st.write(f"Found {len(missing_nodes)} missing and {len(duplicate_nodes)} duplicate nodes")

        # Remove duplicate nodes (keep first occurrence)
        if duplicate_nodes:
            for node in duplicate_nodes:
                found_first = False
                for route in routes:
                    if node in route:
                        if found_first:
                            route.remove(node)
                        else:
                            found_first = True

        # Calculate initial total cost
        def calculate_route_cost(route: List[int], with_penalties: bool = True) -> float:
            cost = sum(distances[route[i]][route[i+1]] for i in range(len(route)-1))
            if with_penalties and time_windows:
                current_time = 0.0
                for i in range(len(route)):
                    if i > 0:
                        current_time += distances[route[i-1]][route[i]] / speed
                    if route[i] in time_windows:
                        tw = time_windows[route[i]]
                        if current_time < tw.earliest:
                            current_time = tw.earliest
                        elif current_time > tw.latest:
                            cost += (current_time - tw.latest) * time_penalty_multiplier
                        current_time += tw.service_time
            return cost

        initial_total_cost = sum(calculate_route_cost(route) for route in routes)

        # Try to insert missing nodes
        for node in missing_nodes:
            best_route_idx = -1
            best_position = -1
            min_cost_increase = float('inf')
            no_improvement_count = 0

            while no_improvement_count < max_repair_iterations:
                found_improvement = False

                # Try inserting into each route
                for route_idx, route in enumerate(routes):
                    if len(route) <= 2:  # Skip empty routes
                        continue

                    # Check capacity
                    route_demand = sum(demands[i] for i in route[1:-1])
                    if route_demand + demands[node] > capacity * 1.1:  # Allow 10% overflow
                        continue

                    # Try each insertion position
                    old_route_cost = calculate_route_cost(route)
                    for pos in range(1, len(route)):
                        new_route = route[:pos] + [node] + route[pos:]
                        new_route_cost = calculate_route_cost(new_route)
                        cost_increase = new_route_cost - old_route_cost

                        # Only accept if cost increase is reasonable
                        if cost_increase < min_cost_increase and cost_increase/old_route_cost <= cost_increase_threshold:
                            min_cost_increase = cost_increase
                            best_route_idx = route_idx
                            best_position = pos
                            found_improvement = True

                if found_improvement:
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                # Early stopping if no improvements found
                if no_improvement_count >= max_repair_iterations:
                    break

            # Insert node at best position if found
            if best_route_idx >= 0:
                route = routes[best_route_idx]
                routes[best_route_idx] = route[:best_position] + [node] + route[best_position:]
            else:
                # Create new route as last resort
                if demands[node] <= capacity:
                    new_route = [0, node, 0]
                    routes.append(new_route)

        # Final verification and cost comparison
        final_total_cost = sum(calculate_route_cost(route) for route in routes)
        final_counts = {i: sum(1 for route in routes for n in route[1:-1] if n == i) 
                       for i in range(1, n_points)}

        remaining_issues = [node for node, count in final_counts.items() if count != 1]
        if remaining_issues:
            st.error(f"Warning: {len(remaining_issues)} nodes have incorrect counts")
        else:
            cost_change = ((final_total_cost - initial_total_cost) / initial_total_cost) * 100
            if cost_change > 0:
                st.warning(f"Solution cost increased by {cost_change:.1f}% during repair")
            else:
                st.success("All nodes verified with no cost increase")

        return routes

def verify_and_fix_routes(routes: List[List[int]], 
                         n_points: int,
                         distances: np.ndarray,
                         demands: List[float],
                         capacity: float,
                         time_windows: Optional[Dict[int, TimeWindow]] = None,
                         speed: float = 1.0,
                         max_repair_iterations: int = 5,
                         cost_increase_threshold: float = 0.2,
                         time_penalty_multiplier: float = 3.0) -> List[List[int]]:
    """
    Verify all nodes are included in routes and fix any missing nodes.
    Args:
        routes: List of vehicle routes
        n_points: Total number of points including depot
        distances: Distance matrix
        demands: List of demands for each node
        capacity: Vehicle capacity
        time_windows: Optional time window constraints
        speed: Travel speed
        max_repair_iterations: Maximum number of repair attempts before early stopping
        cost_increase_threshold: Maximum allowed proportional cost increase for insertions
        time_penalty_multiplier: Penalty factor for time window violations
    """
    import streamlit as st

    # Remove duplicate depot entries (keep only start/end)
    for i, route in enumerate(routes):
        if not route:
            continue
        cleaned_route = [node for j, node in enumerate(route) 
                        if node != 0 or j == 0 or j == len(route)-1]
        if cleaned_route[0] != 0:
            cleaned_route.insert(0, 0)
        if cleaned_route[-1] != 0:
            cleaned_route.append(0)
        routes[i] = cleaned_route

    # Count node occurrences
    node_counts = {i: 0 for i in range(1, n_points)}  # Exclude depot
    for route in routes:
        for node in route[1:-1]:  # Skip depot at start/end
            node_counts[node] = node_counts.get(node, 0) + 1

    # Find missing and duplicate nodes
    missing_nodes = [node for node, count in node_counts.items() if count == 0]
    duplicate_nodes = [node for node, count in node_counts.items() if count > 1]

    if missing_nodes or duplicate_nodes:
        st.write(f"Found {len(missing_nodes)} missing and {len(duplicate_nodes)} duplicate nodes")

    # Remove duplicate nodes (keep first occurrence)
    if duplicate_nodes:
        for node in duplicate_nodes:
            found_first = False
            for route in routes:
                if node in route:
                    if found_first:
                        route.remove(node)
                    else:
                        found_first = True

    # Calculate initial total cost
    def calculate_route_cost(route: List[int], with_penalties: bool = True) -> float:
        cost = sum(distances[route[i]][route[i+1]] for i in range(len(route)-1))
        if with_penalties and time_windows:
            current_time = 0.0
            for i in range(len(route)):
                if i > 0:
                    current_time += distances[route[i-1]][route[i]] / speed
                if route[i] in time_windows:
                    tw = time_windows[route[i]]
                    if current_time < tw.earliest:
                        current_time = tw.earliest
                    elif current_time > tw.latest:
                        cost += (currenttime - tw.latest) * time_penalty_multiplier
                        current_time += tw.service_time
        return cost

    initial_total_cost = sum(calculate_route_cost(route) for route in routes)

    # Try to insert missing nodes
    for node in missing_nodes:
        best_route_idx = -1
        best_position = -1
        min_cost_increase = float('inf')
        no_improvement_count = 0

        while no_improvement_count < max_repair_iterations:
            found_improvement = False

            # Try inserting into each route
            for route_idx, route in enumerate(routes):
                if len(route) <= 2:  # Skip empty routes
                    continue

                # Check capacity
                route_demand = sum(demands[i] for i in route[1:-1])
                if route_demand + demands[node] > capacity * 1.1:  # Allow 10% overflow
                    continue

                # Try each insertion position
                old_route_cost = calculate_route_cost(route)
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [node] + route[pos:]
                    new_route_cost = calculate_route_cost(new_route)
                    cost_increase = new_route_cost - old_route_cost

                    # Only accept if cost increase is reasonable
                    if cost_increase < min_cost_increase and cost_increase/old_route_cost <= cost_increase_threshold:
                        min_cost_increase = cost_increase
                        best_route_idx = route_idx
                        best_position = pos
                        found_improvement = True

            if found_improvement:
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Early stopping if no improvements found
            if no_improvement_count >= max_repair_iterations:
                break

        # Insert node at best position if found
        if best_route_idx >= 0:
            route = routes[best_route_idx]
            routes[best_route_idx] = route[:best_position] + [node] + route[best_position:]
        else:
            # Create new route as last resort
            if demands[node] <= capacity:
                new_route = [0, node, 0]
                routes.append(new_route)

    # Final verification and cost comparison
    final_total_cost = sum(calculate_route_cost(route) for route in routes)
    final_counts = {i: sum(1 for route in routes for n in route[1:-1] if n == i) 
                   for i in range(1, n_points)}

    remaining_issues = [node for node, count in final_counts.items() if count != 1]
    if remaining_issues:
        st.error(f"Warning: {len(remaining_issues)} nodes have incorrect counts")
    else:
        cost_change = ((final_total_cost - initial_total_cost) / initial_total_cost) * 100
        if cost_change > 0:
            st.warning(f"Solution cost increased by {cost_change:.1f}% during repair")
        else:
            st.success("All nodes verified with no cost increase")

    return routes

def solve(self,
         points: np.ndarray,
         route_nodes: List[int],
         n_iterations: int = 100,
         time_windows: Optional[Dict[int, TimeWindow]] = None,
         conductivities: Optional[Dict] = None,
         demands: Optional[List[float]] = None,
         capacity: Optional[float] = None) -> Tuple[List[int], float, Dict[int, float]]:
    """Solve TSP/VRP with vectorized operations and parallel processing"""
    import streamlit as st

    st.write("\n=== ACO Solver Starting ===")
    st.write(f"Number of points: {len(points)}")

    # Precompute nearest neighbors
    self.precompute_nearest_neighbors(points)

    # Initialize pheromone matrix
    n_points = len(points)
    pheromone = self.initialize_pheromone(n_points, conductivities)

    # Calculate dynamic number of ants based on problem size
    self.n_ants = min(
        int(self.base_ants * np.log2(n_points / 10 + 1)),
        200  # Maximum ants cap
    )

    best_path = None
    best_cost = float('inf')
    best_arrival_times = {}
    previous_best_cost = float('inf')
    stagnation_counter = 0

    # Determine optimal chunk size for parallel processing
    chunk_size = max(1, min(self.n_ants // (self.max_parallel_ants * 2), 20))

    for iteration in range(n_iterations):
        iteration_start = time.time()

        # Parallel solution construction
        executor_class = ProcessPoolExecutor if self.n_ants > 50 else ProcessPoolExecutor
        with executor_class(max_workers=self.max_parallel_ants) as executor:
            # Prepare arguments for parallel processing
            args_list = [(points, pheromone, route_nodes)] * self.n_ants

            # Submit tasks in chunks
            futures = []
            for i in range(0, self.n_ants, chunk_size):
                chunk_args = args_list[i:i + chunk_size]
                futures.extend([executor.submit(self.construct_solution_parallel, args) 
                             for args in chunk_args])

            all_paths = []
            all_costs = []
            all_arrival_times = []

            for future in as_completed(futures):
                try:
                    path = future.result()
                    arrival_times = self.calculate_arrival_times(path, self.distance_matrix, time_windows)
                    base_cost = np.sum([self.distance_matrix[path[i]][path[i+1]] 
                                      for i in range(len(path)-1)])

                    # Apply penalties if constraints are enabled
                    penalty = 0.0
                    if time_windows:
                        for node, arrival in arrival_times.items():
                            if node in time_windows:
                                tw = time_windows[node]
                                if arrival > tw.latest:
                                    penalty += (arrival - tw.latest) * self.time_penalty_factor

                    if demands is not None and capacity is not None:
                        route_demand = sum(demands[i] for i in path[1:-1])
                        if route_demand > capacity:
                            penalty += (route_demand - capacity) * self.base_time_penalty

                    total_cost = base_cost + penalty

                    all_paths.append(path)
                    all_costs.append(total_cost)
                    all_arrival_times.append(arrival_times)

                except Exception as e:
                    st.write(f"Error in ant construction: {str(e)}")
                    continue

        if all_costs:
            min_cost_idx = np.argmin(all_costs)
            if all_costs[min_cost_idx] < best_cost:
                best_path = all_paths[min_cost_idx]
                best_cost = all_costs[min_cost_idx]
                best_arrival_times = all_arrival_times[min_cost_idx]
                stagnation_counter = 0

                improvement = ((previous_best_cost - best_cost) / previous_best_cost 
                            if previous_best_cost != float('inf') else 1.0)
                if improvement > 0.05:  # Log only significant improvements
                    st.write(f"Cost improved by {improvement*100:.1f}%")
            else:
                stagnation_counter += 1

            previous_best_cost = best_cost

            # Update pheromones using vectorized operations
            pheromone = self.update_pheromone_vectorized(pheromone, all_paths, all_costs)

        # Adapt parameters
        self.adapt_parameters(iteration, best_cost, previous_best_cost, len(points))

        # Early stopping check
        if stagnation_counter >= self.stagnation_limit:
            st.write(f"Stopping early due to stagnation at iteration {iteration}")
            break

    return best_path, best_cost, best_arrival_times
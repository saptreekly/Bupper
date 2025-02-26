import numpy as np
from typing import List, Tuple, Dict, Optional
import random
import math
from utils import TimeWindow
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

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

@dataclass
class SolutionMetrics:
    """Track solution improvement metrics"""
    iteration: int
    cost: float
    violations: int
    computation_time: float
    improvement_rate: float

class ACO:
    # Class-level debug flag
    VERBOSE = False  # Set to True for detailed logging

    def __init__(self, 
                 base_ants: int = 40,
                 base_evaporation: float = 0.15,
                 alpha: float = 1.5,
                 beta: float = 2.5,
                 q0: float = 0.1,
                 evap_increase: float = 0.05,
                 stagnation_limit: int = 5,
                 speed: float = 1.0,
                 time_penalty_factor: float = 2.0,  # Penalty multiplier for time window violations
                 base_time_penalty: float = 2.0,    # Base penalty for time window violations
                 lateness_multiplier: float = 1.5,  # Multiplier for increasing penalties
                 max_parallel_ants: int = 4,        # Maximum number of parallel ant constructions
                 verbose: bool = False):            # Control logging verbosity
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

    def log(self, message: str, force: bool = False) -> None:
        """Conditional logging based on verbosity"""
        if self.VERBOSE or force:
            import streamlit as st
            st.write(message)

    def calculate_distances(self, points: np.ndarray) -> np.ndarray:
        """Calculate distance matrix between all points using vectorized operations."""
        return np.sqrt(np.sum((points[:, np.newaxis] - points) ** 2, axis=2))

    def initialize_pheromone(self, n_points: int) -> np.ndarray:
        """Initialize pheromone matrix."""
        return np.ones((n_points, n_points)) * self.q0

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
                        previous_best: float) -> None:
        """Adapt ACO parameters based on solution improvement."""
        if current_cost < previous_best:
            self.last_improvement_iteration = iteration
            self.n_ants = self.base_ants
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


    def construct_solution(self, 
                     route_nodes: List[int],
                     distances: np.ndarray,
                     pheromone: np.ndarray) -> List[int]:
        """Construct a single ant's solution with minimal logging."""
        unvisited = set(route_nodes[1:])  # Skip depot
        current = 0  # Start at depot
        path = [current]

        while unvisited:
            moves = []
            probs = []

            for j in unvisited:
                # Calculate move probability components
                pheromone_value = pheromone[current][j] ** self.alpha
                distance_value = max(distances[current][j], 1e-6)  # Avoid division by zero
                distance_factor = (1.0 / distance_value) ** self.beta
                prob = pheromone_value * distance_factor

                # Handle non-finite probabilities
                if not np.isfinite(prob) or np.isnan(prob):
                    prob = 1e-6

                moves.append(j)
                probs.append(prob)

            # Normalize probabilities
            total = sum(probs)
            if total == 0 or not np.isfinite(total):
                normalized_probs = [1.0 / len(probs)] * len(probs)
            else:
                normalized_probs = [max(1e-6, p/total) for p in probs]
                # Renormalize if needed
                total = sum(normalized_probs)
                normalized_probs = [p/total for p in normalized_probs]

            # Select next city
            selected_idx = random.choices(
                population=range(len(moves)),
                weights=normalized_probs
            )[0]
            next_city = moves[selected_idx]
            path.append(next_city)
            unvisited.remove(next_city)
            current = next_city

        path.append(0)  # Return to depot
        return path

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
        """Update pheromone levels based on solution quality."""
        pheromone *= (1.0 - current_evaporation)

        for path, cost in zip(all_paths, all_costs):
            deposit = 1.0 / (cost + 1e-10)  # Avoid division by zero
            for i in range(len(path) - 1):
                current = path[i]
                next_city = path[i + 1]
                pheromone[current][next_city] += deposit
                pheromone[next_city][current] += deposit

        return pheromone

    def solve(self, 
             points: np.ndarray,
             route_nodes: List[int],
             demands: List[float],
             capacity: float,
             n_iterations: int = 100,
             time_windows: Optional[Dict[int, TimeWindow]] = None,
             alns_frequency: int = 10) -> Tuple[List[int], float, Dict[int, float]]:
        """
        Solve TSP using adaptive ACO with time windows and repair step
        """
        import streamlit as st

        # Log initial route nodes
        st.write("\n=== ACO Solver Starting ===")
        st.write(f"Initial route nodes: {route_nodes}")
        delivery_nodes = set(route_nodes[1:])  # Exclude depot
        st.write(f"Number of delivery points: {len(delivery_nodes)}")

        start_time = time.time()
        distances = self.calculate_distances(points)
        n_points = len(points)
        self.n_ants = int(math.log2(n_points) * self.base_ants)
        pheromone = self.initialize_pheromone(n_points)
        self.demands = demands
        self.capacity = capacity

        best_path = None
        best_cost = float('inf')
        best_arrival_times = {}
        previous_best_cost = float('inf')
        initial_best_cost = float('inf')
        initial_violations = 0

        # Track node inclusion
        node_inclusion_count = {node: 0 for node in delivery_nodes}
        skipped_nodes = set()
        constraint_violations = {
            'capacity': set(),
            'time_window': set(),
            'other': set()
        }

        for iteration in range(n_iterations):
            iteration_start = time.time()

            # Parallel ant solution construction
            with ThreadPoolExecutor(max_workers=self.max_parallel_ants) as executor:
                future_to_ant = {
                    executor.submit(
                        self.construct_solution, route_nodes, distances, pheromone
                    ): i for i in range(self.n_ants)
                }

                all_paths = []
                all_costs = []
                all_arrival_times = []

                for future in as_completed(future_to_ant):
                    try:
                        path = future.result()
                        arrival_times = self.calculate_arrival_times(path, distances, time_windows)
                        cost = self.calculate_total_cost(path, distances, arrival_times, time_windows)

                        # Track node inclusion
                        included_nodes = set(path[1:-1])  # Exclude depot
                        for node in included_nodes:
                            node_inclusion_count[node] = node_inclusion_count.get(node, 0) + 1

                        # Track constraint violations
                        if time_windows:
                            for node, arrival in arrival_times.items():
                                if node in time_windows:
                                    tw = time_windows[node]
                                    if arrival > tw.latest:
                                        constraint_violations['time_window'].add(node)

                        route_demand = sum(demands[i] for i in path[1:-1])
                        if route_demand > capacity:
                            constraint_violations['capacity'].update(path[1:-1])

                        all_paths.append(path)
                        all_costs.append(cost)
                        all_arrival_times.append(arrival_times)

                    except Exception as e:
                        self.log(f"Error in ant construction: {str(e)}")
                        continue

            # Store first solution metrics after parallel construction
            if iteration == 0 and all_paths:
                initial_best_cost = min(all_costs)
                best_arrival_times_idx = all_costs.index(initial_best_cost)
                initial_violations = self.count_time_violations(
                    all_arrival_times[best_arrival_times_idx], time_windows)
                st.write(f"Initial solution - Cost: {initial_best_cost:.2f}, "
                        f"Violations: {initial_violations}")

            # Update best solution
            min_cost_idx = np.argmin(all_costs)
            if all_costs[min_cost_idx] < best_cost:
                best_path = all_paths[min_cost_idx]
                best_cost = all_costs[min_cost_idx]
                best_arrival_times = all_arrival_times[min_cost_idx]

                # Log significant improvements (>5%)
                improvement = (previous_best_cost - best_cost) / previous_best_cost
                if improvement > 0.05:  # 5% threshold
                    self.log(f"Cost improved by {improvement*100:.1f}%", force=True)

            # Adapt parameters and apply ALNS
            self.adapt_parameters(iteration, best_cost, previous_best_cost)
            if iteration % alns_frequency == 0 and best_path is not None:
                improved_path, improved_times = self.apply_alns(best_path, distances, time_windows)
                improved_cost = self.calculate_total_cost(improved_path, distances, improved_times, time_windows)

                # Track nodes affected by ALNS
                before_nodes = set(best_path[1:-1])
                after_nodes = set(improved_path[1:-1])
                if before_nodes != after_nodes:
                    added = after_nodes - before_nodes
                    removed = before_nodes - after_nodes
                    if added:
                        st.write(f"ALNS added nodes: {added}")
                    if removed:
                        st.write(f"ALNS removed nodes: {removed}")

                if improved_cost < best_cost:
                    best_path = improved_path
                    best_cost = improved_cost
                    best_arrival_times = improved_times

            previous_best_cost = best_cost

            # Update pheromone
            pheromone = self.update_pheromone(pheromone, all_paths, all_costs, self.base_evaporation)

        # Analyze node inclusion
        st.write("\n=== Node Inclusion Analysis ===")
        never_included = {node for node, count in node_inclusion_count.items() if count == 0}
        rarely_included = {node for node, count in node_inclusion_count.items() 
                          if 0 < count < n_iterations * self.n_ants * 0.1}  # Less than 10% inclusion

        if never_included:
            st.write(f"\nNodes never included in any solution: {never_included}")
        if rarely_included:
            st.write(f"\nNodes rarely included (<10% of solutions): {rarely_included}")

        if constraint_violations['time_window']:
            st.write(f"\nNodes with time window violations: {constraint_violations['time_window']}")
        if constraint_violations['capacity']:
            st.write(f"\nNodes contributing to capacity violations: {constraint_violations['capacity']}")

        # Verify all nodes are included in final solution
        if best_path:
            st.write("\n=== Final Solution Analysis ===")
            final_nodes = set(best_path[1:-1])
            missing_nodes = delivery_nodes - final_nodes
            if missing_nodes:
                st.write(f"Missing nodes in final solution: {missing_nodes}")
                best_path = verify_and_fix_routes(
                    [best_path], len(points), distances, 
                    self.demands, self.capacity, time_windows, self.speed)[0]
                # Recalculate metrics after fixes
                best_arrival_times = self.calculate_arrival_times(best_path, distances, time_windows)
                best_cost = self.calculate_total_cost(best_path, distances, best_arrival_times, time_windows)
                st.write("Applied fixes to include missing nodes")

        # Print final summary
        total_time = time.time() - start_time
        final_violations = self.count_time_violations(best_arrival_times, time_windows)

        st.write("\n=== Optimization Summary ===")
        st.write(f"Initial cost: {initial_best_cost:.2f}")
        st.write(f"Final cost: {best_cost:.2f}")
        st.write(f"Initial violations: {initial_violations}")
        st.write(f"Final violations: {final_violations}")
        st.write(f"Total computation time: {total_time:.2f}s")
        st.write(f"Average iteration time: {total_time/n_iterations:.2f}s")

        if final_violations < initial_violations:
            st.success(f"Reduced violations by {initial_violations - final_violations}")
        elif final_violations > 0:
            st.warning(f"Remaining violations: {final_violations}")

        return best_path, best_cost, best_arrival_times
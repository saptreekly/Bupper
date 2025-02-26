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
                         speed: float = 1.0) -> List[List[int]]:
    """
    Verify all nodes are included in routes and fix any missing nodes.
    """
    # Collect all assigned nodes (excluding depot)
    assigned = set()
    for route in routes:
        assigned.update(set(route[1:-1]))  # Exclude depot (0) at start and end

    # Find missing nodes
    all_nodes = set(range(1, n_points))  # All nodes except depot
    missing = all_nodes - assigned

    if missing:
        import streamlit as st
        st.write(f"\n=== Missing Nodes Detected ===")
        st.write(f"Found {len(missing)} unassigned nodes: {missing}")

        # Try to insert each missing node
        for node in missing:
            best_route_idx = -1
            best_position = -1
            min_cost_increase = float('inf')

            # Try inserting into each route
            for route_idx, route in enumerate(routes):
                if len(route) <= 2:  # Skip empty routes (depot-depot)
                    continue

                # Check capacity constraint first
                route_demand = sum(demands[i] for i in route[1:-1])
                if route_demand + demands[node] > capacity:
                    continue

                # Try each insertion position (excluding start/end depot positions)
                for pos in range(1, len(route)):
                    # Create candidate route
                    new_route = route[:pos] + [node] + route[pos:]

                    # Calculate cost increase
                    old_cost = sum(distances[route[i]][route[i+1]] 
                                 for i in range(len(route)-1))
                    new_cost = sum(distances[new_route[i]][new_route[i+1]] 
                                 for i in range(len(new_route)-1))
                    cost_increase = new_cost - old_cost

                    # Check time window feasibility
                    feasible = True
                    if time_windows:
                        arrival_times = {}
                        current_time = 0.0

                        for i in range(len(new_route) - 1):
                            current = new_route[i]
                            next_node = new_route[i + 1]
                            travel_time = distances[current][next_node] / speed

                            if current in time_windows:
                                tw = time_windows[current]
                                if current_time < tw.earliest:
                                    current_time = tw.earliest
                                elif current_time > tw.latest:
                                    feasible = False
                                    break
                                current_time += tw.service_time

                            current_time += travel_time
                            arrival_times[next_node] = current_time

                    if feasible and cost_increase < min_cost_increase:
                        min_cost_increase = cost_increase
                        best_route_idx = route_idx
                        best_position = pos

            if best_route_idx >= 0:
                # Insert node at best position
                route = routes[best_route_idx]
                routes[best_route_idx] = (
                    route[:best_position] + [node] + route[best_position:]
                )
                st.write(f"Inserted node {node} into route {best_route_idx} "
                        f"at position {best_position}")
                st.write(f"Cost increase: {min_cost_increase:.2f}")
            else:
                st.error(f"Could not find feasible insertion for node {node}!")

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
                self.n_ants = min(self.n_ants * 2, 200)  # Cap maximum ants
                current_evaporation = min(
                    self.base_evaporation + (stagnation_time / 10) * self.evap_increase,
                    0.9
                )

    def apply_alns(self,
                  current_route: List[int],
                  distances: np.ndarray,
                  time_windows: Dict[int, TimeWindow],
                  removal_count: int = 3) -> Tuple[List[int], Dict[int, float]]:
        """Apply Adaptive Large Neighborhood Search to improve solution."""
        if len(current_route) <= removal_count + 2:  # +2 for depot visits
            return current_route, self.calculate_arrival_times(
                current_route, distances, time_windows)

        # Select random nodes to remove (excluding depot)
        removable = current_route[1:-1]  # Exclude first and last depot visits
        if len(removable) <= removal_count:
            return current_route, self.calculate_arrival_times(
                current_route, distances, time_windows)

        remove_indices = random.sample(range(len(removable)), removal_count)
        removed_nodes = [removable[i] for i in remove_indices]

        # Create partial route
        remaining_route = [n for i, n in enumerate(current_route) 
                         if n not in removed_nodes]

        # Try different insertion positions
        best_route = remaining_route
        best_cost = float('inf')
        best_arrival_times = None

        for node in removed_nodes:
            current_best_pos = None
            current_best_cost = float('inf')

            # Try inserting at each position (excluding depot positions)
            for pos in range(1, len(remaining_route)):
                candidate_route = (
                    remaining_route[:pos] + [node] + remaining_route[pos:]
                )
                arrival_times = self.calculate_arrival_times(
                    candidate_route, distances, time_windows)
                cost = self.calculate_total_cost(
                    candidate_route, distances, arrival_times, time_windows)

                if cost < current_best_cost:
                    current_best_pos = pos
                    current_best_cost = cost

            if current_best_pos is not None:
                remaining_route = (
                    remaining_route[:current_best_pos] + 
                    [node] + 
                    remaining_route[current_best_pos:]
                )

                # Update best solution if improved
                arrival_times = self.calculate_arrival_times(
                    remaining_route, distances, time_windows)
                cost = self.calculate_total_cost(
                    remaining_route, distances, arrival_times, time_windows)

                if cost < best_cost:
                    best_route = remaining_route.copy()
                    best_cost = cost
                    best_arrival_times = arrival_times

        return best_route, best_arrival_times

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
        start_time = time.time()
        distances = self.calculate_distances(points)
        n_points = len(points)
        self.n_ants = int(math.log2(n_points) * self.base_ants)
        pheromone = self.initialize_pheromone(n_points)

        best_path = None
        best_cost = float('inf')
        best_arrival_times = {}
        previous_best_cost = float('inf')

        # For logging improvements
        initial_best_cost = float('inf')
        initial_violations = 0

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

                        all_paths.append(path)
                        all_costs.append(cost)
                        all_arrival_times.append(arrival_times)

                    except Exception as e:
                        self.log(f"Error in ant construction: {str(e)}")
                        continue

            # Store first solution metrics
            if iteration == 0 and all_paths:
                initial_best_cost = min(all_costs)
                initial_violations = self.count_time_violations(
                    all_arrival_times[all_costs.index(initial_best_cost)],
                    time_windows)

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

                if improved_cost < best_cost:
                    best_path = improved_path
                    best_cost = improved_cost
                    best_arrival_times = improved_times

            previous_best_cost = best_cost

            # Update pheromone
            pheromone = self.update_pheromone(pheromone, all_paths, all_costs, self.base_evaporation)

            # Record metrics
            iteration_time = time.time() - iteration_start
            improvement_rate = (
                (previous_best_cost - best_cost) / previous_best_cost 
                if previous_best_cost != float('inf') else 0
            )

            self.metrics.append(SolutionMetrics(
                iteration=iteration,
                cost=best_cost,
                violations=self.count_time_violations(
                    best_arrival_times, time_windows),
                computation_time=iteration_time,
                improvement_rate=improvement_rate
            ))

        # Verify all nodes are included
        if hasattr(self, 'demands') and hasattr(self, 'capacity'):
            best_path = verify_and_fix_routes(
                [best_path], len(points), distances, 
                self.demands, self.capacity, time_windows, self.speed)[0]
            # Recalculate metrics after fixes
            best_arrival_times = self.calculate_arrival_times(best_path, distances, time_windows)
            best_cost = self.calculate_total_cost(best_path, distances, best_arrival_times, time_windows)

        # Print final summary
        total_time = time.time() - start_time
        final_violations = self.count_time_violations(best_arrival_times, time_windows)

        import streamlit as st
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
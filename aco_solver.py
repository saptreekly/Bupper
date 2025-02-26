import numpy as np
from typing import List, Tuple, Dict, Optional
import random
import math
from utils import TimeWindow

class ACO:
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
                 lateness_multiplier: float = 1.5): # Multiplier for increasing penalties
        self.base_ants = base_ants
        self.base_evaporation = base_evaporation
        self.alpha = alpha
        self.beta = beta
        self.q0 = q0
        self.evap_increase = evap_increase
        self.stagnation_limit = stagnation_limit
        self.speed = speed
        self.time_penalty_factor = time_penalty_factor  # Store the penalty factor
        self.base_time_penalty = base_time_penalty
        self.lateness_multiplier = lateness_multiplier

    def calculate_distances(self, points: np.ndarray) -> np.ndarray:
        """Calculate distance matrix between all points."""
        n = len(points)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                distances[i][j] = np.sqrt(np.sum((points[i] - points[j]) ** 2))
        return distances

    def initialize_pheromone(self, n_points: int) -> np.ndarray:
        """Initialize pheromone matrix."""
        return np.ones((n_points, n_points)) * self.q0

    def calculate_time_window_penalty(self,
                                   arrival_time: float,
                                   time_window: TimeWindow) -> float:
        """Calculate adaptive penalty for time window violation."""
        if arrival_time < time_window.earliest:
            # No penalty for early arrival (will wait)
            return 0
        elif arrival_time > time_window.latest:
            # Exponentially increasing penalty based on lateness
            lateness = arrival_time - time_window.latest
            penalty_factor = self.base_time_penalty * (self.lateness_multiplier ** math.log2(1 + lateness))
            return lateness * penalty_factor
        return 0

    def repair_time_windows(self,
                         route: List[int],
                         distances: np.ndarray,
                         time_windows: Dict[int, TimeWindow]) -> Tuple[List[int], Dict[int, float]]:
        """
        Attempt to repair time window violations using a greedy approach.
        Returns improved route and arrival times.
        """
        if len(route) <= 2:  # Nothing to repair for routes with just depot
            return route, {0: 0.0}

        best_route = route.copy()
        best_arrival_times = self.calculate_arrival_times(route, distances, time_windows)
        best_violations = self.count_time_violations(best_arrival_times, time_windows)

        # Try simple swaps to reduce violations
        improvement = True
        while improvement and best_violations > 0:
            improvement = False
            # Skip depot (index 0) in swaps
            for i in range(1, len(route) - 1):
                for j in range(i + 1, len(route) - 1):
                    # Try swapping positions i and j
                    new_route = route.copy()
                    new_route[i], new_route[j] = new_route[j], new_route[i]

                    new_arrival_times = self.calculate_arrival_times(
                        new_route, distances, time_windows)
                    new_violations = self.count_time_violations(
                        new_arrival_times, time_windows)

                    if new_violations < best_violations:
                        best_route = new_route
                        best_arrival_times = new_arrival_times
                        best_violations = new_violations
                        improvement = True
                        break
                if improvement:
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
        """
        Construct a single ant's solution.
        """
        unvisited = set(route_nodes[1:])  # Skip depot
        current = 0  # Start at depot
        path = [current]

        while unvisited:
            moves = []
            probs = []

            for j in unvisited:
                pheromone_factor = pheromone[current][j] ** self.alpha
                distance_factor = (1.0 / distances[current][j]) ** self.beta
                prob = pheromone_factor * distance_factor
                moves.append(j)
                probs.append(prob)

            # Select next city
            if not probs:  # No moves available
                break

            total = sum(probs)
            normalized_probs = [p/total for p in probs]
            selected_idx = random.choices(
                population=range(len(moves)),
                weights=normalized_probs
            )[0]

            next_city = moves[selected_idx]
            path.append(next_city)
            unvisited.remove(next_city)
            current = next_city

        # Return to depot
        path.append(0)
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
             n_iterations: int = 100,
             time_windows: Optional[Dict[int, TimeWindow]] = None) -> Tuple[List[int], float, Dict[int, float]]:
        """
        Solve TSP using adaptive ACO with time windows and repair step
        """
        distances = self.calculate_distances(points)
        n_points = len(points)
        self.n_ants = int(math.log2(n_points) * self.base_ants)
        pheromone = self.initialize_pheromone(n_points)

        best_path = None
        best_cost = float('inf')
        best_arrival_times = {}

        # For logging improvements
        initial_best_cost = float('inf')
        initial_violations = 0

        for iteration in range(n_iterations):
            all_paths = []
            all_costs = []
            all_arrival_times = []

            # Construct solutions
            for _ in range(self.n_ants):
                path = self.construct_solution(route_nodes, distances, pheromone)
                arrival_times = self.calculate_arrival_times(path, distances, time_windows)
                cost = self.calculate_total_cost(path, distances, arrival_times, time_windows)

                # Store first solution metrics
                if iteration == 0 and not all_paths:
                    initial_best_cost = cost
                    initial_violations = self.count_time_violations(arrival_times, time_windows)

                all_paths.append(path)
                all_costs.append(cost)
                all_arrival_times.append(arrival_times)

            # Update best solution
            min_cost_idx = np.argmin(all_costs)
            if all_costs[min_cost_idx] < best_cost:
                best_path = all_paths[min_cost_idx]
                best_cost = all_costs[min_cost_idx]
                best_arrival_times = all_arrival_times[min_cost_idx]

        # Apply repair step to best solution
        if time_windows:
            original_violations = self.count_time_violations(best_arrival_times, time_windows)
            original_cost = best_cost

            repaired_path, repaired_times = self.repair_time_windows(
                best_path, distances, time_windows)
            repaired_cost = self.calculate_total_cost(
                repaired_path, distances, repaired_times, time_windows)
            repaired_violations = self.count_time_violations(repaired_times, time_windows)

            # Update if repair improved solution
            if repaired_violations < original_violations:
                best_path = repaired_path
                best_arrival_times = repaired_times
                best_cost = repaired_cost

            # Log improvements
            import streamlit as st
            st.write("\n=== Time Window Optimization Results ===")
            st.write(f"Initial cost: {initial_best_cost:.2f}")
            st.write(f"Initial violations: {initial_violations}")
            st.write(f"Pre-repair cost: {original_cost:.2f}")
            st.write(f"Pre-repair violations: {original_violations}")
            st.write(f"Final cost: {best_cost:.2f}")
            st.write(f"Final violations: {repaired_violations}")

            if repaired_violations < original_violations:
                st.success(f"Repair step reduced violations by {original_violations - repaired_violations}")
            elif repaired_violations == original_violations:
                st.info("Repair step maintained same number of violations")
            else:
                st.warning("Repair step did not improve violations")

        return best_path, best_cost, best_arrival_times
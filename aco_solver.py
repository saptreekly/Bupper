import numpy as np
from typing import List, Tuple, Dict, Optional
import random
import math
from utils import TimeWindow

class ACO:
    def __init__(self, 
                 base_ants: int = 40,  # Increased from 20 to 40
                 base_evaporation: float = 0.15,  # Increased from 0.1
                 alpha: float = 1.5,  # Increased from 1.0
                 beta: float = 2.5,  # Increased from 2.0
                 q0: float = 0.1,
                 evap_increase: float = 0.05,
                 stagnation_limit: int = 5,
                 speed: float = 1.0,
                 time_penalty_factor: float = 2.0):  # Penalty multiplier for time window violations
        """
        Initialize ACO solver with adaptive parameters

        Args:
            base_ants: Base number of ants (adjusted by problem size)
            base_evaporation: Initial evaporation rate
            alpha: Pheromone importance factor
            beta: Heuristic information importance factor
            q0: Initial pheromone value
            evap_increase: Amount to increase evaporation rate
            stagnation_limit: Iterations without improvement before adapting
            speed: Travel speed (distance/time unit)
            time_penalty_factor: Penalty multiplier for time window violations
        """
        self.base_ants = base_ants
        self.base_evaporation = base_evaporation
        self.alpha = alpha
        self.beta = beta
        self.q0 = q0
        self.evap_increase = evap_increase
        self.stagnation_limit = stagnation_limit
        self.speed = speed
        self.time_penalty_factor = time_penalty_factor
        self.arrival_times = {}  # Store arrival times for logging

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
        """Calculate penalty for time window violation."""
        if arrival_time < time_window.earliest:
            # No penalty for early arrival (will wait)
            return 0
        elif arrival_time > time_window.latest:
            # Penalty proportional to lateness
            return (arrival_time - time_window.latest) * self.time_penalty_factor
        return 0

    def construct_solution(self, 
                         start: int,
                         distances: np.ndarray,
                         pheromone: np.ndarray,
                         time_windows: Optional[Dict[int, TimeWindow]] = None,
                         local2global: Optional[List[int]] = None) -> Tuple[List[int], Dict[int, float]]:
        """
        Construct a single ant's solution considering time windows
        """
        n = len(distances)
        unvisited = set(range(n))
        unvisited.remove(start)
        current = start
        path = [current]
        current_time = 0.0
        node_arrival_times = {local2global[start] if local2global else start: current_time}

        while unvisited:
            moves = []
            probs = []

            for j in unvisited:
                travel_time = distances[current][j] / self.speed
                next_arrival = current_time + travel_time

                # Calculate base attractiveness
                pheromone_factor = pheromone[current][j] ** self.alpha
                distance_factor = (1.0 / distances[current][j]) ** self.beta

                # Convert local index to global for time window check
                global_j = local2global[j] if local2global else j

                # Add time window influence
                time_penalty = 0
                if time_windows and global_j in time_windows:
                    time_penalty = self.calculate_time_window_penalty(
                        next_arrival, time_windows[global_j])
                    # Reduce attractiveness based on time window violation
                    if time_penalty > 0:
                        distance_factor *= math.exp(-time_penalty)

                prob = pheromone_factor * distance_factor
                moves.append((j, next_arrival, time_penalty))
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

            next_city, arrival_time, _ = moves[selected_idx]
            path.append(next_city)
            global_next = local2global[next_city] if local2global else next_city
            node_arrival_times[global_next] = arrival_time

            # Update current state
            unvisited.remove(next_city)
            current = next_city
            global_current = local2global[current] if local2global else current

            if time_windows and global_current in time_windows:
                tw = time_windows[global_current]
                current_time = max(arrival_time, tw.earliest) + tw.service_time
            else:
                current_time = arrival_time

        # Return to depot (always index 0 in both local and global space)
        path.append(0)
        final_travel_time = distances[current][0] / self.speed
        node_arrival_times[0] = current_time + final_travel_time

        return path, node_arrival_times

    def calculate_total_cost(self,
                           path: List[int],
                           distances: np.ndarray,
                           arrival_times: Dict[int, float],
                           time_windows: Optional[Dict[int, TimeWindow]] = None,
                           local2global: Optional[List[int]] = None) -> float:
        """Calculate total cost including distance and time window penalties."""
        # Base distance cost
        distance_cost = sum(distances[path[i]][path[i+1]] 
                          for i in range(len(path)-1))

        # Time window penalties
        if not time_windows:
            return distance_cost

        time_penalties = 0
        for local_node, arrival_time in arrival_times.items():
            global_node = local2global[local_node] if local2global else local_node
            if global_node in time_windows:
                penalty = self.calculate_time_window_penalty(
                    arrival_time, time_windows[global_node])
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
             n_iterations: int = 100,
             time_windows: Optional[Dict[int, TimeWindow]] = None,
             local2global: Optional[List[int]] = None) -> Tuple[List[int], float, Dict[int, float]]:
        """
        Solve TSP using adaptive ACO with time windows

        Returns:
            best_path: List of indices representing best path
            best_cost: Total cost including penalties
            best_arrival_times: Dictionary of arrival times at each node
        """
        n_points = len(points)
        self.n_ants = int(math.log2(n_points) * self.base_ants)
        distances = self.calculate_distances(points)
        pheromone = self.initialize_pheromone(n_points)

        best_path = None
        best_cost = float('inf')
        best_arrival_times = {}
        current_evaporation = self.base_evaporation
        stagnation_counter = 0

        for _ in range(n_iterations):
            all_paths = []
            all_costs = []
            all_arrival_times = []
            iteration_best_cost = float('inf')

            # Construct solutions
            for _ in range(self.n_ants):
                path, arrival_times = self.construct_solution(
                    0, distances, pheromone, time_windows, local2global)
                cost = self.calculate_total_cost(
                    path, distances, arrival_times, time_windows, local2global)

                all_paths.append(path)
                all_costs.append(cost)
                all_arrival_times.append(arrival_times)
                iteration_best_cost = min(iteration_best_cost, cost)

                if cost < best_cost:
                    best_cost = cost
                    best_path = path.copy()
                    best_arrival_times = arrival_times.copy()
                    stagnation_counter = 0
                    current_evaporation = self.base_evaporation

            # Update stagnation counter and evaporation rate
            if iteration_best_cost >= best_cost:
                stagnation_counter += 1
                if stagnation_counter >= self.stagnation_limit:
                    current_evaporation = min(0.9, 
                                           current_evaporation + self.evap_increase)

            # Update pheromone
            pheromone = self.update_pheromone(pheromone, all_paths, all_costs,
                                            current_evaporation)

        return best_path, best_cost, best_arrival_times
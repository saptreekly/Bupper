import numpy as np
from typing import List, Tuple, Dict, Optional
from utils import TimeWindow

def calculate_route_length(route: List[int], 
                         distances: np.ndarray) -> float:
    """Calculate total length of a route."""
    return sum(distances[route[i]][route[i+1]] 
              for i in range(len(route)-1))

def calculate_arrival_times(route: List[int],
                          distances: np.ndarray,
                          speed: float = 1.0,
                          time_windows: Optional[Dict[int, TimeWindow]] = None) -> List[float]:
    """Calculate arrival times at each node"""
    times = [0.0]  # Start at time 0
    current_time = 0.0

    for i in range(len(route)-1):
        current = route[i]
        next_node = route[i+1]

        # Add travel time
        travel_time = distances[current][next_node] / speed

        # Add service and waiting time if time windows exist
        if time_windows and current in time_windows:
            tw = time_windows[current]
            current_time = tw.get_departure_time(current_time)

        current_time += travel_time
        times.append(current_time)

    return times

def is_time_feasible(route: List[int],
                    distances: np.ndarray,
                    time_windows: Dict[int, TimeWindow],
                    speed: float = 1.0) -> bool:
    """Check if route satisfies all time windows"""
    arrival_times = calculate_arrival_times(route, distances, speed, time_windows)

    for node, time in zip(route, arrival_times):
        if node in time_windows:
            if not time_windows[node].is_feasible(time):
                return False
    return True

def three_opt_swap(route: List[int], i: int, j: int, k: int) -> List[int]:
    """Perform 3-opt swap operation."""
    new_route = route[:i]
    if k < len(route) - 1:
        new_route.extend(reversed(route[i:j + 1]))
        new_route.extend(route[j + 1:k + 1])
        new_route.extend(route[k + 1:])
    else:
        new_route.extend(reversed(route[i:j + 1]))
        new_route.extend(route[j + 1:])
    return new_route

def three_opt_improvement(route: List[int], 
                         distances: np.ndarray,
                         time_windows: Optional[Dict[int, TimeWindow]] = None,
                         speed: float = 1.0) -> Tuple[List[int], float]:
    """
    Apply 3-opt local search improvement with time windows

    Args:
        route: Initial route
        distances: Distance matrix
        time_windows: Optional time window constraints
        speed: Travel speed (distance/time unit)

    Returns:
        improved_route: Improved route
        improved_length: Length of improved route
    """
    improvement = True
    best_route = route
    best_length = calculate_route_length(route, distances)

    while improvement:
        improvement = False

        for i in range(1, len(route) - 4):
            for j in range(i + 1, len(route) - 2):
                for k in range(j + 1, len(route)):
                    new_route = three_opt_swap(best_route, i, j, k)
                    new_length = calculate_route_length(new_route, distances)

                    # Check if improvement is time-feasible
                    is_feasible = (not time_windows or 
                                 is_time_feasible(new_route, distances, 
                                                time_windows, speed))

                    if new_length < best_length and is_feasible:
                        best_route = new_route
                        best_length = new_length
                        improvement = True
                        break
                if improvement:
                    break
            if improvement:
                break

    return best_route, best_length
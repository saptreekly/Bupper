import numpy as np
from typing import List, Tuple, Dict, Optional
from utils import TimeWindow

def calculate_total_distance(routes: List[List[int]], distances: np.ndarray) -> float:
    """Calculate total distance across all routes"""
    total = 0.0
    for route in routes:
        if not route:
            continue
        for i in range(len(route) - 1):
            total += distances[route[i]][route[i + 1]]
    return total

def try_insert_node(route: List[int], 
                   node: int,
                   position: int,
                   distances: np.ndarray,
                   time_windows: Optional[Dict[int, TimeWindow]] = None,
                   speed: float = 1.0) -> Tuple[List[int], float]:
    """Try inserting a node at a specific position and calculate new length"""
    new_route = route.copy()
    new_route.insert(position, node)
    
    # Calculate new route length
    length = sum(distances[new_route[i]][new_route[i + 1]] 
                for i in range(len(new_route) - 1))
    
    # Check time window feasibility
    if time_windows:
        current_time = 0.0
        for i in range(len(new_route) - 1):
            current = new_route[i]
            next_node = new_route[i + 1]
            
            # Add travel time
            travel_time = distances[current][next_node] / speed
            
            # Check and update time with service and waiting
            if current in time_windows:
                tw = time_windows[current]
                if current_time < tw.earliest:
                    current_time = tw.earliest
                elif current_time > tw.latest:
                    return None, float('inf')  # Time window violation
                current_time += tw.service_time
            
            current_time += travel_time
    
    return new_route, length

def optimize_cross_route(routes: List[List[int]],
                        distances: np.ndarray,
                        demands: List[float],
                        capacity: float,
                        time_windows: Optional[Dict[int, TimeWindow]] = None,
                        speed: float = 1.0,
                        max_iterations: int = 100) -> Tuple[List[List[int]], float]:
    """
    Optimize routes by moving nodes between routes
    
    Args:
        routes: List of vehicle routes
        distances: Distance matrix
        demands: List of demands for each node
        capacity: Vehicle capacity
        time_windows: Optional time window constraints
        speed: Travel speed
        max_iterations: Maximum number of improvement attempts
    
    Returns:
        improved_routes: List of improved routes
        total_distance: Total distance of improved solution
    """
    best_routes = [route.copy() for route in routes]
    best_total = calculate_total_distance(routes, distances)
    improvement_found = True
    iteration = 0
    
    while improvement_found and iteration < max_iterations:
        improvement_found = False
        iteration += 1
        
        # Try moving each node to each position in each other route
        for source_idx, source_route in enumerate(best_routes):
            if not source_route or len(source_route) <= 2:  # Skip empty or minimal routes
                continue
                
            for node_idx in range(1, len(source_route) - 1):  # Skip depot
                node = source_route[node_idx]
                node_demand = demands[node]
                
                # Try inserting into each other route
                for target_idx, target_route in enumerate(best_routes):
                    if target_idx == source_idx:
                        continue
                        
                    # Check capacity constraint
                    route_demand = sum(demands[i] for i in target_route[1:-1])
                    if route_demand + node_demand > capacity:
                        continue
                    
                    # Try all possible insertion positions
                    for pos in range(1, len(target_route)):
                        # Create temporary routes for this move
                        temp_routes = [r.copy() for r in best_routes]
                        # Remove node from source route
                        temp_source = [n for i, n in enumerate(source_route) if i != node_idx]
                        
                        # Try insertion
                        new_target, new_length = try_insert_node(
                            target_route, node, pos, distances, time_windows, speed)
                            
                        if new_target is None:  # Time window violation
                            continue
                            
                        # Update temporary routes
                        temp_routes[source_idx] = temp_source
                        temp_routes[target_idx] = new_target
                        
                        # Calculate total distance
                        new_total = calculate_total_distance(temp_routes, distances)
                        
                        if new_total < best_total:
                            best_routes = temp_routes
                            best_total = new_total
                            improvement_found = True
                            break
                            
                    if improvement_found:
                        break
                if improvement_found:
                    break
            if improvement_found:
                break
                
    return best_routes, best_total

import numpy as np
from typing import List, Tuple

def calculate_route_length(route: List[int], 
                         distances: np.ndarray) -> float:
    """Calculate total length of a route."""
    return sum(distances[route[i]][route[i+1]] 
              for i in range(len(route)-1))

def two_opt_swap(route: List[int], i: int, j: int) -> List[int]:
    """Perform 2-opt swap operation."""
    new_route = route[:i]
    new_route.extend(reversed(route[i:j + 1]))
    new_route.extend(route[j + 1:])
    return new_route

def two_opt_improvement(route: List[int], 
                       distances: np.ndarray) -> Tuple[List[int], float]:
    """
    Apply 2-opt local search improvement
    
    Args:
        route: Initial route
        distances: Distance matrix
        
    Returns:
        improved_route: Improved route
        improved_length: Length of improved route
    """
    improvement = True
    best_route = route
    best_length = calculate_route_length(route, distances)
    
    while improvement:
        improvement = False
        
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route) - 1):
                new_route = two_opt_swap(best_route, i, j)
                new_length = calculate_route_length(new_route, distances)
                
                if new_length < best_length:
                    best_route = new_route
                    best_length = new_length
                    improvement = True
                    break
            if improvement:
                break
                
    return best_route, best_length

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
                   speed: float = 1.0,
                   allow_time_violation: bool = False,
                   time_violation_penalty: float = 1.5) -> Tuple[List[int], float]:
    """Try inserting a node at a specific position and calculate new length with penalties"""
    new_route = route.copy()
    new_route.insert(position, node)

    # Calculate new route length
    length = sum(distances[new_route[i]][new_route[i + 1]] 
                for i in range(len(new_route) - 1))

    # Check time window feasibility
    if time_windows:
        current_time = 0.0
        total_violation = 0.0

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
                    if not allow_time_violation:
                        return None, float('inf')
                    violation = current_time - tw.latest
                    total_violation += violation
                current_time += tw.service_time

            current_time += travel_time

        # Add time window violation penalty if allowed
        if allow_time_violation and total_violation > 0:
            length += total_violation * time_violation_penalty

    return new_route, length

def create_new_route(node: int, capacity: float) -> List[int]:
    """Create a new route with a single node"""
    return [0, node, 0]  # depot -> node -> depot

def optimize_cross_route(routes: List[List[int]],
                        distances: np.ndarray,
                        demands: List[float],
                        capacity: float,
                        time_windows: Optional[Dict[int, TimeWindow]] = None,
                        speed: float = 1.0,
                        max_iterations: int = 100,
                        allow_capacity_overflow: float = 0.1,
                        allow_time_violation: bool = True,
                        time_violation_penalty: float = 1.5,
                        capacity_penalty: float = 2.0) -> Tuple[List[List[int]], float]:
    """
    Optimize routes by moving nodes between routes with fallback mechanisms

    Args:
        routes: List of vehicle routes
        distances: Distance matrix
        demands: List of demands for each node
        capacity: Vehicle capacity
        time_windows: Optional time window constraints
        speed: Travel speed
        max_iterations: Maximum number of improvement attempts
        allow_capacity_overflow: Allowed capacity overflow as fraction of capacity
        allow_time_violation: Whether to allow time window violations
        time_violation_penalty: Penalty factor for time window violations
        capacity_penalty: Penalty factor for capacity violations

    Returns:
        improved_routes: List of improved routes
        total_distance: Total distance of improved solution
    """
    import streamlit as st

    best_routes = [route.copy() for route in routes]
    best_total = calculate_total_distance(routes, distances)
    improvement_found = True
    iteration = 0

    # Track unassigned nodes
    all_nodes = set()
    for route in routes:
        all_nodes.update(set(route[1:-1]))  # Exclude depot

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
                best_insertion = None
                min_cost_increase = float('inf')
                fallback_insertion = None
                min_fallback_cost = float('inf')

                for target_idx, target_route in enumerate(best_routes):
                    if target_idx == source_idx:
                        continue

                    # Check capacity with allowed overflow
                    route_demand = sum(demands[i] for i in target_route[1:-1])
                    max_allowed_capacity = capacity * (1 + allow_capacity_overflow)

                    # Try all possible insertion positions
                    for pos in range(1, len(target_route)):
                        # Try strict insertion first
                        new_route, new_length = try_insert_node(
                            target_route, node, pos, distances, 
                            time_windows, speed, False)

                        if new_route is not None:
                            if route_demand + node_demand <= capacity:
                                # Feasible insertion
                                cost_increase = new_length - sum(
                                    distances[target_route[i]][target_route[i+1]]
                                    for i in range(len(target_route)-1))

                                if cost_increase < min_cost_increase:
                                    min_cost_increase = cost_increase
                                    best_insertion = (target_idx, pos, new_route, False)

                        # Try relaxed insertion if strict fails
                        if allow_time_violation:
                            relaxed_route, relaxed_length = try_insert_node(
                                target_route, node, pos, distances,
                                time_windows, speed, True, time_violation_penalty)

                            if route_demand + node_demand <= max_allowed_capacity:
                                cost_increase = relaxed_length - sum(
                                    distances[target_route[i]][target_route[i+1]]
                                    for i in range(len(target_route)-1))

                                # Add capacity penalty if needed
                                if route_demand + node_demand > capacity:
                                    overflow = route_demand + node_demand - capacity
                                    cost_increase += overflow * capacity_penalty

                                if cost_increase < min_fallback_cost:
                                    min_fallback_cost = cost_increase
                                    fallback_insertion = (target_idx, pos, relaxed_route, True)

                # Apply best insertion (strict or fallback)
                if best_insertion:
                    target_idx, pos, new_route, is_relaxed = best_insertion
                    # Remove node from source route
                    best_routes[source_idx] = [n for i, n in enumerate(source_route) if i != node_idx]
                    # Update target route
                    best_routes[target_idx] = new_route
                    improvement_found = True

                    st.write(f"Moved node {node} to route {target_idx} (strict insertion)")

                elif fallback_insertion:
                    target_idx, pos, new_route, is_relaxed = fallback_insertion
                    # Remove node from source route
                    best_routes[source_idx] = [n for i, n in enumerate(source_route) if i != node_idx]
                    # Update target route
                    best_routes[target_idx] = new_route
                    improvement_found = True

                    st.write(f"Moved node {node} to route {target_idx} (relaxed constraints)")
                    st.write(f"Additional cost: {min_fallback_cost:.2f}")

                elif len(source_route) <= 3:  # Route would become empty
                    # Create new route if necessary
                    if node_demand <= capacity:
                        new_route = create_new_route(node, capacity)
                        best_routes.append(new_route)
                        best_routes[source_idx] = [0, 0]  # Empty route
                        st.write(f"Created new route for node {node}")

                if improvement_found:
                    break
            if improvement_found:
                break

    # Verify all nodes are assigned
    final_nodes = set()
    for route in best_routes:
        final_nodes.update(set(route[1:-1]))

    missing = all_nodes - final_nodes
    if missing:
        st.error(f"Warning: {len(missing)} nodes remain unassigned: {missing}")

    return best_routes, calculate_total_distance(best_routes, distances)
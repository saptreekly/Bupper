import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.cluster import KMeans, MiniBatchKMeans
import streamlit as st
import math
from concurrent.futures import ThreadPoolExecutor
from functools import partial

def calculate_min_vehicles(n_points: int,
                         demands: List[float],
                         capacity: float,
                         min_vehicles: int = 5,
                         max_vehicles: int = 50,  # Increased from 10
                         max_points_per_vehicle: int = 25) -> int:
    """Calculate minimum number of vehicles needed based on demands and capacity."""
    # Sum all demands (excluding depot at index 0)
    total_demand = sum(demands[1:])

    # Calculate minimum vehicles needed based on different constraints
    vehicles_by_demand = math.ceil(total_demand / capacity)
    vehicles_by_size = math.ceil((n_points - 1) / max_points_per_vehicle)

    # Add scaling factor for larger problems
    scaling_factor = math.log2(n_points / 100) if n_points > 100 else 1
    adjusted_min = math.ceil(max(vehicles_by_demand, vehicles_by_size) * scaling_factor)

    # Clamp to acceptable range with higher limits
    return max(min_vehicles, min(adjusted_min, max_vehicles))

def parallel_distance_calculation(node_point: np.ndarray,
                                route_points: np.ndarray) -> float:
    """Calculate minimum distance from a point to route points using vectorization."""
    distances = np.sqrt(np.sum((route_points - node_point) ** 2, axis=1))
    return np.min(distances) if len(distances) > 0 else float('inf')

def cluster_points(points: np.ndarray, 
                  n_clusters: Optional[int] = None,
                  demands: Optional[List[float]] = None,
                  capacity: Optional[float] = None) -> Tuple[List[List[int]], List[int]]:
    """
    Cluster points using efficient K-means with parallel processing for large datasets
    """
    # Calculate required vehicles
    if n_clusters is None and demands is not None and capacity is not None:
        n_clusters = calculate_min_vehicles(
            len(points), demands, capacity
        )
        st.write(f"\nAutomatically determined {n_clusters} vehicles needed")
    elif n_clusters is None:
        points_per_vehicle = 25
        n_clusters = max(5, min(50, (len(points) - 1) // points_per_vehicle))
        st.write(f"\nUsing default of {n_clusters} vehicles based on problem size")

    # Separate depot and delivery points
    delivery_points = points[1:]  # Skip depot

    # Use MiniBatchKMeans for large datasets
    if len(delivery_points) > 500:
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=100,
            random_state=42
        )
    else:
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42
        )

    # Fit clustering model
    delivery_labels = kmeans.fit_predict(delivery_points)

    # Initialize route construction
    route_indices = [[] for _ in range(n_clusters)]
    assigned_nodes = set([0])  # Depot is always assigned

    # First pass: Assign nodes to their clusters
    for i in range(n_clusters):
        cluster_nodes = [j + 1 for j, label in enumerate(delivery_labels) if label == i]
        route_indices[i].extend(cluster_nodes)
        assigned_nodes.update(cluster_nodes)

    # Handle unassigned nodes with parallel processing
    all_nodes = set(range(len(points)))
    unassigned = all_nodes - assigned_nodes

    if unassigned:
        st.write(f"\nAssigning {len(unassigned)} nodes to routes...")

        # Process unassigned nodes in parallel
        with ThreadPoolExecutor() as executor:
            for node in unassigned:
                node_point = points[node]
                best_route = -1
                min_distance = float('inf')

                # Calculate distances to all routes in parallel
                distance_calc = partial(parallel_distance_calculation, node_point)
                route_points = [points[route] for route in route_indices if route]
                distances = list(executor.map(distance_calc, route_points))

                # Find best route considering capacity
                for i, dist in enumerate(distances):
                    if dist < min_distance:
                        if not demands or not capacity or \
                           sum(demands[n] for n in route_indices[i]) + demands[node] <= capacity * 1.1:
                            min_distance = dist
                            best_route = i

                if best_route >= 0:
                    route_indices[best_route].insert(-1, node)
                else:
                    # Create new route if necessary
                    new_route = [0, node, 0]
                    route_indices.append(new_route)

    # Ensure depot at start/end of each route
    for i in range(len(route_indices)):
        if not route_indices[i]:
            route_indices[i] = [0, 0]
        else:
            if route_indices[i][0] != 0:
                route_indices[i].insert(0, 0)
            if route_indices[i][-1] != 0:
                route_indices[i].append(0)

    # Create labels array efficiently
    labels = np.zeros(len(points), dtype=int)
    for i, route in enumerate(route_indices):
        labels[route[1:-1]] = i  # Vectorized assignment

    # Verify assignment
    assigned = set()
    for route in route_indices:
        assigned.update(route[1:-1])

    if missing := (all_nodes - {0} - assigned):
        st.error(f"Nodes not assigned to any route: {missing}")
    else:
        st.success(f"All nodes assigned across {len(route_indices)} routes")

    return route_indices, labels

def check_capacity_constraints(route: List[int], 
                             demands: List[float],
                             capacity: float) -> bool:
    """Check if route satisfies capacity constraint"""
    total_demand = sum(demands[i] for i in route[1:])  # Skip depot
    return total_demand <= capacity
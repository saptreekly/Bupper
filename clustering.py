import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.cluster import KMeans
import streamlit as st
import math

def calculate_min_vehicles(n_points: int,
                         demands: List[float],
                         capacity: float,
                         min_vehicles: int = 2,
                         max_vehicles: int = 10) -> int:
    """Calculate minimum number of vehicles needed based on demands and capacity."""
    # Sum all demands (excluding depot at index 0)
    total_demand = sum(demands[1:])

    # Calculate theoretical minimum vehicles needed
    min_vehicles_demand = math.ceil(total_demand / capacity)

    # Consider number of points (aim for reasonable route sizes)
    points_per_vehicle = 30  # Target max points per vehicle
    min_vehicles_size = math.ceil((n_points - 1) / points_per_vehicle)

    # Take maximum of both constraints
    required_vehicles = max(min_vehicles_demand, min_vehicles_size)

    # Clamp to acceptable range
    return max(min_vehicles, min(required_vehicles, max_vehicles))

def cluster_points(points: np.ndarray, 
                  n_clusters: Optional[int] = None,
                  demands: Optional[List[float]] = None,
                  capacity: Optional[float] = None) -> Tuple[List[List[int]], List[int]]:
    """
    Cluster points using K-means with automatic vehicle count determination
    """
    # Calculate minimum required vehicles if not specified
    if n_clusters is None and demands is not None and capacity is not None:
        n_clusters = calculate_min_vehicles(
            len(points), demands, capacity
        )
        st.write(f"\nAutomatically determined {n_clusters} vehicles needed")
    elif n_clusters is None:
        n_clusters = max(2, min(10, (len(points) - 1) // 30))
        st.write(f"\nUsing default of {n_clusters} vehicles based on problem size")

    # Separate depot and delivery points
    delivery_points = points[1:]  # Skip depot (index 0)

    # Fit K-means on delivery points only
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    delivery_labels = kmeans.fit_predict(delivery_points)

    # Initialize route construction
    route_indices = [[] for _ in range(n_clusters)]
    assigned_nodes = set([0])  # Depot is always assigned

    # First pass: Assign nodes to their clusters
    st.write("\n=== Initial Cluster Assignments ===")
    for i in range(n_clusters):
        cluster_nodes = [j + 1 for j, label in enumerate(delivery_labels) if label == i]
        route_indices[i].extend(cluster_nodes)
        assigned_nodes.update(cluster_nodes)

    # Check for unassigned nodes
    all_nodes = set(range(len(points)))
    unassigned = all_nodes - assigned_nodes

    if unassigned:
        st.write(f"\nFound {len(unassigned)} unassigned nodes")
        # Try to assign nodes to nearest feasible route
        for node in unassigned:
            best_route = -1
            min_distance = float('inf')
            node_point = points[node]

            for i, route in enumerate(route_indices):
                # Skip empty routes
                if not route:
                    continue

                # Check capacity if available
                if demands and capacity:
                    route_demand = sum(demands[n] for n in route)
                    if route_demand + demands[node] > capacity * 1.1:  # Allow 10% overflow
                        continue

                # Calculate distance to route centroid
                route_points = points[route]
                centroid = np.mean(route_points, axis=0)
                dist = np.sqrt(np.sum((node_point - centroid) ** 2))

                if dist < min_distance:
                    min_distance = dist
                    best_route = i

            if best_route >= 0:
                # Insert node before final depot if present
                if route_indices[best_route] and route_indices[best_route][-1] == 0:
                    route_indices[best_route].insert(-1, node)
                else:
                    route_indices[best_route].append(node)
                st.write(f"Assigned node {node} to route {best_route}")
            else:
                # Create new route if necessary and possible
                if len(route_indices) < max_vehicles:
                    new_route = [0, node, 0]  # depot -> node -> depot
                    route_indices.append(new_route)
                    st.write(f"Created new route for node {node}")
                else:
                    st.error(f"Could not assign node {node} - consider increasing vehicle count")

    # Ensure depot at start/end of each route
    for i in range(len(route_indices)):
        if not route_indices[i]:  # Empty route
            route_indices[i] = [0, 0]
        else:
            if route_indices[i][0] != 0:
                route_indices[i].insert(0, 0)
            if route_indices[i][-1] != 0:
                route_indices[i].append(0)

    # Create full labels array
    labels = np.zeros(len(points), dtype=int)
    for i, route in enumerate(route_indices):
        for node in route[1:-1]:  # Skip depot
            labels[node] = i

    # Verify assignment
    assigned = set()
    for route in route_indices:
        assigned.update(route[1:-1])  # Exclude depot

    if missing := (all_nodes - {0} - assigned):  # Exclude depot
        st.error(f"Nodes not assigned to any route: {missing}")
    else:
        st.success("All nodes assigned to routes")

    return route_indices, labels

def check_capacity_constraints(route: List[int], 
                             demands: List[float],
                             capacity: float) -> bool:
    """Check if route satisfies capacity constraint"""
    total_demand = sum(demands[i] for i in route[1:])  # Skip depot
    return total_demand <= capacity
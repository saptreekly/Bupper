import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.cluster import KMeans
import streamlit as st # Added import for streamlit logging


def calculate_distance_to_route(point: np.ndarray, 
                              route_points: np.ndarray) -> float:
    """Calculate minimum distance from a point to any point in a route."""
    if len(route_points) == 0:
        return float('inf')
    distances = np.sqrt(np.sum((route_points - point) ** 2, axis=1))
    return np.min(distances)

def cluster_points(points: np.ndarray, n_clusters: int) -> Tuple[List[List[int]], List[int]]:
    """
    Cluster points using K-means, ensuring depot inclusion and complete coverage

    Args:
        points: Array of (x, y) coordinates
        n_clusters: Number of clusters (vehicles)

    Returns:
        route_indices: List of lists containing global node indices for each route
        labels: Global array of labels (one per point, including depot)
    """
    # Separate depot and delivery points
    delivery_points = points[1:]  # Skip depot (index 0)

    # Fit K-means on delivery points only
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    delivery_labels = kmeans.fit_predict(delivery_points)

    # Create full labels array with depot label (-1) and delivery point labels
    labels = np.concatenate([[-1], delivery_labels])

    # Initialize route construction
    route_indices = [[] for _ in range(n_clusters)]
    assigned_nodes = set([0])  # Depot is always assigned

    # First pass: Assign nodes to their clusters
    for i, label in enumerate(delivery_labels, start=1):  # Start from 1 to skip depot
        route_indices[label].append(i)
        assigned_nodes.add(i)

    # Ensure depot is at start and end of each route
    for i in range(n_clusters):
        if not route_indices[i]:  # Empty route
            route_indices[i] = [0, 0]  # Minimal route: depot-depot
        else:
            # Add depot at start and end if not present
            if route_indices[i][0] != 0:
                route_indices[i].insert(0, 0)
            if route_indices[i][-1] != 0:
                route_indices[i].append(0)

    # Check for unassigned nodes
    all_nodes = set(range(len(points)))
    unassigned = all_nodes - assigned_nodes

    if unassigned:
        st.write(f"\nFound {len(unassigned)} unassigned nodes: {unassigned}")

        # Assign each unassigned node to nearest route
        for node in unassigned:
            node_point = points[node]
            best_route = 0
            min_distance = float('inf')

            # Find closest route
            for i, route in enumerate(route_indices):
                route_points = points[route]
                dist = calculate_distance_to_route(node_point, route_points)
                if dist < min_distance:
                    min_distance = dist
                    best_route = i

            # Insert node before final depot
            route_indices[best_route].insert(-1, node)
            st.write(f"Assigned node {node} to route {best_route}")

    # Handle small routes (merge or ensure minimal viable route)
    MIN_ROUTE_SIZE = 3  # Depot + at least one delivery + depot
    for i in range(n_clusters):
        if len(route_indices[i]) < MIN_ROUTE_SIZE:
            if len(route_indices[i]) == 2 and route_indices[i][0] == 0 and route_indices[i][1] == 0:
                # Empty route, can be left as is
                continue

            # Try to merge with nearest route
            route_points = points[route_indices[i]]
            best_merge = -1
            min_distance = float('inf')

            for j in range(n_clusters):
                if i != j and len(route_indices[j]) >= MIN_ROUTE_SIZE:
                    other_points = points[route_indices[j]]
                    dist = calculate_distance_to_route(route_points[1], other_points)  # Compare first delivery point
                    if dist < min_distance:
                        min_distance = dist
                        best_merge = j

            if best_merge >= 0:
                # Merge routes
                nodes_to_merge = route_indices[i][1:-1]  # Skip depot at start and end
                route_indices[best_merge][:-1].extend(nodes_to_merge)  # Insert before final depot
                route_indices[i] = [0, 0]  # Convert to empty route

                st.write(f"Merged small route {i} into route {best_merge}")

    # Verify and log final routes
    st.write("\n=== Final Route Verification ===")
    all_assigned = set()

    for i, route in enumerate(route_indices):
        st.write(f"\nRoute {i}: {route}")

        # Verify depot at start and end
        if route[0] != 0 or route[-1] != 0:
            st.error(f"Route {i} does not start and end with depot!")

        # Track assigned nodes
        all_assigned.update(route)

    # Verify all nodes are assigned
    missing = all_nodes - all_assigned
    if missing:
        st.error(f"Nodes not assigned to any route: {missing}")
    else:
        st.success("All nodes are assigned to routes")

    return route_indices, labels

def check_capacity_constraints(route: List[int], 
                             demands: List[float],
                             capacity: float) -> bool:
    """Check if route satisfies capacity constraint"""
    total_demand = sum(demands[i] for i in route[1:])  # Skip depot
    return total_demand <= capacity
import numpy as np
from typing import List, Tuple
from sklearn.cluster import KMeans

def cluster_points(points: np.ndarray, n_clusters: int) -> Tuple[List[List[int]], List[int]]:
    """
    Cluster points using K-means, excluding depot (node 0)

    Args:
        points: Array of (x, y) coordinates
        n_clusters: Number of clusters (vehicles)

    Returns:
        route_indices: List of lists containing global node indices for each route
        labels: Cluster assignment for each point
    """
    # Separate depot and delivery points
    delivery_points = points[1:]  # Skip depot (index 0)

    # Fit K-means on delivery points only
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    delivery_labels = kmeans.fit_predict(delivery_points)

    # Add depot label (special value -1)
    labels = np.concatenate([[-1], delivery_labels])

    # Group point indices by cluster, always including depot
    route_indices = []
    for i in range(n_clusters):
        # Start each route with depot (global index 0)
        route = [0]
        # Add indices of points in this cluster (adding 1 because global indices start after depot)
        cluster_mask = delivery_labels == i
        cluster_indices = np.where(cluster_mask)[0] + 1  # Convert to global indices
        route.extend(cluster_indices)
        route_indices.append(route)

    return route_indices, labels

def check_capacity_constraints(route: List[int], 
                           demands: List[float],
                           capacity: float) -> bool:
    """Check if route satisfies capacity constraint"""
    # Skip depot (first point) when summing demands
    total_demand = sum(demands[i] for i in route[1:])  # Skip depot
    return total_demand <= capacity
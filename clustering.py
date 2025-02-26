import numpy as np
from typing import List, Tuple
from sklearn.cluster import KMeans

def cluster_points(points: np.ndarray, n_clusters: int) -> Tuple[List[np.ndarray], List[int]]:
    """
    Cluster points using K-means, excluding depot (node 0)

    Args:
        points: Array of (x, y) coordinates
        n_clusters: Number of clusters (vehicles)

    Returns:
        clustered_points: List of arrays containing points for each cluster
        labels: Cluster assignment for each point
    """
    # Separate depot and delivery points
    depot = points[0]
    delivery_points = points[1:]

    # Fit K-means on delivery points only
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    delivery_labels = kmeans.fit_predict(delivery_points)

    # Add depot label (special value -1)
    labels = np.concatenate([[-1], delivery_labels])

    # Group points by cluster, always including depot
    clustered_points = []
    for i in range(n_clusters):
        cluster_mask = delivery_labels == i
        cluster_points = np.vstack([depot, delivery_points[cluster_mask]])
        clustered_points.append(cluster_points)

    return clustered_points, labels

def check_capacity_constraints(cluster_points: np.ndarray, 
                            demands: List[float],
                            capacity: float) -> bool:
    """
    Check if cluster satisfies capacity constraint

    Args:
        cluster_points: Points in the cluster
        demands: Demand for each point
        capacity: Vehicle capacity

    Returns:
        bool: True if constraints are satisfied
    """
    # Skip depot (first point) when summing demands
    total_demand = sum(demands[1:])  # Only consider delivery points
    return total_demand <= capacity
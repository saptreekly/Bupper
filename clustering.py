import numpy as np
from typing import List, Tuple
from sklearn.cluster import KMeans

def cluster_points(points: np.ndarray, n_clusters: int) -> Tuple[List[np.ndarray], List[int]]:
    """
    Cluster points using K-means
    
    Args:
        points: Array of (x, y) coordinates
        n_clusters: Number of clusters (vehicles)
        
    Returns:
        clustered_points: List of arrays containing points for each cluster
        labels: Cluster assignment for each point
    """
    # Fit K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(points)
    
    # Group points by cluster
    clustered_points = []
    for i in range(n_clusters):
        cluster_mask = labels == i
        cluster_points = points[cluster_mask]
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
    total_demand = sum(demands)
    return total_demand <= capacity

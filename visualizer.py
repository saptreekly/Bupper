import matplotlib.pyplot as plt
import numpy as np
from typing import List
import io
import streamlit as st

def plot_routes(points: np.ndarray, 
               routes: List[List[int]],
               labels: List[int],
               title: str = "VRP Routes") -> None:
    """
    Plot multiple vehicle routes

    Args:
        points: Array of (x, y) coordinates, must match global indices
        routes: List of routes with global indices
        labels: Cluster assignment for each point
        title: Plot title
    """
    # Debug logging
    st.write("Debug Information:")
    st.write(f"Points array shape: {points.shape}")
    for i, route in enumerate(routes):
        st.write(f"Route {i} indices: {route}")
        st.write(f"Route {i} max index: {max(route) if route else -1}")
        if route and max(route) >= len(points):
            raise ValueError(
                f"Route {i} contains index {max(route)} which is >= points array "
                f"length {len(points)}"
            )

    # Verify route indices are within bounds
    n_points = len(points)
    for route in routes:
        if any(idx >= n_points for idx in route):
            raise ValueError(f"Route contains index >= {n_points}. All indices must be < {n_points}")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Generate distinct colors for each route
    n_routes = len(routes)
    colors = plt.cm.rainbow(np.linspace(0, 1, n_routes))

    # Plot points with cluster colors
    unique_labels = np.unique(labels)
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(points[mask, 0], points[mask, 1], 
                  c=[colors[i]], s=50, label=f'Cluster {i}')

    # Plot routes
    for i, route in enumerate(routes):
        if not route:  # Skip empty routes
            continue

        # Plot route with corresponding color
        for j in range(len(route) - 1):
            start = points[route[j]]
            end = points[route[j + 1]]
            ax.plot([start[0], end[0]], [start[1], end[1]], 
                   c=colors[i], linestyle='-', alpha=0.6)

    # Add labels
    for i, point in enumerate(points):
        ax.annotate(f'Node {i}', (point[0], point[1]), 
                   xytext=(5, 5), textcoords='offset points')

    ax.set_title(title)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.legend()

    # Display in Streamlit
    st.pyplot(fig)
    plt.close()
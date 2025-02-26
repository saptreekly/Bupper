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
    Plot multiple vehicle routes using global indices

    Args:
        points: Full global array of (x, y) coordinates
        routes: List of routes with global node indices
        labels: Cluster assignment for each point
        title: Plot title
    """
    # Enhanced debug logging
    st.write("\n=== Route Visualization Debug Info ===")
    st.write(f"Points array type: {type(points)}")
    st.write(f"Points array shape: {points.shape}")
    st.write(f"Points array length: {len(points)}")

    # Verify route indices against array bounds
    n_points = len(points)
    for i, route in enumerate(routes):
        if route:
            max_idx = max(route)
            st.write(f"Route {i}: {route}")
            st.write(f"Route {i} length: {len(route)}")
            st.write(f"Route {i} max index: {max_idx}")
            if max_idx >= n_points:
                error_msg = (
                    f"Error: Route {i} contains index {max_idx} which exceeds "
                    f"points array size {n_points}. Routes must use indices "
                    f"from 0 to {n_points-1}."
                )
                st.error(error_msg)
                raise ValueError(error_msg)

    # Create the plot
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
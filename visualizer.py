import matplotlib.pyplot as plt
import numpy as np
from typing import List
import io
import streamlit as st

def plot_routes(points: np.ndarray, 
               routes: List[List[int]],
               labels: List[int],
               title: str = "VRP Routes",
               show_labels: bool = False) -> None:
    """
    Plot multiple vehicle routes using global indices

    Args:
        points: Full global array of (x, y) coordinates
        routes: List of routes with global node indices
        labels: Cluster assignment for each point (depot has label -1)
        title: Plot title
        show_labels: Whether to show node ID labels (default: False)
    """
    # Enhanced debug logging
    st.write("\n=== Route Visualization Debug Info ===")
    st.write(f"Points array type: {type(points)}")
    st.write(f"Points array shape: {points.shape}")
    st.write(f"Points array length: {len(points)}")
    st.write(f"All labels: {labels}")

    # Filter out depot label (-1) and get unique cluster labels
    cluster_labels = np.unique(labels[labels != -1])
    n_clusters = len(cluster_labels)
    st.write(f"Number of clusters (excluding depot): {n_clusters}")
    st.write(f"Cluster labels: {cluster_labels}")

    # Generate distinct colors for actual clusters (excluding depot)
    colors = plt.cm.rainbow(np.linspace(0, 1, n_clusters))
    st.write(f"Generated {len(colors)} colors for {n_clusters} clusters")

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot depot point with special marker
    depot_mask = labels == -1
    ax.scatter(points[depot_mask, 0], points[depot_mask, 1],
              c='black', marker='s', s=100, label='Depot')

    # Plot delivery points with cluster colors
    for i, label in enumerate(cluster_labels):  # Iterate over non-depot labels
        mask = labels == label
        ax.scatter(points[mask, 0], points[mask, 1],
                  c=[colors[i]], s=50, label=f'Cluster {label}')

    # Plot routes
    for i, route in enumerate(routes):
        if not route:  # Skip empty routes
            continue

        # Plot route with corresponding cluster color
        # Use i directly since routes match cluster indices
        for j in range(len(route) - 1):
            start = points[route[j]]
            end = points[route[j + 1]]
            ax.plot([start[0], end[0]], [start[1], end[1]],
                   c=colors[i], linestyle='-', alpha=0.6)

    # Add labels only if show_labels is True
    if show_labels:
        for i, point in enumerate(points):
            label_color = 'black' if i == 0 else 'dimgrey'
            ax.annotate(f'Node {i}', (point[0], point[1]),
                       xytext=(5, 5), textcoords='offset points',
                       color=label_color)

    ax.set_title(title)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.legend()

    # Display in Streamlit
    st.pyplot(fig)
    plt.close()
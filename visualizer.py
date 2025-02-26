import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional
import io
import streamlit as st

def plot_routes(points: np.ndarray, 
               routes: List[List[int]],
               labels: List[int],
               title: str = "VRP Routes",
               global_points: Optional[np.ndarray] = None) -> None:
    """
    Plot multiple vehicle routes with automatic array size validation

    Args:
        points: Array of (x, y) coordinates
        routes: List of routes with node indices
        labels: Cluster assignment for each point
        title: Plot title
        global_points: Optional full points array to use if local indices are out of bounds
    """
    # Debug logging
    st.write("\n=== Plotting Debug Information ===")
    st.write(f"Local points array shape: {points.shape}")
    if global_points is not None:
        st.write(f"Global points array shape: {global_points.shape}")

    # Check route indices against array bounds
    points_to_use = points
    max_local_idx = len(points) - 1
    needs_global_array = False

    for i, route in enumerate(routes):
        st.write(f"\nRoute {i} debugging:")
        st.write(f"  Indices: {route}")
        if route:
            max_route_idx = max(route)
            st.write(f"  Max index: {max_route_idx}")
            if max_route_idx > max_local_idx:
                st.write(f"  ⚠️ Index {max_route_idx} exceeds local array size {max_local_idx}")
                if global_points is not None:
                    needs_global_array = True
                    if max_route_idx >= len(global_points):
                        raise ValueError(
                            f"Route {i} contains index {max_route_idx} which exceeds "
                            f"even the global array size {len(global_points)}"
                        )
                else:
                    raise ValueError(
                        f"Route {i} contains index {max_route_idx} >= points array "
                        f"length {len(points)} and no global array provided"
                    )

    # Switch to global array if needed
    if needs_global_array:
        st.write("\n⚠️ Switching to global points array due to out-of-bounds indices")
        points_to_use = global_points
        st.write(f"Using array with shape: {points_to_use.shape}")

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Generate distinct colors for each route
    n_routes = len(routes)
    colors = plt.cm.rainbow(np.linspace(0, 1, n_routes))

    # Plot points with cluster colors
    unique_labels = np.unique(labels)
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(points_to_use[mask, 0], points_to_use[mask, 1], 
                  c=[colors[i]], s=50, label=f'Cluster {i}')

    # Plot routes
    for i, route in enumerate(routes):
        if not route:  # Skip empty routes
            continue

        # Plot route with corresponding color
        for j in range(len(route) - 1):
            start = points_to_use[route[j]]
            end = points_to_use[route[j + 1]]
            ax.plot([start[0], end[0]], [start[1], end[1]], 
                   c=colors[i], linestyle='-', alpha=0.6)

    # Add labels
    for i, point in enumerate(points_to_use):
        ax.annotate(f'Node {i}', (point[0], point[1]), 
                   xytext=(5, 5), textcoords='offset points')

    ax.set_title(title)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.legend()

    # Display in Streamlit
    st.pyplot(fig)
    plt.close()
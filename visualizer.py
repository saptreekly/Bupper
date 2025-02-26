import matplotlib.pyplot as plt
import numpy as np
from typing import List
import io
import streamlit as st

def plot_route(points: np.ndarray, 
              route: List[int],
              title: str = "TSP Route") -> None:
    """
    Plot TSP route
    
    Args:
        points: Array of (x, y) coordinates
        route: List of indices representing the route
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot points
    ax.scatter(points[:, 0], points[:, 1], c='red', s=50)
    
    # Plot route
    for i in range(len(route) - 1):
        start = points[route[i]]
        end = points[route[i + 1]]
        ax.plot([start[0], end[0]], [start[1], end[1]], 'b-')
    
    # Add labels
    for i, point in enumerate(points):
        ax.annotate(f'City {i}', (point[0], point[1]), 
                   xytext=(5, 5), textcoords='offset points')
    
    ax.set_title(title)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    
    # Display in Streamlit
    st.pyplot(fig)
    plt.close()

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import streamlit as st
from dataclasses import dataclass
import time

@dataclass
class PhysarumParams:
    """Parameters for Physarum simulation"""
    gamma: float = 1.3  # Flow feedback strength
    mu: float = 0.1    # Decay rate
    dt: float = 0.01   # Time step
    convergence_threshold: float = 1e-6

class PhysarumSolver:
    """Physarum-inspired network optimizer"""
    
    def __init__(self, 
                 points: np.ndarray,
                 params: Optional[PhysarumParams] = None):
        """
        Initialize solver with points and parameters
        Args:
            points: Nx2 array of point coordinates
            params: Simulation parameters
        """
        self.points = points
        self.n_points = len(points)
        self.params = params or PhysarumParams()
        
        # Initialize graph and edge properties
        self.G = nx.complete_graph(self.n_points)
        self.pos = {i: points[i] for i in range(self.n_points)}
        
        # Calculate edge lengths
        self.lengths = {}
        self.conductivity = {}
        for (i, j) in self.G.edges():
            length = np.sqrt(np.sum((points[i] - points[j])**2))
            self.lengths[(i, j)] = length
            self.lengths[(j, i)] = length
            # Initialize conductivity proportional to 1/length
            self.conductivity[(i, j)] = 1.0 / length
            self.conductivity[(j, i)] = 1.0 / length
            
        self.pressures = np.zeros(self.n_points)
        self.flows = {edge: 0.0 for edge in self.G.edges()}

    def compute_flows(self, source: int, sink: int) -> None:
        """
        Compute flows using Kirchhoff's laws
        Args:
            source: Source node index
            sink: Sink node index
        """
        n = self.n_points
        A = np.zeros((n, n))  # Conductance matrix
        b = np.zeros(n)      # RHS vector
        
        # Build conductance matrix
        for i, j in self.G.edges():
            conductance = self.conductivity[(i, j)] / self.lengths[(i, j)]
            A[i, i] += conductance
            A[j, j] += conductance
            A[i, j] -= conductance
            A[j, i] -= conductance
            
        # Set boundary conditions
        A[source] = 0
        A[sink] = 0
        A[source, source] = 1
        A[sink, sink] = 1
        b[source] = 1  # Source pressure
        b[sink] = 0    # Sink pressure
        
        # Solve for pressures
        self.pressures = np.linalg.solve(A, b)
        
        # Calculate flows
        for i, j in self.G.edges():
            conductance = self.conductivity[(i, j)] / self.lengths[(i, j)]
            flow = conductance * (self.pressures[i] - self.pressures[j])
            self.flows[(i, j)] = abs(flow)
            self.flows[(j, i)] = abs(flow)

    def update_conductivity(self) -> float:
        """
        Update conductivity based on flows
        Returns:
            max_change: Maximum conductivity change
        """
        max_change = 0.0
        for edge in self.G.edges():
            flow = self.flows[edge]
            old_conductivity = self.conductivity[edge]
            
            # Update according to Tero model
            new_conductivity = (old_conductivity + 
                              self.params.dt * (
                                  pow(flow, self.params.gamma) - 
                                  self.params.mu * old_conductivity
                              ))
            
            # Ensure non-negative conductivity
            new_conductivity = max(1e-6, new_conductivity)
            self.conductivity[edge] = new_conductivity
            max_change = max(max_change, 
                           abs(new_conductivity - old_conductivity))
            
        return max_change

    def calculate_network_cost(self) -> float:
        """Calculate total network cost"""
        return sum(self.conductivity[edge] * self.lengths[edge] 
                  for edge in self.G.edges())

    def visualize_network(self, iteration: int) -> plt.Figure:
        """
        Visualize current network state
        Args:
            iteration: Current iteration number
        Returns:
            fig: matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Draw edges with thickness proportional to conductivity
        max_conductivity = max(self.conductivity.values())
        for (i, j) in self.G.edges():
            relative_thickness = self.conductivity[(i, j)] / max_conductivity
            if relative_thickness > 0.01:  # Only draw significant edges
                ax.plot([self.points[i,0], self.points[j,0]],
                       [self.points[i,1], self.points[j,1]],
                       'k-', linewidth=relative_thickness*5,
                       alpha=relative_thickness)
        
        # Draw nodes
        ax.scatter(self.points[:,0], self.points[:,1], 
                  c='blue', s=100)
        
        ax.set_title(f'Network State (Iteration {iteration})')
        ax.set_aspect('equal')
        plt.close()  # Prevent display in notebook context
        return fig

    def solve(self, 
             source: int,
             sink: int,
             max_iterations: int = 1000) -> Tuple[Dict, List[float]]:
        """
        Run Physarum simulation
        Args:
            source: Source node index
            sink: Sink node index
            max_iterations: Maximum number of iterations
        Returns:
            best_conductivity: Best conductivity configuration
            costs: History of network costs
        """
        costs = []
        best_cost = float('inf')
        best_conductivity = None
        
        for iteration in range(max_iterations):
            # Compute flows
            self.compute_flows(source, sink)
            
            # Update conductivity
            max_change = self.update_conductivity()
            
            # Calculate and track cost
            cost = self.calculate_network_cost()
            costs.append(cost)
            
            # Update best solution
            if cost < best_cost:
                best_cost = cost
                best_conductivity = self.conductivity.copy()
            
            # Log progress periodically
            if iteration % 100 == 0:
                st.write(f"Iteration {iteration}: Cost = {cost:.2f}, "
                        f"Max change = {max_change:.6f}")
            
            # Check convergence
            if max_change < self.params.convergence_threshold:
                st.write(f"Converged after {iteration} iterations")
                break
                
        return best_conductivity, costs

def create_random_points(n_points: int, 
                       size: float = 100.0) -> np.ndarray:
    """Generate random points for testing"""
    points = np.random.rand(n_points, 2) * size
    return points

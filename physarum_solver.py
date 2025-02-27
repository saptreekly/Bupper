import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import streamlit as st
from dataclasses import dataclass
import time
from utils import TimeWindow
from scipy import sparse
from concurrent.futures import ThreadPoolExecutor, as_completed

@dataclass
class PhysarumParams:
    """Parameters for Physarum simulation"""
    gamma: float = 1.3  # Flow feedback strength
    mu: float = 0.1    # Decay rate
    dt: float = 0.01   # Time step
    convergence_threshold: float = 1e-6
    min_conductivity: float = 0.01  # Minimum conductivity threshold
    stagnation_limit: int = 20  # Maximum iterations without improvement
    min_improvement: float = 0.001  # Minimum relative improvement threshold
    max_conductivity: float = 10.0  # Maximum conductivity cap

class PhysarumSolver:
    """Optimized Physarum-inspired network optimizer"""

    def __init__(self, 
                 points: np.ndarray,
                 params: Optional[PhysarumParams] = None,
                 time_windows: Optional[Dict[int, TimeWindow]] = None,
                 speed: float = 1.0):
        self.points = points
        self.n_points = len(points)
        self.params = params or PhysarumParams()
        self.time_windows = time_windows
        self.speed = speed

        # Calculate distances using vectorized operations
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        self.distances = np.sqrt(np.sum(diff * diff, axis=-1))

        # Initialize sparse conductivity matrix
        self.conductivity_matrix = sparse.lil_matrix((self.n_points, self.n_points))
        self.initialize_conductivity()

    def initialize_conductivity(self) -> None:
        """Initialize conductivity using k-nearest neighbors"""
        k = min(5, self.n_points - 1)  # Adaptive k based on problem size

        # Find k nearest neighbors for each node
        for i in range(self.n_points):
            # Get indices of k nearest neighbors (excluding self)
            dist_to_i = self.distances[i]
            nearest = np.argpartition(dist_to_i, k+1)[:k+1]
            nearest = nearest[nearest != i]  # Remove self if present

            # Initialize conductivity for nearest neighbors
            for j in nearest:
                init_cond = 1.0 / (self.distances[i,j] + 1e-6)
                self.conductivity_matrix[i,j] = init_cond
                self.conductivity_matrix[j,i] = init_cond

    def compute_flows(self) -> np.ndarray:
        """Compute flows using vectorized operations"""
        n = self.n_points
        total_flows = sparse.lil_matrix((n, n))
        conductance = self.conductivity_matrix.multiply(1.0 / (self.distances + 1e-6))

        # Parallelize flow computation for each destination
        with ThreadPoolExecutor() as executor:
            futures = []
            for dest in range(1, n):  # Skip depot
                futures.append(executor.submit(self._compute_flow_for_dest, 
                                            conductance, dest))

            # Collect results
            for future in as_completed(futures):
                flow_matrix = future.result()
                total_flows += flow_matrix

        return total_flows.tocsr()

    def _compute_flow_for_dest(self, conductance: sparse.spmatrix, dest: int) -> sparse.spmatrix:
        """Compute flow for a single destination"""
        n = self.n_points

        # Setup linear system
        A = conductance.tocsr()
        diag = np.array(A.sum(axis=1)).flatten()
        A = A - sparse.diags(diag)

        # Set boundary conditions
        b = np.zeros(n)
        b[0] = 1.0  # Source at depot
        b[dest] = -1.0  # Sink

        # Solve system
        pressures = sparse.linalg.spsolve(A, b)

        # Calculate flows
        flow_matrix = sparse.lil_matrix((n, n))
        rows, cols = conductance.nonzero()
        flows = conductance.multiply(pressures[rows] - pressures[cols])
        flow_matrix[rows, cols] = np.abs(flows.data)

        return flow_matrix

    def update_conductivity(self, flows: sparse.spmatrix) -> float:
        """Update conductivity using vectorized operations"""
        # Calculate growth and decay terms
        growth = flows.power(self.params.gamma).tocsr()
        decay = self.conductivity_matrix.multiply(self.params.mu)

        # Update conductivity
        delta = self.params.dt * (growth - decay)
        new_conductivity = self.conductivity_matrix + delta

        # Apply bounds
        new_conductivity.data = np.clip(new_conductivity.data,
                                      self.params.min_conductivity,
                                      self.params.max_conductivity)

        # Calculate maximum change
        max_change = np.max(np.abs(delta.data)) if delta.nnz > 0 else 0

        self.conductivity_matrix = new_conductivity
        return max_change

    def calculate_network_cost(self) -> float:
        """Calculate total network cost"""
        return np.sum(self.conductivity_matrix.multiply(self.distances))

    def visualize_network(self, iteration: int) -> plt.Figure:
        """Visualize current network state with enhanced visibility"""
        fig, ax = plt.subplots(figsize=(10, 10))

        # Draw edges with enhanced visibility
        max_conductivity = self.conductivity_matrix.max()
        min_conductivity = self.params.min_conductivity

        # Define color scheme
        edge_color = '#1E56A0'  # Royal blue
        strong_edge_color = '#D63230'  # Crimson for strong connections

        rows, cols = self.conductivity_matrix.nonzero()
        for i, j in zip(rows, cols):
            relative_strength = (self.conductivity_matrix[i, j] - min_conductivity) / (max_conductivity - min_conductivity)

            if relative_strength > 0.001:  # Lower threshold for visibility
                # Adjust line thickness (increased scale)
                thickness = max(0.5, relative_strength * 8.0)

                # Adjust opacity (higher minimum)
                opacity = max(0.4, min(0.9, 0.4 + relative_strength * 0.5))

                # Use different styles based on strength
                if relative_strength > 0.5:
                    color = strong_edge_color
                    linestyle = '-'  # Solid for strong connections
                else:
                    color = edge_color
                    linestyle = '--'  # Dashed for weaker connections

                ax.plot([self.points[i,0], self.points[j,0]],
                       [self.points[i,1], self.points[j,1]],
                       color=color,
                       linestyle=linestyle,
                       linewidth=thickness,
                       alpha=opacity,
                       zorder=1)  # Ensure edges are below nodes

        # Draw nodes with enhanced visibility
        node_size = 150
        # Draw nodes
        ax.scatter(self.points[:,0], self.points[:,1], 
                  c='#2B2D42',  # Dark blue-gray
                  s=node_size,
                  zorder=2,  # Ensure nodes are above edges
                  edgecolor='white',
                  linewidth=1)

        # Highlight source and sink (depot in this case)
        ax.scatter([self.points[0,0]], [self.points[0,1]], 
                  c='#06A77D',  # Green
                  s=node_size*1.5,
                  label='Depot',
                  zorder=3,
                  edgecolor='white',
                  linewidth=1.5)


        ax.set_title(f'Network State (Iteration {iteration})')
        ax.set_aspect('equal')
        ax.legend(fontsize=10, markerscale=1.5)

        # Clean up axes
        ax.set_xticks([])
        ax.set_yticks([])

        plt.close()  # Prevent display in notebook context
        return fig

    def extract_route(self, best_conductivity: Dict) -> List[int]:
        """
        Extract TSP route ensuring Hamiltonian cycle through all nodes.
        Uses high-conductivity edges as preferences for the TSP solver.
        """
        # Create weighted graph from conductivities
        G = nx.Graph()

        # Add all nodes first to ensure complete graph
        for i in range(self.n_points):
            G.add_node(i)

        # Add edges with weights based on conductivity
        for (i, j), cond in best_conductivity.items():
            # Use inverse conductivity as weight (stronger connections = shorter distances)
            # Add a small epsilon to avoid division by zero
            weight = 1.0 / (cond + 1e-6)

            # Add higher penalty for weak connections
            if cond < self.params.min_conductivity:
                weight *= 10.0

            G.add_edge(i, j, weight=weight)

        # Ensure graph is complete for TSP
        for i in range(self.n_points):
            for j in range(i+1, self.n_points):
                if not G.has_edge(i, j):
                    # Add edge with high weight to discourage use
                    dist = np.sqrt(np.sum((self.points[i] - self.points[j]) ** 2))
                    G.add_edge(i, j, weight=dist * 10.0)

        # Use Christofides algorithm for approximate TSP solution
        # This guarantees a tour that is at most 1.5 times the optimal
        route = list(nx.approximation.christofides(G))

        # Ensure route starts and ends at depot (0)
        if route[0] != 0:
            depot_pos = route.index(0)
            route = route[depot_pos:] + route[:depot_pos]

        # Add return to depot
        if route[-1] != 0:
            route.append(0)

        return route

    def solve(self, max_iterations: int = 1000) -> Tuple[Dict, List[float]]:
        """Run optimized Physarum simulation"""
        if self.n_points <= 1:
            return {}, []

        costs = []
        best_cost = float('inf')
        stagnation_count = 0
        previous_cost = float('inf')

        for iteration in range(max_iterations):
            # Compute flows using parallel processing
            flows = self.compute_flows()

            # Update conductivity
            max_change = self.update_conductivity(flows)

            # Calculate cost (less frequently)
            if iteration % 10 == 0:
                cost = self.calculate_network_cost()
                costs.append(cost)

                # Check for improvement
                rel_improvement = (previous_cost - cost) / previous_cost if previous_cost != float('inf') else 1.0

                if cost < best_cost:
                    best_cost = cost
                    stagnation_count = 0
                    if rel_improvement < self.params.min_improvement:
                        stagnation_count += 1
                else:
                    stagnation_count += 1

                previous_cost = cost

                # Log only significant changes
                if iteration == 0 or rel_improvement > 0.01:
                    st.write(f"Iteration {iteration}: Cost = {cost:.2f}")

            # Early stopping checks
            if max_change < self.params.convergence_threshold:
                st.write(f"Converged after {iteration} iterations")
                break

            if stagnation_count >= self.params.stagnation_limit:
                st.write(f"Stopping due to stagnation after {iteration} iterations")
                break

        # Convert final conductivity to dictionary format
        rows, cols = self.conductivity_matrix.nonzero()
        conductivities = {(int(i), int(j)): float(self.conductivity_matrix[i,j])
                         for i, j in zip(rows, cols)}

        return conductivities, costs

def create_random_points(n_points: int, 
                       size: float = 100.0) -> np.ndarray:
    """Generate random points for testing"""
    points = np.random.rand(n_points, 2) * size
    return points
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import streamlit as st
from dataclasses import dataclass
import time
from utils import TimeWindow

@dataclass
class PhysarumParams:
    """Parameters for Physarum simulation"""
    gamma: float = 1.3  # Flow feedback strength
    mu: float = 0.1    # Decay rate
    dt: float = 0.01   # Time step
    convergence_threshold: float = 1e-6
    min_conductivity: float = 0.01  # Minimum conductivity threshold
    connectivity_check_interval: int = 50  # Check connectivity every N iterations

class PhysarumSolver:
    """Physarum-inspired network optimizer"""

    def __init__(self, 
                 points: np.ndarray,
                 params: Optional[PhysarumParams] = None,
                 time_windows: Optional[Dict[int, TimeWindow]] = None,
                 speed: float = 1.0):
        """Initialize solver with points and parameters"""
        self.points = points
        self.n_points = len(points)
        self.params = params or PhysarumParams()
        self.time_windows = time_windows
        self.speed = speed

        # Initialize graph and edge properties
        self.G = nx.complete_graph(self.n_points)
        self.pos = {i: points[i] for i in range(self.n_points)}

        # Calculate edge lengths and initialize conductivities
        self.lengths = {}
        self.conductivity = {}
        for (i, j) in self.G.edges():
            length = np.sqrt(np.sum((points[i] - points[j])**2))
            self.lengths[(i, j)] = length
            self.lengths[(j, i)] = length

            # Initialize with higher minimum conductivity
            initial_conductivity = max(1.0 / length, self.params.min_conductivity)
            self.conductivity[(i, j)] = initial_conductivity
            self.conductivity[(j, i)] = initial_conductivity

        self.pressures = np.zeros(self.n_points)
        self.flows = {edge: 0.0 for edge in self.G.edges()}

    def check_connectivity(self) -> bool:
        """Check if network is fully connected with significant conductivities."""
        G_significant = nx.Graph()
        for (i, j), cond in self.conductivity.items():
            if cond > self.params.min_conductivity:
                G_significant.add_edge(i, j)
        return nx.is_connected(G_significant)

    def restore_connectivity(self):
        """Restore minimum connections to ensure network connectivity."""
        G_significant = nx.Graph()
        for (i, j), cond in self.conductivity.items():
            if cond > self.params.min_conductivity:
                G_significant.add_edge(i, j)

        components = list(nx.connected_components(G_significant))
        if len(components) > 1:
            st.write(f"Restoring connectivity between {len(components)} components")
            # Connect components using minimum spanning tree approach
            for i in range(len(components)-1):
                comp1 = list(components[i])
                comp2 = list(components[i+1])
                min_dist = float('inf')
                best_edge = None

                # Find shortest edge between components
                for n1 in comp1:
                    for n2 in comp2:
                        dist = self.lengths[(n1, n2)]
                        if dist < min_dist:
                            min_dist = dist
                            best_edge = (n1, n2)

                if best_edge:
                    # Restore conductivity
                    self.conductivity[best_edge] = max(
                        self.conductivity[best_edge],
                        self.params.min_conductivity * 2
                    )
                    self.conductivity[(best_edge[1], best_edge[0])] = self.conductivity[best_edge]

    def compute_flows(self, source: int, sink: int) -> None:
        """Compute flows using Kirchhoff's laws"""
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
        """Update conductivity based on flows"""
        max_change = 0.0

        # Calculate average flow for adaptive thresholding
        avg_flow = np.mean(list(self.flows.values()))
        flow_threshold = avg_flow * 0.1  # 10% of average flow

        for edge in self.G.edges():
            flow = self.flows[edge]
            old_conductivity = self.conductivity[edge]

            # Modified update rule with flow threshold
            if flow > flow_threshold:
                growth_term = pow(flow, self.params.gamma)
            else:
                growth_term = pow(flow_threshold, self.params.gamma)

            # Update according to modified Tero model
            new_conductivity = old_conductivity + self.params.dt * (
                growth_term - self.params.mu * old_conductivity
            )

            # Apply minimum conductivity constraint
            new_conductivity = max(self.params.min_conductivity, new_conductivity)

            self.conductivity[edge] = new_conductivity
            max_change = max(max_change, abs(new_conductivity - old_conductivity))

        return max_change

    def calculate_network_cost(self) -> float:
        """Calculate total network cost"""
        return sum(self.conductivity[edge] * self.lengths[edge] 
                  for edge in self.G.edges())

    def visualize_network(self, iteration: int) -> plt.Figure:
        """Visualize current network state with enhanced visibility"""
        fig, ax = plt.subplots(figsize=(10, 10))

        # Draw edges with enhanced visibility
        max_conductivity = max(self.conductivity.values())
        min_conductivity = self.params.min_conductivity

        # Define color scheme
        edge_color = '#1E56A0'  # Royal blue
        strong_edge_color = '#D63230'  # Crimson for strong connections

        for (i, j) in self.G.edges():
            relative_strength = (self.conductivity[(i, j)] - min_conductivity) / (max_conductivity - min_conductivity)

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

        # Highlight source and sink
        ax.scatter([self.points[0,0]], [self.points[0,1]], 
                  c='#06A77D',  # Green
                  s=node_size*1.5,
                  label='Source',
                  zorder=3,
                  edgecolor='white',
                  linewidth=1.5)
        ax.scatter([self.points[-1,0]], [self.points[-1,1]], 
                  c='#D63230',  # Red
                  s=node_size*1.5,
                  label='Sink',
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

    def solve(self, 
             source: int,
             sink: int,
             max_iterations: int = 1000) -> Tuple[Dict, List[float]]:
        """Run Physarum simulation"""
        st.write("\n=== Physarum Solver Starting ===")
        st.write(f"Source: {source}, Sink: {sink}")
        st.write(f"Number of nodes: {self.n_points}")

        costs = []
        best_cost = float('inf')
        best_conductivity = None
        stagnation_counter = 0

        for iteration in range(max_iterations):
            # Compute flows
            self.compute_flows(source, sink)

            # Update conductivity
            max_change = self.update_conductivity()

            # Periodically check and restore connectivity
            if iteration % self.params.connectivity_check_interval == 0:
                if not self.check_connectivity():
                    self.restore_connectivity()
                    st.write(f"Restored connectivity at iteration {iteration}")

            # Calculate and track cost
            cost = self.calculate_network_cost()
            costs.append(cost)

            # Update best solution
            if cost < best_cost:
                best_cost = cost
                best_conductivity = self.conductivity.copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Log progress periodically
            if iteration % 100 == 0:
                st.write(f"Iteration {iteration}: Cost = {cost:.2f}, "
                        f"Max change = {max_change:.6f}")

            # Check convergence
            if max_change < self.params.convergence_threshold or stagnation_counter > 50:
                st.write(f"Converged after {iteration} iterations")
                break

        return best_conductivity, costs

    def extract_route(self, best_conductivity: Dict) -> List[int]:
        """
        Extract route from conductivity configuration
        Args:
            best_conductivity: Conductivity values for each edge
        Returns:
            route: List of node indices forming the route
        """
        # Create graph with edges weighted by conductivity
        G = nx.Graph()
        for (i, j), cond in best_conductivity.items():
            if cond > 1e-4:  # Only consider significant connections
                G.add_edge(i, j, weight=1.0/cond)  # Use inverse for shortest path

        # Find route starting from depot (node 0)
        route = [0]  # Start at depot
        current = 0
        unvisited = set(range(1, self.n_points))  # Skip depot

        while unvisited:
            # Find closest unvisited node
            min_dist = float('inf')
            next_node = None
            for node in unvisited:
                try:
                    path = nx.shortest_path(G, current, node, weight='weight')
                    dist = sum(best_conductivity[(path[i], path[i+1])] 
                             for i in range(len(path)-1))
                    if dist < min_dist:
                        min_dist = dist
                        next_node = node
                except nx.NetworkXNoPath:
                    continue

            if next_node is None:
                break  # No feasible path found

            route.append(next_node)
            unvisited.remove(next_node)
            current = next_node

        route.append(0)  # Return to depot
        return route

def create_random_points(n_points: int, 
                       size: float = 100.0) -> np.ndarray:
    """Generate random points for testing"""
    points = np.random.rand(n_points, 2) * size
    return points
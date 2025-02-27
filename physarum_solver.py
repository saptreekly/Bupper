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
    k_neighbors: int = 5  # Number of nearest neighbors for initial connectivity
    weak_flow_threshold: float = 0.1  # Threshold for identifying weak flows
    decay_multiplier: float = 1.5  # Increased decay rate for weak connections

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

        # Initialize graph with k-nearest neighbors
        self.G = nx.Graph()
        self.pos = {i: points[i] for i in range(self.n_points)}

        # Calculate all pairwise distances
        distances = np.array([[np.sqrt(np.sum((p1 - p2) ** 2))
                             for p2 in points]
                             for p1 in points])

        # Connect each node to its k nearest neighbors
        for i in range(self.n_points):
            # Get indices of k nearest neighbors (excluding self)
            nearest = np.argsort(distances[i])[1:self.params.k_neighbors + 1]
            for j in nearest:
                self.G.add_edge(i, j)

        # Initialize edge properties
        self.lengths = {}
        self.conductivity = {}
        for (i, j) in self.G.edges():
            length = distances[i][j]
            self.lengths[(i, j)] = length
            self.lengths[(j, i)] = length

            # Initialize with higher minimum conductivity
            initial_conductivity = max(1.0 / length, self.params.min_conductivity)
            self.conductivity[(i, j)] = initial_conductivity
            self.conductivity[(j, i)] = initial_conductivity

        self.pressures = np.zeros(self.n_points)
        self.flows = {edge: 0.0 for edge in self.G.edges()}

        # Ensure initial connectivity
        if not self.check_connectivity():
            self.restore_connectivity()
            st.write("Added minimum connections to ensure initial connectivity")

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
                        dist = np.sqrt(np.sum((self.points[n1] - self.points[n2]) ** 2))
                        if dist < min_dist:
                            min_dist = dist
                            best_edge = (n1, n2)

                if best_edge:
                    # Add edge to graph and initialize properties
                    self.G.add_edge(*best_edge)
                    self.lengths[best_edge] = min_dist
                    self.lengths[(best_edge[1], best_edge[0])] = min_dist

                    # Restore conductivity
                    self.conductivity[best_edge] = self.params.min_conductivity * 2
                    self.conductivity[(best_edge[1], best_edge[0])] = self.params.min_conductivity * 2

    def compute_flows(self, depot: int = 0) -> None:
        """
        Compute flows treating depot as source and all other nodes as destinations.
        This simulates the foraging behavior where paths extend from the depot.
        """
        n = self.n_points
        total_flows = np.zeros((n, n))

        # For each non-depot node as destination
        for dest in range(1, n):
            # Setup conductance matrix for this destination
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
            A[depot] = 0
            A[dest] = 0
            A[depot, depot] = 1
            A[dest, dest] = 1
            b[depot] = 1  # Flow source at depot
            b[dest] = 0   # Flow sink at destination

            # Solve for pressures
            pressures = np.linalg.solve(A, b)

            # Calculate flows for this destination
            for i, j in self.G.edges():
                conductance = self.conductivity[(i, j)] / self.lengths[(i, j)]
                flow = abs(conductance * (pressures[i] - pressures[j]))
                total_flows[i, j] += flow
                total_flows[j, i] += flow

        # Update flows dictionary with accumulated values
        for i, j in self.G.edges():
            self.flows[(i, j)] = total_flows[i, j]
            self.flows[(j, i)] = total_flows[j, i]

    def update_conductivity(self) -> float:
        """Update conductivity based on flows with dynamic decay"""
        max_change = 0.0

        # Calculate average flow for adaptive thresholding
        avg_flow = np.mean(list(self.flows.values()))
        flow_threshold = avg_flow * self.params.weak_flow_threshold

        for edge in self.G.edges():
            flow = self.flows[edge]
            old_conductivity = self.conductivity[edge]

            # Dynamic decay rate based on flow strength
            decay_rate = (self.params.mu * self.params.decay_multiplier 
                        if flow < flow_threshold else self.params.mu)

            # Growth term based on flow threshold
            if flow > flow_threshold:
                growth_term = pow(flow, self.params.gamma)
            else:
                growth_term = pow(flow_threshold, self.params.gamma) * 0.5  # Reduced growth for weak flows

            # Update conductivity with dynamic decay
            new_conductivity = old_conductivity + self.params.dt * (
                growth_term - decay_rate * old_conductivity
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
        """
        Run Physarum simulation treating node 0 as depot
        """
        st.write("\n=== Physarum TSP Solver Starting ===")
        st.write(f"Number of nodes: {self.n_points}")
        st.write(f"Using node 0 as depot")

        costs = []
        best_cost = float('inf')
        best_conductivity = None
        stagnation_counter = 0

        for iteration in range(max_iterations):
            # Compute flows from depot
            self.compute_flows(depot=0)

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

def create_random_points(n_points: int, 
                       size: float = 100.0) -> np.ndarray:
    """Generate random points for testing"""
    points = np.random.rand(n_points, 2) * size
    return points
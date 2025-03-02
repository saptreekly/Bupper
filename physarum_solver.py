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
        """Compute flows using optimized sparse operations"""
        n = self.n_points
        if n <= 1:
            return sparse.csr_matrix((n, n))

        # Convert to CSR once and maintain format
        conductance = self.conductivity_matrix.multiply(1.0 / (self.distances + 1e-6))
        conductance = conductance.tocsr()

        # Verify matrix has entries
        if conductance.nnz == 0:
            st.error("Empty conductance matrix - no valid connections")
            return sparse.csr_matrix((n, n))

        # Use single thread for small problems
        if n < 20:
            total_flows = sparse.lil_matrix((n, n))
            for dest in range(1, n):
                try:
                    flow = self._compute_flow_for_dest(conductance, dest)
                    total_flows += flow
                except Exception as e:
                    st.error(f"Error computing flow for destination {dest}: {str(e)}")
                    continue
            return total_flows.tocsr()

        # Parallel processing for larger problems
        n_workers = min(4, (n - 1) // 5)  # Limit workers based on problem size
        total_flows = sparse.lil_matrix((n, n))

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for dest in range(1, n):
                futures.append(executor.submit(self._compute_flow_for_dest, conductance, dest))

            for future in as_completed(futures):
                try:
                    flow = future.result()
                    if flow is not None:
                        total_flows += flow
                except Exception as e:
                    st.error(f"Error in parallel flow computation: {str(e)}")
                    continue

        return total_flows.tocsr()

    def _compute_flow_for_dest(self, conductance: sparse.spmatrix, dest: int) -> Optional[sparse.spmatrix]:
        """Compute flow for single destination with dimension checks"""
        n = self.n_points

        # Verify conductance matrix
        if not isinstance(conductance, sparse.spmatrix):
            st.error("Invalid conductance matrix type")
            return None

        if conductance.shape != (n, n):
            st.error(f"Invalid conductance matrix dimensions")
            return None

        try:
            # Setup linear system (maintain CSR format)
            A = conductance.copy()
            diag = np.array(A.sum(axis=1)).flatten()

            # Verify diagonal array shape
            if diag.shape[0] != n:
                st.error("Invalid diagonal array dimensions")
                return None

            A = A - sparse.diags(diag, format='csr')

            # Set boundary conditions
            b = np.zeros(n)
            b[0] = 1.0
            b[dest] = -1.0

            # Use conjugate gradient solver for better performance
            try:
                pressures = sparse.linalg.cg(A, b, maxiter=1000)[0]
            except sparse.linalg.LinearOperatorError:
                # Fallback to direct solver if CG fails
                pressures = sparse.linalg.spsolve(A, b)

            # Verify pressure vector shape
            if pressures.shape[0] != n:
                st.error("Invalid pressure vector dimensions")
                return None

            # Get nonzero indices
            rows, cols = conductance.nonzero()

            # Verify indices
            if rows.size != cols.size or rows.size == 0:
                st.error("Invalid sparse matrix indices")
                return None

            # Calculate flows using sparse operations
            flow_matrix = sparse.lil_matrix((n, n))
            flows = conductance.multiply(pressures[rows] - pressures[cols])
            flow_matrix[rows, cols] = np.abs(flows.data)

            return flow_matrix.tocsr()

        except Exception as e:
            st.error(f"Flow computation error: {str(e)}")
            return None

    def update_conductivity(self, flows: sparse.spmatrix) -> float:
        """Update conductivity using vectorized operations"""
        if flows is None:
            return 0.0

        try:
            # Verify matrix shapes
            if flows.shape != (self.n_points, self.n_points):
                st.error(f"Flow matrix has wrong shape: {flows.shape}")
                return 0.0

            # Keep in CSR format throughout
            flows = flows.tocsr()
            growth = flows.power(self.params.gamma)
            decay = self.conductivity_matrix.multiply(self.params.mu)

            # Update conductivity
            delta = self.params.dt * (growth - decay)
            new_conductivity = self.conductivity_matrix + delta

            # Apply bounds efficiently
            new_conductivity.data = np.clip(
                new_conductivity.data,
                self.params.min_conductivity,
                self.params.max_conductivity
            )

            # Track maximum change
            max_change = np.max(np.abs(delta.data)) if delta.nnz > 0 else 0

            # Update matrix (maintain CSR format)
            self.conductivity_matrix = new_conductivity.tocsr()

            return max_change

        except Exception as e:
            st.error(f"Conductivity update failed: {str(e)}")
            st.write(f"Debug - Flows shape: {flows.shape}, nnz: {flows.nnz}")
            return 0.0

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
        """Run optimized Physarum simulation with progress tracking"""
        if self.n_points <= 1:
            return {}, []

        costs = []
        best_cost = float('inf')
        stagnation_count = 0
        previous_cost = float('inf')

        # Create UI elements
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Create collapsible logs section
        logs_expander = st.expander("View Physarum Logs", expanded=False)
        solver_logs = []

        def log_message(msg: str):
            solver_logs.append(f"{time.strftime('%H:%M:%S')} - {msg}")
            with logs_expander:
                st.text("\n".join(solver_logs))

        try:
            log_message("Starting Physarum optimization")
            log_message(f"Problem size: {self.n_points} nodes")

            for iteration in range(max_iterations):
                # Update progress every 5 iterations
                if iteration % 5 == 0:
                    progress = iteration / max_iterations
                    progress_bar.progress(progress)
                    if iteration % 50 == 0:  # Reduce status updates
                        status_text.text(f"Physarum Iteration {iteration}/{max_iterations}")
                        log_message(f"Completed {iteration} iterations")

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
                        if rel_improvement > 0.01:  # Log only significant improvements
                            improvement_msg = f"Cost improved by {rel_improvement*100:.1f}%"
                            status_text.text(improvement_msg)
                            log_message(improvement_msg)
                    else:
                        stagnation_count += 1
                        log_message(f"No improvement (stagnation: {stagnation_count})")

                    previous_cost = cost

                # Early stopping checks
                if max_change < self.params.convergence_threshold:
                    msg = f"Converged after {iteration} iterations"
                    status_text.text(msg)
                    log_message(msg)
                    break

                if stagnation_count >= self.params.stagnation_limit:
                    msg = f"Stopping due to stagnation after {iteration} iterations"
                    status_text.text(msg)
                    log_message(msg)
                    break

            # Log final summary
            log_message("\n=== Physarum Summary ===")
            log_message(f"Final cost: {best_cost:.2f}")
            log_message(f"Total iterations: {iteration + 1}")

            # Add download button with unique key once at the end
            log_text = "\n".join(solver_logs)
            button_key = f"physarum_logs_{int(time.time())}"
            with logs_expander:
                st.download_button(
                    "Download Logs",
                    log_text,
                    file_name="physarum_logs.txt",
                    mime="text/plain",
                    key=button_key
                )

            # Convert final conductivity to dictionary format
            rows, cols = self.conductivity_matrix.nonzero()
            conductivities = {(int(i), int(j)): float(self.conductivity_matrix[i,j])
                             for i, j in zip(rows, cols)}

            return conductivities, costs

        finally:
            # Ensure progress bar completion
            progress_bar.progress(1.0)

def create_random_points(n_points: int, 
                       size: float = 100.0) -> np.ndarray:
    """Generate random points for testing"""
    points = np.random.rand(n_points, 2) * size
    return points
import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Optional
import streamlit as st
import time
from physarum_solver import PhysarumSolver, PhysarumParams
from aco_solver import ACO
from utils import TimeWindow

class HybridSolver:
    """Hybrid solver combining Physarum and ACO approaches with configurable constraints"""

    def __init__(self,
                 points: np.ndarray,
                 time_windows: Optional[Dict[int, TimeWindow]] = None,
                 speed: float = 1.0,
                 convergence_threshold: float = 1e-4,
                 max_hybrid_iterations: int = 3,
                 enable_time_windows: bool = True,
                 enable_capacity: bool = True):
        """Initialize hybrid solver with configurable constraints"""
        self.points = points
        self.n_points = len(points)
        self.time_windows = time_windows if enable_time_windows else None
        self.speed = speed
        self.convergence_threshold = convergence_threshold
        self.max_hybrid_iterations = max_hybrid_iterations
        self.k_neighbors = 5  # Number of nearest neighbors for initial connectivity
        self.enable_time_windows = enable_time_windows
        self.enable_capacity = enable_capacity

        # Initialize Physarum solver with enhanced parameters
        physarum_params = PhysarumParams(
            gamma=1.3,  # Flow feedback strength
            mu=0.1,     # Base decay rate
            dt=0.01,    # Time step
            min_conductivity=0.01
        )
        self.physarum = PhysarumSolver(points, physarum_params, 
                                     time_windows if enable_time_windows else None, 
                                     speed)

        # Initialize ACO with modified parameters
        self.aco = ACO(
            base_evaporation=0.15,
            alpha=1.5,
            beta=2.5,
            evap_increase=0.05,
            stagnation_limit=5,
            speed=speed
        )

    def filter_network(self, conductivities: Dict, recovery_mode: bool = False) -> nx.Graph:
        """
        Filter network while preserving k-nearest neighbors for each node
        Args:
            conductivities: Edge conductivity values
            recovery_mode: If True, use a lower threshold for filtering
        """
        # Calculate adaptive threshold based on average conductivity
        values = np.array(list(conductivities.values()))
        mean_conductivity = np.mean(values)
        std_conductivity = np.std(values)

        # Adjust threshold based on mode
        threshold = mean_conductivity - (0.5 if not recovery_mode else 0.25) * std_conductivity

        # Create filtered graph
        G = nx.Graph()
        G.add_nodes_from(range(self.n_points))

        # Calculate distances between all points
        distances = np.array([[np.sqrt(np.sum((p1 - p2) ** 2))
                             for p2 in self.points]
                             for p1 in self.points])

        # First, add edges above threshold
        strong_edges = set()
        for (i, j), cond in conductivities.items():
            if cond > threshold:
                weight = 1.0 / (cond + 1e-6)  # Inverse of conductivity
                G.add_edge(i, j, weight=weight)
                strong_edges.add((i, j))
                strong_edges.add((j, i))

        # Then ensure k-nearest neighbors for each node
        for i in range(self.n_points):
            # Get current number of neighbors
            current_neighbors = len(list(G.neighbors(i)))
            if current_neighbors < self.k_neighbors:
                # Sort other nodes by distance
                nearest = np.argsort(distances[i])
                for j in nearest[1:]:  # Skip self
                    if current_neighbors >= self.k_neighbors:
                        break
                    if not G.has_edge(i, j):
                        # Add edge with weight based on distance
                        weight = distances[i][j]
                        G.add_edge(i, j, weight=weight)
                        current_neighbors += 1

        return G

    def check_convergence(self, conductivities: Dict, previous_conductivities: Dict) -> bool:
        """Check if Physarum optimization has converged"""
        if not previous_conductivities:
            return False

        changes = []
        for edge, cond in conductivities.items():
            if edge in previous_conductivities:
                rel_change = abs(cond - previous_conductivities[edge]) / (previous_conductivities[edge] + 1e-6)
                changes.append(rel_change)

        return np.mean(changes) < self.convergence_threshold

    def solve(self, demands: List[float], capacity: float) -> Tuple[List[int], float, Dict]:
        """
        Solve TSP/VRP using hybrid approach with configurable constraints
        Returns:
            best_route: Optimal route
            best_cost: Route cost
            arrival_times: Arrival times at each node
        """
        # Create main UI elements
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Create collapsible logs section
        logs_available = st.empty()  # Placeholder for notification
        logs_expander = st.expander("View Solver Logs", expanded=False)

        # Initialize log collection
        solver_logs = []
        def log_message(msg: str):
            solver_logs.append(f"{time.strftime('%H:%M:%S')} - {msg}")
            with logs_expander:
                st.text("\n".join(solver_logs))
            logs_available.info("📝 Solver logs available")

        # Initial status
        status_text.text("=== Starting Hybrid Physarum-ACO Solver ===")
        log_message("Solver initialized")
        log_message(f"Time Windows: {'Enabled' if self.enable_time_windows else 'Disabled'}")
        log_message(f"Capacity Constraints: {'Enabled' if self.enable_capacity else 'Disabled'}")

        best_route = None
        best_cost = float('inf')
        best_arrival_times = {}
        recovery_mode = False
        previous_conductivities = {}
        stagnation_counter = 0
        early_stop_threshold = 0.001  # 0.1% improvement threshold

        try:
            for iteration in range(self.max_hybrid_iterations):
                # Update progress
                progress = (iteration + 1) / self.max_hybrid_iterations
                progress_bar.progress(progress)
                status_text.text(f"Hybrid Iteration {iteration + 1}/{self.max_hybrid_iterations}")
                log_message(f"Starting iteration {iteration + 1}")

                # Run Physarum simulation with early stopping
                conductivities, _ = self.physarum.solve(max_iterations=500)

                # Check for Physarum convergence
                if previous_conductivities:
                    avg_change = self.check_convergence(conductivities, previous_conductivities)
                    if avg_change < early_stop_threshold:
                        status_text.text("Physarum optimization converged early")
                        log_message("Physarum converged - proceeding to final optimization")
                        break
                previous_conductivities = conductivities.copy()

                # Filter network based on conductivities
                filtered_network = self.filter_network(conductivities, recovery_mode)
                log_message(f"Network filtered: {filtered_network.number_of_edges()} edges remaining")

                # Run ACO on filtered network
                status_text.text("Running ACO optimization...")
                try:
                    # Prepare parameters based on enabled constraints
                    aco_params = {
                        'points': self.points,
                        'route_nodes': list(range(self.n_points)),
                        'n_iterations': 100,
                        'conductivities': conductivities
                    }

                    if self.enable_capacity:
                        aco_params.update({
                            'demands': demands,
                            'capacity': capacity
                        })

                    if self.enable_time_windows:
                        aco_params['time_windows'] = self.time_windows

                    route, cost, arrival_times = self.aco.solve(**aco_params)

                    # Update best solution
                    if cost < best_cost:
                        improvement = ((best_cost - cost) / best_cost * 100 
                                    if best_cost != float('inf') else 100)
                        status_text.text(f"Solution improved by {improvement:.1f}%")
                        log_message(f"Found better solution: cost={cost:.2f}, improvement={improvement:.1f}%")
                        best_route = route
                        best_cost = cost
                        best_arrival_times = arrival_times
                        stagnation_counter = 0

                        # Reinforce ACO solution in Physarum
                        boost_count = 0
                        for i in range(len(route) - 1):
                            edge = (route[i], route[i + 1])
                            if edge in conductivities:
                                boost = 1.5 + (best_cost - cost) / best_cost
                                conductivities[edge] *= boost
                                conductivities[(edge[1], edge[0])] = conductivities[edge]
                                boost_count += 1

                        # Penalize unused edges
                        used_edges = set((route[i], route[i+1]) for i in range(len(route)-1))
                        penalty_count = 0
                        for edge in conductivities:
                            if edge not in used_edges and (edge[1], edge[0]) not in used_edges:
                                conductivities[edge] *= 0.8
                                penalty_count += 1

                        log_message(f"Reinforced {boost_count} edges and penalized {penalty_count} edges")
                        self.physarum.conductivity = conductivities
                        recovery_mode = False
                    else:
                        stagnation_counter += 1
                        log_message(f"No improvement found (stagnation count: {stagnation_counter})")
                        if stagnation_counter >= 3:
                            if not recovery_mode:
                                status_text.text("Switching to recovery mode...")
                                log_message("Activating recovery mode")
                                recovery_mode = True
                                stagnation_counter = 0
                            else:
                                status_text.text("Terminating due to stagnation")
                                log_message("Optimization terminated due to stagnation")
                                break

                except Exception as e:
                    error_msg = f"ACO failed: {str(e)}"
                    status_text.text(error_msg)
                    log_message(error_msg)
                    if not recovery_mode:
                        recovery_msg = "Switching to recovery mode..."
                        status_text.text(recovery_msg)
                        log_message(recovery_msg)
                        recovery_mode = True
                        continue
                    break

            # Add final summary to logs
            log_message("\n=== Optimization Summary ===")
            log_message(f"Final cost: {best_cost:.2f}")
            log_message(f"Route length: {len(best_route)}")
            log_message(f"Total iterations: {iteration + 1}")

            # Add download button for logs with unique key
            log_text = "\n".join(solver_logs)
            button_key = f"hybrid_solver_logs_{int(time.time())}"
            with logs_expander:
                st.download_button(
                    "Download Logs",
                    log_text,
                    file_name="solver_logs.txt",
                    mime="text/plain",
                    key=button_key
                )

            return best_route, best_cost, best_arrival_times

        finally:
            # Ensure progress bar completion
            progress_bar.progress(1.0)
            logs_available.empty()  # Remove logs notification
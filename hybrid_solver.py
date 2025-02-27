import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Optional
import streamlit as st
from physarum_solver import PhysarumSolver, PhysarumParams
from aco_solver import ACO
from utils import TimeWindow

class HybridSolver:
    """Hybrid solver combining Physarum and ACO approaches"""

    def __init__(self,
                 points: np.ndarray,
                 time_windows: Optional[Dict[int, TimeWindow]] = None,
                 speed: float = 1.0,
                 convergence_threshold: float = 1e-4,
                 max_hybrid_iterations: int = 3):
        """Initialize hybrid solver with both Physarum and ACO components"""
        self.points = points
        self.n_points = len(points)
        self.time_windows = time_windows
        self.speed = speed
        self.convergence_threshold = convergence_threshold
        self.max_hybrid_iterations = max_hybrid_iterations
        self.k_neighbors = 5  # Minimum number of neighbors to preserve

        # Initialize Physarum solver with enhanced parameters
        physarum_params = PhysarumParams(
            gamma=1.3,  # Flow feedback strength
            mu=0.1,     # Base decay rate
            dt=0.01,    # Time step
            min_conductivity=0.01
        )
        self.physarum = PhysarumSolver(points, physarum_params, time_windows, speed)

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
                G.add_edge(i, j, weight=weight, conductivity=cond)
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
                        if (i, j) in conductivities:
                            cond = conductivities[(i, j)]
                        else:
                            cond = 1.0 / (weight + 1e-6)  # Initialize new conductivity
                        G.add_edge(i, j, weight=weight, conductivity=cond)
                        current_neighbors += 1

        return G

    def solve(self, demands: List[float], capacity: float) -> Tuple[List[int], float, Dict]:
        """
        Solve TSP/VRP using hybrid approach
        Returns:
            best_route: Optimal route
            best_cost: Route cost
            arrival_times: Arrival times at each node
        """
        st.write("\n=== Starting Hybrid Physarum-ACO Solver ===")

        best_route = None
        best_cost = float('inf')
        best_arrival_times = {}
        recovery_mode = False

        for iteration in range(self.max_hybrid_iterations):
            st.write(f"\nHybrid Iteration {iteration + 1}")

            # Run Physarum simulation
            st.write("Running Physarum optimization...")
            conductivities, _ = self.physarum.solve(max_iterations=500)

            # Filter network based on conductivities
            filtered_network = self.filter_network(conductivities, recovery_mode)

            # Initialize pheromone levels based on conductivities
            initial_pheromone = {}
            for i, j, data in filtered_network.edges(data=True):
                pheromone = data['conductivity'] / (data['weight'] + 1e-6)
                initial_pheromone[(i, j)] = pheromone
                initial_pheromone[(j, i)] = pheromone

            # Run ACO on filtered network
            st.write("Running ACO on filtered network...")
            try:
                route, cost, arrival_times = self.aco.solve(
                    self.points,
                    list(range(self.n_points)),
                    demands,
                    capacity,
                    n_iterations=100,
                    time_windows=self.time_windows,
                    initial_pheromone=initial_pheromone
                )

                # Update best solution
                if cost < best_cost:
                    improvement = ((best_cost - cost) / best_cost * 100 
                                if best_cost != float('inf') else 100)
                    st.write(f"Solution improved by {improvement:.1f}%")
                    best_route = route
                    best_cost = cost
                    best_arrival_times = arrival_times

                    # Reinforce ACO solution in Physarum with stronger feedback
                    for i in range(len(route) - 1):
                        edge = (route[i], route[i + 1])
                        if edge in conductivities:
                            # Increase reinforcement for better solutions
                            boost = 1.5 + (best_cost - cost) / best_cost
                            conductivities[edge] *= boost
                            conductivities[(edge[1], edge[0])] = conductivities[edge]

                    # Penalize unused edges
                    used_edges = set((route[i], route[i+1]) for i in range(len(route)-1))
                    for edge in conductivities:
                        if edge not in used_edges and (edge[1], edge[0]) not in used_edges:
                            conductivities[edge] *= 0.8  # Penalty for unused edges

                    self.physarum.conductivity = conductivities
                    recovery_mode = False  # Reset recovery mode after success
                else:
                    st.write("No improvement found in this iteration")
                    if not recovery_mode:
                        st.write("Switching to recovery mode...")
                        recovery_mode = True
                        continue
                    break

            except Exception as e:
                st.warning(f"ACO failed to find valid route: {str(e)}")
                if not recovery_mode:
                    st.write("Switching to recovery mode...")
                    recovery_mode = True
                    continue
                break

        return best_route, best_cost, best_arrival_times
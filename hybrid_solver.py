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

    def filter_network(self, conductivities: Dict) -> nx.Graph:
        """Filter network to keep only strong connections"""
        # Calculate adaptive threshold based on average conductivity
        values = np.array(list(conductivities.values()))
        mean_conductivity = np.mean(values)
        std_conductivity = np.std(values)
        threshold = mean_conductivity - 0.5 * std_conductivity

        # Create filtered graph
        G = nx.Graph()
        G.add_nodes_from(range(self.n_points))

        # Add edges above threshold
        for (i, j), cond in conductivities.items():
            if cond > threshold:
                # Weight is inverse of conductivity (stronger paths = shorter distances)
                weight = 1.0 / (cond + 1e-6)
                G.add_edge(i, j, weight=weight)

        # Ensure graph remains connected
        if not nx.is_connected(G):
            st.warning("Adding minimal connections to maintain connectivity")
            components = list(nx.connected_components(G))
            for i in range(len(components) - 1):
                comp1, comp2 = list(components[i]), list(components[i + 1])
                # Find shortest connection between components
                min_dist = float('inf')
                best_edge = None
                for n1 in comp1:
                    for n2 in comp2:
                        dist = np.sqrt(np.sum((self.points[n1] - self.points[n2]) ** 2))
                        if dist < min_dist:
                            min_dist = dist
                            best_edge = (n1, n2)
                if best_edge:
                    G.add_edge(*best_edge, weight=min_dist)

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

        for iteration in range(self.max_hybrid_iterations):
            st.write(f"\nHybrid Iteration {iteration + 1}")

            # Run Physarum simulation
            st.write("Running Physarum optimization...")
            conductivities, _ = self.physarum.solve(max_iterations=500)

            # Filter network based on conductivities
            filtered_network = self.filter_network(conductivities)

            # Extract edge weights for ACO
            distances = np.zeros((self.n_points, self.n_points))
            for i, j, data in filtered_network.edges(data=True):
                distances[i, j] = data['weight']
                distances[j, i] = data['weight']

            # Run ACO on filtered network
            st.write("Running ACO on filtered network...")
            route, cost, arrival_times = self.aco.solve(
                self.points,
                list(range(self.n_points)),
                demands,
                capacity,
                n_iterations=100,
                time_windows=self.time_windows
            )

            # Update best solution
            if cost < best_cost:
                improvement = ((best_cost - cost) / best_cost * 100 
                             if best_cost != float('inf') else 100)
                st.write(f"Solution improved by {improvement:.1f}%")
                best_route = route
                best_cost = cost
                best_arrival_times = arrival_times

                # Reinforce ACO solution in Physarum
                for i in range(len(route) - 1):
                    edge = (route[i], route[i + 1])
                    if edge in conductivities:
                        conductivities[edge] *= 1.5  # Boost conductivity
                        conductivities[(edge[1], edge[0])] = conductivities[edge]

                self.physarum.conductivity = conductivities
            else:
                st.write("No improvement found in this iteration")
                break

        return best_route, best_cost, best_arrival_times
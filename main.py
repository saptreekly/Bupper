import streamlit as st
import numpy as np
from aco_solver import ACO
from local_search import three_opt_improvement
from utils import generate_random_points, generate_random_time_windows
from visualizer import plot_routes
from clustering import cluster_points, check_capacity_constraints
from cross_route_optimizer import optimize_cross_route

def main():
    st.title("Vehicle Routing Problem Solver")

    st.write("""
    ### Multi-Vehicle Routing Problem Solver with Time Windows
    This application uses:
    1. K-means clustering for territory assignment
    2. Adaptive Ant Colony Optimization (ACO) with:
       - Dynamic ant count based on problem size
       - Adaptive evaporation rate
       - Penalty-based time window handling
    3. 3-opt Local Search Improvement
    4. Capacity constraints for each vehicle
    5. Time window constraints for each location
    """)

    # Input parameters
    st.sidebar.header("Problem Parameters")

    # Instance size
    st.sidebar.subheader("Instance Size")
    n_points = st.sidebar.slider("Number of Points", 10, 100, 20,
        help="Total number of delivery points (excluding depot)")
    n_vehicles = st.sidebar.slider("Number of Vehicles", 2, 10, 3,
        help="Number of available vehicles (clusters)")

    # Time Window Parameters
    st.sidebar.subheader("Time Window Parameters")
    time_horizon = st.sidebar.slider("Time Horizon", 50.0, 200.0, 100.0,
        help="Maximum planning horizon")
    min_window = st.sidebar.slider("Min Time Window", 5.0, 30.0, 10.0,
        help="Minimum width of time windows")
    max_window = st.sidebar.slider("Max Time Window", 20.0, 60.0, 30.0,
        help="Maximum width of time windows")

    # Vehicle Parameters
    st.sidebar.subheader("Vehicle Parameters")
    vehicle_capacity = st.sidebar.slider("Vehicle Capacity", 10, 100, 50,
        help="Maximum capacity for each vehicle")
    vehicle_speed = st.sidebar.slider("Vehicle Speed", 0.1, 5.0, 1.0,
        help="Travel speed (distance/time unit)")

    # ACO Parameters
    st.sidebar.subheader("ACO Parameters")
    time_penalty_factor = st.sidebar.slider("Time Window Penalty Factor", 1.0, 5.0, 2.0,
        help="Penalty multiplier for time window violations")

    if st.button("Generate and Solve VRP"):
        try:
            # Generate random points and time windows
            points = generate_random_points(n_points + 1)  # +1 for depot
            time_windows = generate_random_time_windows(
                n_points + 1,
                horizon=time_horizon,
                min_window=min_window,
                max_window=max_window
            )

            st.write("\n=== Points Array Debug Info ===")
            st.write(f"Global points array shape: {points.shape}")

            # Cluster points
            with st.spinner("Clustering points..."):
                route_indices, labels = cluster_points(points, n_vehicles)

            st.write("\n=== Route Indices Debug Info ===")
            for i, route in enumerate(route_indices):
                st.write(f"Initial route {i}: {route}")

            # Initialize ACO solver
            aco = ACO(base_evaporation=0.15,
                     alpha=1.5,
                     beta=2.5,
                     evap_increase=0.05,
                     stagnation_limit=5,
                     speed=vehicle_speed,
                     time_penalty_factor=time_penalty_factor)

            # Solve for each cluster
            all_routes = []
            all_lengths = []
            all_arrival_times = []

            with st.spinner("Solving routes for each vehicle..."):
                for route_nodes in route_indices:
                    if len(route_nodes) < 2:  # Skip empty routes
                        all_routes.append([])
                        all_lengths.append(0)
                        all_arrival_times.append({})
                        continue

                    # Simple demand model: each point has demand of 1
                    demands = [1] * len(points)  # Global demands array

                    # Check capacity constraint
                    if not check_capacity_constraints(route_nodes, demands, vehicle_capacity):
                        st.warning(f"Route exceeds vehicle capacity!")
                        continue

                    # Solve TSP for this cluster using global indices
                    route, cost, arrival_times = aco.solve(
                        points, 
                        route_nodes,
                        n_iterations=100,
                        time_windows=time_windows
                    )

                    all_routes.append(route)
                    all_lengths.append(cost)
                    all_arrival_times.append(arrival_times)

            st.write("\n=== Final Routes Debug Info ===")
            for i, route in enumerate(all_routes):
                st.write(f"Final route {i}: {route}")

            # Display results
            st.subheader("Results")

            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Routes", len([r for r in all_routes if r]))
                for i, length in enumerate(all_lengths):
                    if length > 0:
                        st.metric(f"Route {i+1} Cost", round(length, 2))

            with col2:
                st.metric("Total Points", len(points))
                for i, route in enumerate(all_routes):
                    if route:
                        st.metric(f"Route {i+1} Points", len(route))

            with col3:
                st.metric("Total Cost", round(sum(all_lengths), 2))
                st.metric("Speed", vehicle_speed)

            # Visualization - Always use full global points array
            st.subheader("Route Visualization")
            plot_routes(points, all_routes, labels,
                     "Vehicle Routes (K-means + ACO)")

            # Time window analysis
            if time_windows:
                st.subheader("Time Window Analysis")
                for route_idx, (route, arrival_times) in enumerate(zip(all_routes, all_arrival_times)):
                    if not route:
                        continue

                    st.write(f"Route {route_idx + 1} Schedule:")
                    violations = []
                    for node in route:
                        arrival = arrival_times.get(node, 0)
                        if node in time_windows:
                            tw = time_windows[node]
                            status = "✅ On time"
                            if arrival < tw.earliest:
                                status = "⏳ Early (waiting)"
                            elif arrival > tw.latest:
                                status = "⚠️ Late"
                                violations.append((node, arrival - tw.latest))

                            st.write(
                                f"Node {node}: Arrival={round(arrival, 1)}, "
                                f"Window=[{round(tw.earliest, 1)}, {round(tw.latest, 1)}] "
                                f"- {status}"
                            )

                    if violations:
                        st.warning(
                            f"Route {route_idx + 1} has {len(violations)} time window "
                            f"violations. Total delay: {round(sum(v[1] for v in violations), 1)} units"
                        )

        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()
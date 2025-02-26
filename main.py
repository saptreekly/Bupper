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
    base_ants = st.sidebar.slider("Base Number of Ants", 5, 50, 20,
        help="Base value for calculating number of ants (adjusted by problem size)")
    n_iterations = st.sidebar.slider("Number of Iterations", 10, 200, 100)
    base_evaporation = st.sidebar.slider("Base Evaporation Rate", 0.01, 0.5, 0.1)

    # Advanced Parameters
    st.sidebar.subheader("Advanced Parameters")
    with st.sidebar.expander("Advanced Settings"):
        alpha = st.slider("Alpha (Pheromone Importance)", 0.1, 5.0, 1.0)
        evap_increase = st.slider("Evaporation Rate Increase", 0.01, 0.2, 0.05)
        stagnation_limit = st.slider("Stagnation Limit", 2, 10, 5)
        enable_cross_route = st.checkbox("Enable Cross-Route Optimization", True,
            help="Try to improve solution by moving nodes between routes")
        if enable_cross_route:
            max_cross_iterations = st.slider("Max Cross-Route Iterations", 10, 200, 100,
                help="Maximum number of cross-route improvement attempts")

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

            # Cluster points
            with st.spinner("Clustering points..."):
                clustered_points, labels = cluster_points(points, n_vehicles)

            # Initialize ACO solver
            aco = ACO(base_ants=base_ants,
                     base_evaporation=base_evaporation,
                     alpha=alpha,
                     evap_increase=evap_increase,
                     stagnation_limit=stagnation_limit,
                     speed=vehicle_speed)

            # Solve for each cluster
            all_routes = []
            all_lengths = []

            with st.spinner("Solving routes for each vehicle..."):
                for i, cluster_points_array in enumerate(clustered_points):
                    if len(cluster_points_array) < 2:
                        all_routes.append([])
                        all_lengths.append(0)
                        continue

                    # Get indices of points in this cluster
                    cluster_indices = np.where(labels == i)[0]

                    # Simple demand model: each point has demand of 1
                    demands = [1] * len(cluster_indices)

                    # Check capacity constraint
                    if not check_capacity_constraints(cluster_points_array, demands, vehicle_capacity):
                        st.warning(f"Cluster {i} exceeds vehicle capacity!")
                        continue

                    # Get time windows for cluster points
                    cluster_time_windows = {
                        idx: time_windows[cluster_indices[idx]]
                        for idx in range(len(cluster_indices))
                        if cluster_indices[idx] in time_windows
                    }

                    # Solve TSP for this cluster
                    route, length = aco.solve(cluster_points_array, n_iterations,
                                           cluster_time_windows)

                    # Convert route indices back to original point indices
                    original_route = [cluster_indices[idx] for idx in route]

                    # Apply 3-opt improvement
                    distances = aco.calculate_distances(points[cluster_indices])
                    improved_route, improved_length = three_opt_improvement(
                        route, distances, cluster_time_windows, vehicle_speed)

                    # Convert improved route to original indices
                    improved_original_route = [cluster_indices[idx] for idx in improved_route]

                    all_routes.append(improved_original_route)
                    all_lengths.append(improved_length)

            # Apply cross-route optimization if enabled
            if enable_cross_route:
                with st.spinner("Performing cross-route optimization..."):
                    # Prepare demands (simple model: each point has demand of 1)
                    demands = [1] * len(points)

                    improved_routes, total_distance = optimize_cross_route(
                        all_routes,
                        aco.calculate_distances(points),
                        demands,
                        vehicle_capacity,
                        time_windows,
                        vehicle_speed,
                        max_cross_iterations
                    )

                    # Update routes if improvement found
                    if total_distance < sum(all_lengths):
                        all_routes = improved_routes
                        all_lengths = [sum(aco.calculate_distances(points)[route[i]][route[i+1]]
                                         for i in range(len(route)-1))
                                     for route in improved_routes]
                        st.success("Cross-route optimization improved the solution!")

            # Display results
            st.subheader("Results")

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Routes", len([r for r in all_routes if r]))
                for i, length in enumerate(all_lengths):
                    if length > 0:
                        st.metric(f"Route {i+1} Length", round(length, 2))

            with col2:
                st.metric("Total Points", len(points))
                for i, route in enumerate(all_routes):
                    if route:
                        st.metric(f"Route {i+1} Points", len(route))

            with col3:
                st.metric("Speed", vehicle_speed)
                if time_windows:
                    st.metric("Time Windows", len(time_windows))

            with col4:
                st.metric("Total Distance", round(sum(all_lengths), 2))
                if enable_cross_route:
                    st.metric("Cross-Route Iterations", max_cross_iterations)

            # Visualization
            st.subheader("Route Visualization")
            plot_routes(points, all_routes, labels,
                     "Vehicle Routes (K-means + ACO + 3-opt)")

            # Time window details
            if time_windows:
                st.subheader("Time Window Details")
                st.write("Time windows for each node:")
                for node, tw in time_windows.items():
                    st.write(f"Node {node}: [{round(tw.earliest, 1)}, {round(tw.latest, 1)}] "
                            f"(Service time: {round(tw.service_time, 1)})")

        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()
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

    # Add new parameters to sidebar
    st.sidebar.subheader("Optimization Parameters")
    repair_iterations = st.sidebar.slider("Max Repair Iterations", 10, 200, 50,
        help="Maximum number of iterations for time window repair")
    time_penalty_factor = st.sidebar.slider("Time Window Penalty Factor", 1.0, 10.0, 2.0,
        help="Penalty multiplier for time window violations")
    repair_threshold = st.sidebar.slider("Repair Cost Threshold", 0.1, 2.0, 0.5,
        help="Maximum allowed cost increase during repair (multiplier)")

    # Advanced ACO Parameters
    st.sidebar.subheader("Advanced ACO Parameters")
    parallel_ants = st.sidebar.slider("Parallel Ant Processes", 1, 8, 4,
        help="Number of parallel ant construction processes")
    alns_frequency = st.sidebar.slider("ALNS Frequency", 5, 20, 10,
        help="Apply ALNS every N iterations")

    # Added Debug Options
    st.sidebar.subheader("Debug Options")
    verbose_logging = st.sidebar.checkbox("Enable Detailed Logging", False,
        help="Show detailed optimization progress (may slow down UI)")


    if st.button("Generate and Solve VRP"):
        try:
            # Generate global points array at the start
            global_points = generate_random_points(n_points + 1)  # +1 for depot

            st.write("\n=== Global Points Array Info ===")
            st.write(f"Global points array shape: {global_points.shape}")
            st.write(f"Number of points (including depot): {len(global_points)}")

            # Generate time windows using global indices
            time_windows = generate_random_time_windows(
                n_points + 1,
                horizon=time_horizon,
                min_window=min_window,
                max_window=max_window
            )

            # Cluster points - keep original points array intact
            with st.spinner("Clustering points..."):
                route_indices, labels = cluster_points(global_points, n_vehicles)

            st.write("\n=== Initial Routes Debug Info ===")
            for i, route in enumerate(route_indices):
                st.write(f"Initial route {i}: {route}")
                if route:
                    st.write(f"Max index in route {i}: {max(route)}")

            # Initialize ACO solver with new parameters
            aco = ACO(base_evaporation=0.15,
                      alpha=1.5,
                      beta=2.5,
                      evap_increase=0.05,
                      stagnation_limit=5,
                      speed=vehicle_speed,
                      time_penalty_factor=time_penalty_factor,
                      max_parallel_ants=parallel_ants,
                      verbose=verbose_logging)  # Add verbosity control

            # Store demands and capacity for verification
            aco.demands = [1.0] * len(global_points)  # Simple demand model
            aco.capacity = vehicle_capacity

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
                    demands = [1.0] * len(global_points)  # Use float for demands

                    # Check capacity constraint
                    if not check_capacity_constraints(route_nodes, demands, vehicle_capacity):
                        st.warning(f"Route exceeds vehicle capacity!")
                        continue

                    # Solve TSP for this cluster using global indices
                    route, cost, arrival_times = aco.solve(
                        global_points,  # Always use global points array
                        route_nodes,
                        demands,
                        vehicle_capacity,
                        n_iterations=100,
                        time_windows=time_windows,
                        alns_frequency=alns_frequency
                    )

                    all_routes.append(route)
                    all_lengths.append(cost)
                    all_arrival_times.append(arrival_times)

            st.write("\n=== Final Routes Debug Info ===")
            for i, route in enumerate(all_routes):
                st.write(f"Final route {i}: {route}")
                if route:
                    st.write(f"Max index in route {i}: {max(route)}")
                    if max(route) >= len(global_points):
                        st.error(f"Route {i} contains invalid index: {max(route)} >= {len(global_points)}")

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
                st.metric("Total Points", len(global_points))  # Use global_points consistently
                for i, route in enumerate(all_routes):
                    if route:
                        st.metric(f"Route {i+1} Points", len(route))

            with col3:
                st.metric("Total Cost", round(sum(all_lengths), 2))
                st.metric("Speed", vehicle_speed)

            # Visualization - Always use full global points array
            st.subheader("Route Visualization")
            plot_routes(global_points, all_routes, labels,
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
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)  # This will show the full traceback

if __name__ == "__main__":
    main()
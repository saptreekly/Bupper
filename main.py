import streamlit as st
import numpy as np
from physarum_solver import PhysarumSolver, PhysarumParams, create_random_points
from aco_solver import ACO
from local_search import three_opt_improvement
from utils import generate_random_points, generate_random_time_windows
from visualizer import plot_routes
from clustering import cluster_points, check_capacity_constraints
from cross_route_optimizer import optimize_cross_route
from benchmark_utils import BenchmarkManager
import matplotlib.pyplot as plt

def verify_and_fix_routes(routes, num_points, distances, demands, capacity, time_windows, speed, max_repair_iterations=50, cost_increase_threshold=0.2, time_penalty_multiplier=3.0):
    fixed_routes = []
    for route in routes:
        total_demand = sum(demands[i] for i in route)
        if total_demand > capacity:
            original_cost = sum(distances[route[i]][route[i+1]] for i in range(len(route)-1))
            best_route = route[:]
            best_cost = original_cost
            for _ in range(max_repair_iterations):
                removed_node = route.pop()
                new_cost = sum(distances[route[i]][route[i+1]] for i in range(len(route)-1))
                if new_cost <= best_cost * (1 + cost_increase_threshold):
                    best_route = route[:]
                    best_cost = new_cost
                else:
                    route.append(removed_node)
                total_demand = sum(demands[i] for i in route)
                if total_demand <= capacity:
                    break

            fixed_routes.append(best_route)

        else:
            fixed_routes.append(route)
    return fixed_routes

def main():
    st.title("Network Optimization Solvers")

    # Add solver selection
    solver_type = st.sidebar.selectbox(
        "Select Solver Type",
        ["Vehicle Routing (ACO)", "Network Design (Physarum)"]
    )

    if solver_type == "Vehicle Routing (ACO)":
        st.write("""
        ### Multi-Vehicle Routing Problem Solver with Time Windows
        This application uses:
        1. Automatic vehicle count determination
        2. K-means clustering for territory assignment
        3. Adaptive Ant Colony Optimization (ACO)
        4. 3-opt Local Search Improvement
        5. Capacity constraints for each vehicle
        6. Time window constraints for each location
        """)

        # Input parameters
        st.sidebar.header("Problem Parameters")

        # Instance size
        st.sidebar.subheader("Instance Size")
        n_points = st.sidebar.slider("Number of Points", 10, 1000, 50,
            help="Total number of delivery points (excluding depot)")

        # Vehicle Parameters
        st.sidebar.subheader("Vehicle Parameters")
        vehicle_capacity = st.sidebar.slider("Vehicle Capacity", 10, 200, 100,
            help="Maximum capacity for each vehicle")
        vehicle_speed = st.sidebar.slider("Vehicle Speed", 0.1, 5.0, 1.0,
            help="Travel speed (distance/time unit)")

        # Remove manual vehicle count override since it's now fully automatic
        st.sidebar.markdown("""
        💡 The number of vehicles will be automatically determined based on:
        - Total demand vs vehicle capacity
        - Number of delivery points
        - Maximum route duration
        """)

        # Time Window Parameters
        st.sidebar.subheader("Time Window Parameters")
        time_horizon = st.sidebar.slider("Time Horizon", 50.0, 200.0, 100.0,
            help="Maximum planning horizon")
        min_window = st.sidebar.slider("Min Time Window", 5.0, 30.0, 10.0,
            help="Minimum width of time windows")
        max_window = st.sidebar.slider("Max Time Window", 20.0, 60.0, 30.0,
            help="Maximum width of time windows")

        # Add new parameters to sidebar
        st.sidebar.subheader("Optimization Parameters")
        repair_iterations = st.sidebar.slider("Max Repair Iterations", 10, 200, 50,
            help="Maximum number of iterations for time window repair")
        time_penalty_factor = st.sidebar.slider("Time Window Penalty Factor", 1.0, 10.0, 2.0,
            help="Penalty multiplier for time window violations")
        repair_threshold = st.sidebar.slider("Repair Cost Threshold", 0.1, 2.0, 0.5,
            help="Maximum allowed cost increase during repair (multiplier)")

        # Add new repair parameters to sidebar
        st.sidebar.subheader("Route Repair Parameters")
        max_repair_iterations = st.sidebar.slider(
            "Max Repair Iterations", 2, 10, 5,
            help="Maximum attempts to improve each repair step")
        cost_increase_threshold = st.sidebar.slider(
            "Max Cost Increase", 0.1, 0.5, 0.2,
            help="Maximum allowed proportional cost increase during repair")
        time_penalty_multiplier = st.sidebar.slider(
            "Time Penalty Factor", 1.0, 5.0, 3.0,
            help="Penalty multiplier for time window violations during repair")


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

        # Cross-Route Optimization Parameters
        st.sidebar.subheader("Cross-Route Parameters")
        allow_capacity_overflow = st.sidebar.slider(
            "Capacity Overflow Allowance", 0.0, 0.3, 0.1,
            help="Maximum allowed capacity overflow as fraction of vehicle capacity")
        allow_time_violation = st.sidebar.checkbox(
            "Allow Time Window Violations", True,
            help="Allow slight violations of time windows with penalty")
        time_violation_penalty = st.sidebar.slider(
            "Time Violation Penalty", 1.0, 5.0, 1.5,
            help="Penalty factor for time window violations")
        capacity_penalty = st.sidebar.slider(
            "Capacity Violation Penalty", 1.0, 5.0, 2.0,
            help="Penalty factor for capacity violations")

        st.sidebar.subheader("Benchmarking")
        run_benchmarks = st.sidebar.checkbox("Run Benchmarks", False)

        if st.button("Generate and Solve VRP"):
            try:
                # Generate points and time windows
                global_points = generate_random_points(n_points + 1)  # +1 for depot
                time_windows = generate_random_time_windows(
                    n_points + 1,
                    horizon=time_horizon,
                    min_window=min_window,
                    max_window=max_window
                )

                # Initialize demands
                demands = [1.0] * len(global_points)  # Simple demand model

                # Cluster points with automatic vehicle determination
                with st.spinner("Clustering points..."):
                    route_indices, labels = cluster_points(
                        global_points,
                        n_clusters=None, # use_auto_vehicles is always true now
                        demands=demands,
                        capacity=vehicle_capacity
                    )

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
                aco.demands = demands  # Store for verification
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

                # Verify and fix final routes
                st.subheader("Route Verification")
                final_routes = verify_and_fix_routes(
                    all_routes,
                    len(global_points),
                    distances=np.array([[np.sqrt(np.sum((p1 - p2) ** 2))
                                       for p2 in global_points]
                                       for p1 in global_points]),
                    demands=demands,
                    capacity=vehicle_capacity,
                    time_windows=time_windows,
                    speed=vehicle_speed,
                    max_repair_iterations=max_repair_iterations,
                    cost_increase_threshold=cost_increase_threshold,
                    time_penalty_multiplier=time_penalty_multiplier
                )

                # Update routes and recalculate costs
                all_routes = final_routes
                all_lengths = [sum(np.sqrt(np.sum((global_points[r[i]] - global_points[r[i+1]]) ** 2))
                                 for i in range(len(r)-1))
                             for r in all_routes]

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

        if run_benchmarks:
            st.title("VRP Solver Benchmarking")

            benchmark_manager = BenchmarkManager()

            # Configure benchmark parameters
            n_runs = st.slider("Number of runs per instance", 1, 5, 3)

            st.subheader("Available Benchmark Sets")

            # Random instance generation
            if st.checkbox("Generate Random Instances"):
                sizes = st.multiselect(
                    "Select instance sizes",
                    options=[25, 50, 100, 200, 500, 1000],
                    default=[100, 200]
                )

                if st.button("Run Random Benchmarks"):
                    with st.spinner("Running benchmarks..."):
                        for size in sizes:
                            points, time_windows = benchmark_manager.generate_random_instance(size)
                            result = benchmark_manager.run_benchmark(
                                aco, f"random_{size}", points, time_windows, n_runs
                            )
                            benchmark_manager.results.append(result)

                    # Plot results
                    benchmark_manager.plot_results()

            # Solomon instances
            if st.checkbox("Run Solomon VRPTW Benchmarks"):
                solomon_urls = {
                    "C101_25": "http://example.com/solomon/C101_25.txt",  # Replace with actual URLs
                    "C101_50": "http://example.com/solomon/C101_50.txt",
                    "C101_100": "http://example.com/solomon/C101_100.txt"
                }

                selected_instances = st.multiselect(
                    "Select Solomon instances",
                    options=list(solomon_urls.keys()),
                    default=["C101_25"]
                )

                if st.button("Run Solomon Benchmarks"):
                    with st.spinner("Running Solomon benchmarks..."):
                        for instance in selected_instances:
                            points, time_windows = benchmark_manager.load_solomon_instance(
                                solomon_urls[instance]
                            )
                            if points is not None:
                                result = benchmark_manager.run_benchmark(
                                    aco, instance, points, time_windows, n_runs
                                )
                                benchmark_manager.results.append(result)

                    # Plot results
                    benchmark_manager.plot_results()

    else:  # Physarum solver
        st.write("""
        ### Physarum-inspired Network Design
        This solver simulates how Physarum polycephalum (slime mold) forms efficient transport networks.
        The model uses:
        1. Flow computation based on Kirchhoff's laws
        2. Adaptive conductivity updates
        3. Biologically-inspired growth and decay mechanisms
        """)

        # Simulation parameters
        st.sidebar.subheader("Physarum Parameters")
        n_points = st.sidebar.slider("Number of Points", 5, 50, 10,
            help="Number of nodes in the network")

        gamma = st.sidebar.slider("Growth Rate (γ)", 0.5, 2.0, 1.3,
            help="Flow feedback strength")
        mu = st.sidebar.slider("Decay Rate (μ)", 0.01, 0.5, 0.1,
            help="Rate of tube decay")
        dt = st.sidebar.slider("Time Step (dt)", 0.001, 0.1, 0.01,
            help="Simulation time step")

        max_iterations = st.sidebar.slider("Max Iterations", 100, 2000, 500,
            help="Maximum number of simulation steps")

        if st.button("Run Physarum Simulation"):
            try:
                # Generate random points
                points = create_random_points(n_points)

                # Initialize solver
                params = PhysarumParams(gamma=gamma, mu=mu, dt=dt)
                solver = PhysarumSolver(points, params)

                # Set source and sink nodes (first and last points)
                source, sink = 0, n_points-1

                # Run simulation with progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                plot_container = st.empty()
                metrics_container = st.empty()

                # Run solver
                with st.spinner("Running simulation..."):
                    best_conductivity, costs = solver.solve(max_iterations)

                    # Show final network state
                    fig = solver.visualize_network(max_iterations)
                    st.pyplot(fig)

                    # Plot cost evolution
                    st.subheader("Network Cost Evolution")
                    cost_fig, ax = plt.subplots()
                    ax.plot(costs)
                    ax.set_xlabel("Iteration")
                    ax.set_ylabel("Network Cost")
                    ax.set_yscale('log')
                    st.pyplot(cost_fig)

                    # Extract final route
                    final_route = solver.extract_route(best_conductivity)
                    st.subheader("Final Route")
                    st.write(f"Route: {final_route}")
                    st.write(f"Route length: {len(final_route)} nodes")

                    # Display metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Final Cost", f"{costs[-1]:.2f}")
                    with col2:
                        improvement = (costs[0] - costs[-1]) / costs[0] * 100
                        st.metric("Cost Reduction", f"{improvement:.1f}%")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.exception(e)

if __name__ == "__main__":
    main()
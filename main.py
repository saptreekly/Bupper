import streamlit as st
import numpy as np
from aco_solver import ACO
from local_search import three_opt_improvement
from utils import validate_points, parse_input_string
from visualizer import plot_routes
from clustering import cluster_points, check_capacity_constraints

def main():
    st.title("Vehicle Routing Problem Solver")

    st.write("""
    ### Multi-Vehicle Routing Problem Solver
    This application uses:
    1. K-means clustering for territory assignment
    2. Adaptive Ant Colony Optimization (ACO) with:
       - Dynamic ant count based on problem size
       - Adaptive evaporation rate
    3. 3-opt Local Search Improvement
    4. Capacity constraints for each vehicle

    Enter coordinates in the format: x1,y1;x2,y2;x3,y3;...
    """)

    # Input parameters
    st.sidebar.header("Algorithm Parameters")

    # Vehicle Parameters
    st.sidebar.subheader("Vehicle Parameters")
    n_vehicles = st.sidebar.slider("Number of Vehicles", 2, 10, 3,
        help="Number of available vehicles (clusters)")
    vehicle_capacity = st.sidebar.slider("Vehicle Capacity", 10, 100, 50,
        help="Maximum capacity for each vehicle")

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

    # Input coordinates
    input_text = st.text_area(
        "Enter coordinates (x,y pairs separated by semicolons):",
        value="0,0;2,2;1,5;5,2;6,6;8,3;4,4;7,1;3,3;5,5",
        help="Example: 0,0;2,2;1,5;5,2;6,6;8,3"
    )

    if st.button("Solve VRP"):
        try:
            # Parse and validate input
            points_list = parse_input_string(input_text)
            points = validate_points(points_list)

            if len(points) < n_vehicles * 2:
                st.error(f"Please enter at least {n_vehicles * 2} points.")
                return

            # Cluster points
            with st.spinner("Clustering points..."):
                clustered_points, labels = cluster_points(points, n_vehicles)

            # Initialize ACO solver
            aco = ACO(base_ants=base_ants,
                     base_evaporation=base_evaporation,
                     alpha=alpha,
                     evap_increase=evap_increase,
                     stagnation_limit=stagnation_limit)

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

                    # Solve TSP for this cluster
                    route, length = aco.solve(cluster_points_array, n_iterations)

                    # Convert route indices back to original point indices
                    original_route = [cluster_indices[idx] for idx in route]

                    # Apply 3-opt improvement
                    distances = aco.calculate_distances(points[cluster_indices])
                    improved_route, improved_length = three_opt_improvement(
                        route, distances)

                    # Convert improved route to original indices
                    improved_original_route = [cluster_indices[idx] for idx in improved_route]

                    all_routes.append(improved_original_route)
                    all_lengths.append(improved_length)

            # Display results
            st.subheader("Results")

            # Summary metrics
            col1, col2 = st.columns(2)
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

            # Visualization
            st.subheader("Route Visualization")
            plot_routes(points, all_routes, labels,
                     "Vehicle Routes (K-means + ACO + 3-opt)")

        except ValueError as e:
            st.error(f"Error: {str(e)}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()
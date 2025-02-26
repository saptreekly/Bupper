import streamlit as st
import numpy as np
from aco_solver import ACO
from local_search import three_opt_improvement
from utils import validate_points, parse_input_string
from visualizer import plot_route

def main():
    st.title("Adaptive TSP Solver: ACO with 3-opt Improvement")

    st.write("""
    ### Traveling Salesman Problem Solver
    This application uses:
    1. Adaptive Ant Colony Optimization (ACO)
       - Dynamic ant count based on problem size
       - Adaptive evaporation rate
    2. 3-opt Local Search Improvement

    Enter coordinates in the format: x1,y1;x2,y2;x3,y3;...
    """)

    # Input parameters
    st.sidebar.header("Algorithm Parameters")

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
        value="0,0;2,2;1,5;5,2;6,6;8,3",
        help="Example: 0,0;2,2;1,5;5,2;6,6;8,3"
    )

    if st.button("Solve TSP"):
        try:
            # Parse and validate input
            points_list = parse_input_string(input_text)
            points = validate_points(points_list)

            if len(points) < 3:
                st.error("Please enter at least 3 points.")
                return

            # Initialize and run ACO
            with st.spinner("Running adaptive ACO algorithm..."):
                aco = ACO(base_ants=base_ants,
                         base_evaporation=base_evaporation,
                         alpha=alpha,
                         evap_increase=evap_increase,
                         stagnation_limit=stagnation_limit)

                best_path, best_length = aco.solve(points, n_iterations)

                # Apply 3-opt improvement
                distances = aco.calculate_distances(points)
                improved_path, improved_length = three_opt_improvement(
                    best_path, distances)

                # Display results
                st.subheader("Results")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Actual Ants Used", 
                            int(np.log2(len(points)) * base_ants))
                with col2:
                    st.metric("Initial ACO Length", 
                            round(best_length, 2))
                with col3:
                    st.metric("After 3-opt Length", 
                            round(improved_length, 2))

                # Visualization
                st.subheader("Route Visualization")
                plot_route(points, improved_path, 
                         "Optimized Route (Adaptive ACO + 3-opt)")

        except ValueError as e:
            st.error(f"Error: {str(e)}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()
import streamlit as st
import numpy as np
from aco_solver import ACO
from local_search import two_opt_improvement
from utils import validate_points, parse_input_string
from visualizer import plot_route

def main():
    st.title("TSP Solver: ACO with 2-opt Improvement")
    
    st.write("""
    ### Traveling Salesman Problem Solver
    This application solves the Traveling Salesman Problem using:
    1. Ant Colony Optimization (ACO)
    2. 2-opt Local Search Improvement
    
    Enter coordinates in the format: x1,y1;x2,y2;x3,y3;...
    """)
    
    # Input parameters
    col1, col2 = st.columns(2)
    with col1:
        n_ants = st.slider("Number of Ants", 5, 50, 20)
        n_iterations = st.slider("Number of Iterations", 10, 200, 100)
    with col2:
        evaporation_rate = st.slider("Evaporation Rate", 0.01, 0.5, 0.1)
        alpha = st.slider("Alpha (Pheromone Importance)", 0.1, 5.0, 1.0)
    
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
            with st.spinner("Running ACO algorithm..."):
                aco = ACO(n_ants=n_ants,
                         evaporation_rate=evaporation_rate,
                         alpha=alpha)
                
                best_path, best_length = aco.solve(points, n_iterations)
                
                # Apply 2-opt improvement
                distances = aco.calculate_distances(points)
                improved_path, improved_length = two_opt_improvement(
                    best_path, distances)
                
                # Display results
                st.subheader("Results")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Initial ACO Solution Length:", 
                            round(best_length, 2))
                with col2:
                    st.write("After 2-opt Improvement:", 
                            round(improved_length, 2))
                
                # Visualization
                st.subheader("Route Visualization")
                plot_route(points, improved_path, 
                         "Optimized Route (ACO + 2-opt)")
                
        except ValueError as e:
            st.error(f"Error: {str(e)}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()

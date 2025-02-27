import streamlit as st
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import requests
import io
import time
from dataclasses import dataclass
from pathlib import Path
from utils import TimeWindow  # Add TimeWindow import

@dataclass
class BenchmarkResult:
    """Store benchmark results for a single instance"""
    instance_name: str
    n_points: int
    n_vehicles: int
    solution_cost: float
    runtime: float
    iterations: int
    best_known: Optional[float] = None
    gap_to_best: Optional[float] = None

class BenchmarkManager:
    """Manage benchmarking datasets and results"""

    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.known_solutions = self._load_known_solutions()

    def _load_known_solutions(self) -> Dict[str, float]:
        """Load known optimal/best solutions for benchmark instances"""
        # Example format - expand with actual benchmark data
        return {
            'solomon_25_1': 1000.0,
            'solomon_50_1': 2000.0,
            'solomon_100_1': 3000.0
        }

    def load_solomon_instance(self, url: str) -> Tuple[np.ndarray, Dict[int, TimeWindow]]:
        """Load Solomon VRPTW instance from URL"""
        try:
            response = requests.get(url)
            response.raise_for_status()

            # Parse Solomon format
            lines = response.text.strip().split('\n')
            n_vehicles = int(lines[4].split()[0])
            capacity = float(lines[4].split()[1])

            points = []
            time_windows = {}

            # Skip header lines
            for line in lines[9:]:
                data = line.split()
                if len(data) >= 7:
                    cust_no = int(data[0])
                    x = float(data[1])
                    y = float(data[2])
                    demand = float(data[3])
                    ready_time = float(data[4])
                    due_time = float(data[5])
                    service_time = float(data[6])

                    points.append([x, y])
                    time_windows[cust_no] = TimeWindow(
                        earliest=ready_time,
                        latest=due_time,
                        service_time=service_time
                    )

            return np.array(points), time_windows

        except Exception as e:
            st.error(f"Error loading Solomon instance: {str(e)}")
            return None, None

    def generate_random_instance(self, 
                               n_points: int,
                               size: float = 100.0) -> Tuple[np.ndarray, Dict[int, TimeWindow]]:
        """Generate random benchmark instance"""
        points = np.random.rand(n_points, 2) * size
        points[0] = [size/2, size/2]  # Center depot

        time_windows = {}
        horizon = size * 2  # Time horizon based on space size

        for i in range(n_points):
            earliest = np.random.uniform(0, horizon/2)
            width = np.random.uniform(horizon/4, horizon/2)
            time_windows[i] = TimeWindow(
                earliest=earliest,
                latest=earliest + width,
                service_time=10.0
            )

        return points, time_windows

    def run_benchmark(self, solver, instance_name: str, 
                     points: np.ndarray,
                     time_windows: Dict[int, TimeWindow],
                     n_runs: int = 3) -> BenchmarkResult:
        """Run benchmark on given instance multiple times"""
        total_runtime = 0
        best_cost = float('inf')
        total_iterations = 0

        for run in range(n_runs):
            start_time = time.time()
            route, cost, _ = solver.solve(points, list(range(len(points))), 
                                        [1.0]*len(points), 100.0,
                                        time_windows=time_windows)
            runtime = time.time() - start_time

            total_runtime += runtime
            best_cost = min(best_cost, cost)
            total_iterations += solver.last_improvement_iteration

        avg_runtime = total_runtime / n_runs
        avg_iterations = total_iterations / n_runs

        # Calculate gap to best known solution if available
        best_known = self.known_solutions.get(instance_name)
        gap = None
        if best_known:
            gap = ((best_cost - best_known) / best_known) * 100

        return BenchmarkResult(
            instance_name=instance_name,
            n_points=len(points),
            n_vehicles=len(set(route[1:-1])),  # Count unique nodes excluding depot
            solution_cost=best_cost,
            runtime=avg_runtime,
            iterations=avg_iterations,
            best_known=best_known,
            gap_to_best=gap
        )

    def plot_results(self):
        """Create performance visualizations using Streamlit"""
        if not self.results:
            st.warning("No benchmark results available")
            return

        # Convert results to DataFrame for easier plotting
        df = pd.DataFrame([vars(r) for r in self.results])

        st.subheader("Benchmark Results")

        # Size vs Runtime plot
        st.write("Problem Size vs Runtime")
        size_runtime_chart = st.line_chart(
            df.set_index('n_points')['runtime']
        )

        # Size vs Solution Quality
        if any(df['gap_to_best'].notna()):
            st.write("Solution Quality Gap vs Problem Size")
            quality_chart = st.line_chart(
                df.set_index('n_points')['gap_to_best']
            )

        # Summary statistics
        st.write("Performance Summary")
        summary_df = df.describe()
        st.dataframe(summary_df)

        # Detailed results table
        st.write("Detailed Results")
        st.dataframe(df)
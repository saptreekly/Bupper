import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import random

@dataclass
class TimeWindow:
    """Time window constraint for a node"""
    earliest: float
    latest: float
    service_time: float = 0.0  # Time spent at node

    def is_feasible(self, arrival_time: float) -> bool:
        """Check if arrival time is within window"""
        return self.earliest <= arrival_time <= self.latest

    def get_departure_time(self, arrival_time: float) -> float:
        """Calculate departure time considering waiting and service"""
        start_time = max(self.earliest, arrival_time)
        return start_time + self.service_time

def generate_random_points(n_points: int, 
                         min_coord: float = 0.0,
                         max_coord: float = 100.0) -> np.ndarray:
    """
    Generate random points within specified bounds

    Args:
        n_points: Number of points to generate
        min_coord: Minimum coordinate value
        max_coord: Maximum coordinate value

    Returns:
        np.ndarray: Array of random points
    """
    return np.random.uniform(min_coord, max_coord, size=(n_points, 2))

def generate_random_time_windows(n_points: int,
                               horizon: float = 100.0,
                               min_window: float = 10.0,
                               max_window: float = 30.0,
                               min_service: float = 1.0,
                               max_service: float = 5.0) -> Dict[int, TimeWindow]:
    """
    Generate random time windows for points

    Args:
        n_points: Number of points
        horizon: Time horizon for planning
        min_window: Minimum time window width
        max_window: Maximum time window width
        min_service: Minimum service time
        max_service: Maximum service time

    Returns:
        Dict[int, TimeWindow]: Time windows for each point
    """
    time_windows = {}

    # Skip depot (index 0)
    for i in range(1, n_points):
        # Generate random window start
        earliest = random.uniform(0, horizon - max_window)
        # Generate random window width
        window_width = random.uniform(min_window, max_window)
        latest = min(earliest + window_width, horizon)
        # Generate random service time
        service_time = random.uniform(min_service, max_service)

        time_windows[i] = TimeWindow(earliest, latest, service_time)

    return time_windows

def validate_points(points: List[Tuple[float, float]]) -> np.ndarray:
    """
    Validate and convert input points to numpy array

    Args:
        points: List of (x, y) coordinates

    Returns:
        np.ndarray: Validated points array
    """
    try:
        points_array = np.array(points)
        if points_array.shape[1] != 2:
            raise ValueError("Each point must have exactly 2 coordinates (x, y)")
        return points_array
    except Exception as e:
        raise ValueError(f"Invalid input format: {str(e)}")

def parse_input_string(input_str: str) -> List[Tuple[float, float]]:
    """
    Parse input string to list of coordinates

    Format: "x1,y1;x2,y2;..."
    """
    try:
        points = []
        pairs = input_str.strip().split(';')
        for pair in pairs:
            if not pair:
                continue
            x, y = map(float, pair.split(','))
            points.append((x, y))
        return points
    except Exception as e:
        raise ValueError(f"Invalid input format. Expected 'x1,y1;x2,y2;...': {str(e)}")

def parse_time_windows(input_str: str) -> Dict[int, TimeWindow]:
    """
    Parse time window input string

    Format: "node_idx:earliest,latest,service_time;..."
    Example: "0:0,10,2;1:5,15,3"
    """
    time_windows = {}
    try:
        if not input_str.strip():
            return time_windows

        entries = input_str.strip().split(';')
        for entry in entries:
            if not entry:
                continue
            node_str, times_str = entry.split(':')
            node_idx = int(node_str)
            earliest, latest, service = map(float, times_str.split(','))
            time_windows[node_idx] = TimeWindow(earliest, latest, service)
        return time_windows
    except Exception as e:
        raise ValueError(
            f"Invalid time window format. Expected 'node:earliest,latest,service;...': {str(e)}"
        )
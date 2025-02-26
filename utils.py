import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

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
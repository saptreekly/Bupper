import numpy as np
from typing import List, Tuple

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

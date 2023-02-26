import numpy as np

def compute_direction_between_points(point1, point2):
    """
    Find direction in radians between two points.
    """
    point1 = np.array(point1)
    point2 = np.array(point2)
    x1, y1 = point1
    x2, y2 = point2
    dx = x2 - x1
    dy = y2 - y1
    return np.arctan2(dy, dx)

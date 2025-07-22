from shapely.geometry import Point, Polygon
import numpy as np

def calculate_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return point1.distance(point2)

def is_point_within_polygon(point, polygon):
    """Check if a point is within a given polygon."""
    return polygon.contains(point)

def generate_random_point_within_polygon(polygon):
    """Generate a random point within a given polygon."""
    min_x, min_y, max_x, max_y = polygon.bounds
    while True:
        random_point = Point(np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y))
        if polygon.contains(random_point):
            return random_point

def project_point_to_utm(point, transformer):
    """Project a point to UTM coordinates using the provided transformer."""
    return transformer.transform(point.x, point.y)

def project_polygon_to_utm(polygon, transformer):
    """Project a polygon to UTM coordinates using the provided transformer."""
    return Polygon([project_point_to_utm(Point(xy), transformer) for xy in polygon.exterior.coords])
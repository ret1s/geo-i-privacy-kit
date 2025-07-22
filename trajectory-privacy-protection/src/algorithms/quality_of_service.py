from shapely.geometry import Point, Polygon

class QualityOfService:
    def __init__(self, qos_radius):
        self.qos_radius = qos_radius

    def is_within_qos(self, point, reference_point):
        """
        Check if a given point is within the specified QoS radius of a reference point.
        
        :param point: The point to check (as a Shapely Point).
        :param reference_point: The reference point (as a Shapely Point).
        :return: True if the point is within the QoS radius, False otherwise.
        """
        return point.distance(reference_point) <= self.qos_radius

    def generate_qos_polygon(self, center_point):
        """
        Generate a QoS polygon around a center point based on the QoS radius.
        
        :param center_point: The center point (as a Shapely Point).
        :return: A Shapely Polygon representing the QoS area.
        """
        return center_point.buffer(self.qos_radius)  # Creates a circular buffer around the point

    def validate_fake_point(self, fake_point, reference_point):
        """
        Validate if a generated fake point maintains the specified QoS radius from the reference point.
        
        :param fake_point: The generated fake point (as a Shapely Point).
        :param reference_point: The reference point (as a Shapely Point).
        :return: True if the fake point is valid, False otherwise.
        """
        return self.is_within_qos(fake_point, reference_point)
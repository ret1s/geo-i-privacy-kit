from shapely.geometry import Point, Polygon
import numpy as np

class GeoIndistinguishability:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def perturb_location(self, location):
        noise = self._generate_noise()
        perturbed_location = Point(location.x + noise[0], location.y + noise[1])
        return perturbed_location

    def _generate_noise(self):
        scale = 1 / (self.epsilon + 1e-9)
        dx = np.random.laplace(0, scale)
        dy = np.random.laplace(0, scale)
        return dx, dy

    def is_within_bounds(self, point, bounds):
        polygon = Polygon(bounds)
        return polygon.contains(point)

    def generate_fake_trajectory(self, start_point, end_point, num_points):
        trajectory = []
        for i in range(num_points):
            fraction = i / (num_points - 1)
            interpolated_point = Point(
                start_point.x + fraction * (end_point.x - start_point.x),
                start_point.y + fraction * (end_point.y - start_point.y)
            )
            perturbed_point = self.perturb_location(interpolated_point)
            trajectory.append(perturbed_point)
        return trajectory
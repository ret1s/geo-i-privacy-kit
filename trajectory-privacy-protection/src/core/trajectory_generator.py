from shapely.geometry import Point, LineString, Polygon
import random
import numpy as np

class TrajectoryGenerator:
    def __init__(self, privacy_epsilon, qos_polygon, start_polygon=None, end_polygon=None):
        self.privacy_epsilon = privacy_epsilon
        self.qos_polygon = qos_polygon
        self.start_polygon = start_polygon
        self.end_polygon = end_polygon

    def generate_fake_trajectory(self, true_trajectory, num_fake_points):
        fake_trajectory = []
        for i in range(num_fake_points):
            true_point = true_trajectory[i % len(true_trajectory)]
            fake_point = self._perturb_point(true_point)
            fake_trajectory.append(fake_point)
        return fake_trajectory

    def _perturb_point(self, point):
        dx, dy = self._planar_laplace_noise()
        perturbed_point = Point(point.x + dx, point.y + dy)
        if self.qos_polygon.contains(perturbed_point):
            return perturbed_point
        else:
            return self._snap_to_qos_boundary(perturbed_point)

    def _planar_laplace_noise(self):
        scale = 100 / (self.privacy_epsilon + 1e-9)
        u1 = random.uniform(-0.5, 0.5)
        u2 = random.uniform(-0.5, 0.5)
        dx = scale * np.sign(u1) * np.log(1 - 2 * abs(u1))
        dy = scale * np.sign(u2) * np.log(1 - 2 * abs(u2))
        return dx, dy

    def _snap_to_qos_boundary(self, point):
        if self.qos_polygon.boundary.contains(point):
            return point
        else:
            return self.qos_polygon.boundary.interpolate(self.qos_polygon.boundary.project(point))
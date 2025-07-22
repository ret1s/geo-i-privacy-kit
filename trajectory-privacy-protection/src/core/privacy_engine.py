# filepath: /trajectory-privacy-protection/trajectory-privacy-protection/src/core/privacy_engine.py

import random
from shapely.geometry import Point
from .geo_indistinguishability import GeoIndistinguishability
from .trajectory_generator import TrajectoryGenerator
from ..algorithms.quality_of_service import QualityOfService
from ..algorithms.point_protection import PointProtection

class PrivacyEngine:
    def __init__(self, epsilon, qos_radius, start_polygon, end_polygon):
        self.epsilon = epsilon
        self.qos_radius = qos_radius
        self.start_polygon = start_polygon
        self.end_polygon = end_polygon
        self.geo_indistinguishability = GeoIndistinguishability(epsilon)
        self.trajectory_generator = TrajectoryGenerator()
        self.qos_checker = QualityOfService(qos_radius)
        self.point_protection = PointProtection(start_polygon, end_polygon)

    def generate_fake_trajectory(self, true_trajectory):
        fake_trajectory = []
        for point in true_trajectory:
            protected_point = self.geo_indistinguishability.apply(point)
            if self.qos_checker.is_within_qos(protected_point):
                fake_point = self.trajectory_generator.generate_fake_point(protected_point)
                fake_trajectory.append(fake_point)
        return fake_trajectory

    def protect_start_end_points(self, start_point, end_point):
        protected_start = self.point_protection.protect_start_point(start_point)
        protected_end = self.point_protection.protect_end_point(end_point)
        return protected_start, protected_end

    def continuous_reporting(self, true_trajectory):
        for point in true_trajectory:
            fake_trajectory = self.generate_fake_trajectory(true_trajectory)
            yield fake_trajectory
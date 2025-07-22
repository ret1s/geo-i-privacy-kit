from shapely.geometry import Point
import random

class PointProtection:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def obfuscate_point(self, original_point):
        noise_x = random.uniform(-self.epsilon, self.epsilon)
        noise_y = random.uniform(-self.epsilon, self.epsilon)
        obfuscated_point = Point(original_point.x + noise_x, original_point.y + noise_y)
        return obfuscated_point

    def obfuscate_start_end_points(self, start_point, end_point):
        obfuscated_start = self.obfuscate_point(start_point)
        obfuscated_end = self.obfuscate_point(end_point)
        return obfuscated_start, obfuscated_end
class RealismChecker:
    def __init__(self, road_network, water_bodies):
        self.road_network = road_network
        self.water_bodies = water_bodies

    def is_realistic(self, point):
        if not self.is_on_road(point):
            return False
        if self.is_near_water(point):
            return False
        return True

    def is_on_road(self, point):
        # Check if the point is on the road network
        return self.road_network.contains(point)

    def is_near_water(self, point, threshold=0.001):
        # Check if the point is near any water bodies
        return any(water.distance(point) < threshold for water in self.water_bodies)
import unittest
from src.core.trajectory_generator import TrajectoryGenerator

class TestTrajectoryGenerator(unittest.TestCase):

    def setUp(self):
        self.generator = TrajectoryGenerator()

    def test_generate_fake_trajectory(self):
        true_location = (106.7009, 10.7769)  # Example coordinates for Ho Chi Minh City
        qos_polygon = None  # Replace with actual QoS polygon if needed
        fake_trajectory = self.generator.generate_fake_trajectory(true_location, qos_polygon)

        self.assertIsNotNone(fake_trajectory)
        self.assertIsInstance(fake_trajectory, list)
        self.assertGreater(len(fake_trajectory), 0)

    def test_trajectory_within_bounds(self):
        true_location = (106.7009, 10.7769)
        qos_polygon = None  # Replace with actual QoS polygon if needed
        fake_trajectory = self.generator.generate_fake_trajectory(true_location, qos_polygon)

        for point in fake_trajectory:
            self.assertTrue(self.generator.is_within_bounds(point))

    def test_start_end_protection(self):
        true_location = (106.7009, 10.7769)
        qos_polygon = None  # Replace with actual QoS polygon if needed
        fake_trajectory = self.generator.generate_fake_trajectory(true_location, qos_polygon)

        self.assertNotEqual(fake_trajectory[0], true_location)
        self.assertNotEqual(fake_trajectory[-1], true_location)

if __name__ == '__main__':
    unittest.main()
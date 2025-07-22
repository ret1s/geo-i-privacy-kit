from src.core.privacy_engine import PrivacyEngine
import unittest

class TestPrivacyEngine(unittest.TestCase):

    def setUp(self):
        self.privacy_engine = PrivacyEngine()

    def test_generate_fake_trajectory(self):
        true_trajectory = [(106.7009, 10.7769), (106.7010, 10.7770)]
        fake_trajectory = self.privacy_engine.generate_fake_trajectory(true_trajectory, privacy_epsilon=0.5)
        self.assertIsInstance(fake_trajectory, list)
        self.assertNotEqual(fake_trajectory, true_trajectory)

    def test_geo_indistinguishability(self):
        point_a = (106.7009, 10.7769)
        point_b = (106.7010, 10.7770)
        self.assertTrue(self.privacy_engine.check_geo_indistinguishability(point_a, point_b, epsilon=0.5))

    def test_start_end_point_protection(self):
        start_point = (106.7009, 10.7769)
        end_point = (106.7010, 10.7770)
        protected_start, protected_end = self.privacy_engine.protect_start_end_points(start_point, end_point)
        self.assertNotEqual(protected_start, start_point)
        self.assertNotEqual(protected_end, end_point)

    def test_quality_of_service(self):
        fake_point = (106.7010, 10.7770)
        self.assertTrue(self.privacy_engine.check_quality_of_service(fake_point, qos_radius=0.01))

if __name__ == '__main__':
    unittest.main()
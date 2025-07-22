from src.algorithms.realism_checker import RealismChecker
import unittest

class TestRealismChecker(unittest.TestCase):
    
    def setUp(self):
        self.realism_checker = RealismChecker()

    def test_check_realistic_point(self):
        # Test with a realistic point
        point = (106.7009, 10.7769)  # Example coordinates
        result = self.realism_checker.check_realistic_point(point)
        self.assertTrue(result, "The point should be considered realistic.")

    def test_check_unrealistic_point(self):
        # Test with an unrealistic point
        point = (200.0, 100.0)  # Out of bounds coordinates
        result = self.realism_checker.check_realistic_point(point)
        self.assertFalse(result, "The point should not be considered realistic.")

    def test_check_edge_case(self):
        # Test with an edge case point
        point = (106.7009, 10.7769)  # Example coordinates on the edge
        result = self.realism_checker.check_realistic_point(point)
        self.assertTrue(result, "The edge case point should be considered realistic.")

if __name__ == '__main__':
    unittest.main()
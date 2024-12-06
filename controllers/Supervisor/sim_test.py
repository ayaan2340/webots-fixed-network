import unittest
import numpy as np
from simulation import Simulation, Road, SimulationObject


class TestSimulation(unittest.TestCase):
    def setUp(self):
        """Set up a fresh simulation instance before each test."""
        self.sim = Simulation()

    def test_initialization(self):
        """Test that simulation is initialized with correct default values."""
        self.assertIsNotNone(self.sim.car)
        self.assertIsNotNone(self.sim.start)
        self.assertIsNotNone(self.sim.end)
        self.assertEqual(self.sim.time, 0.0)
        self.assertEqual(self.sim.speed, 0.0)
        self.assertEqual(self.sim.steering_angle, 0.0)

        # Check that roads were initialized
        self.assertEqual(len(self.sim.vertical_roads), 6)
        self.assertEqual(len(self.sim.horizontal_roads), 6)
        self.assertEqual(len(self.sim.intersections), 4)
        self.assertEqual(len(self.sim.all_roads), 16)  # 6 + 6 + 4

    def test_normalize_angle(self):
        """Test angle normalization function."""
        test_cases = [
            (0, 0),                 # Zero angle
            (2 * np.pi, 0),        # Full rotation
            (np.pi / 2, np.pi / 2),    # Quarter rotation
            (-np.pi / 2, -np.pi / 2),  # Negative quarter rotation
        ]

        for input_angle, expected in test_cases:
            with self.subTest(input_angle=input_angle):
                result = self.sim.normalize_angle(input_angle)
                self.assertAlmostEqual(result, expected, places=6)

        # Test that bigger angles get normalized into [-π, π] range
        for input_angle in [3 * np.pi, -3 * np.pi, 5 * np.pi, -5 * np.pi]:
            with self.subTest(input_angle=input_angle):
                result = self.sim.normalize_angle(input_angle)
                self.assertTrue(-np.pi <= result <= np.pi)

    def test_is_on_road(self):
        """Test road collision detection."""
        def is_on_vertical_road(pos, road):
            x, y = pos
            rx, ry = road.position
            rw, rh = road.size
            return (rx - rw / 2 <= x <= rx + rw / 2 and
                    ry - rh / 2 <= y <= ry + rh / 2)

        # Test vertical road
        vertical_road = self.sim.vertical_roads[0]
        pos_x, pos_y = vertical_road.position
        width, height = vertical_road.size

        # Test points on vertical road
        self.assertTrue(is_on_vertical_road((pos_x, pos_y), vertical_road))  # Center
        self.assertTrue(
            is_on_vertical_road(
                (pos_x + width / 2 - 0.1,
                 pos_y),
                vertical_road))  # Right edge
        self.assertTrue(
            is_on_vertical_road(
                (pos_x - width / 2 + 0.1,
                 pos_y),
                vertical_road))  # Left edge
        self.assertFalse(
            is_on_vertical_road(
                (pos_x + width / 2 + 0.1,
                 pos_y),
                vertical_road))  # Just off right

    def test_car_movement(self):
        """Test basic car movement and physics."""
        # Place car at origin
        self.sim.car.position = (0, 0)
        self.sim.car.rotation = 0  # Facing right

        # Set forward movement
        self.sim.set_controls(np.array([0.5, 0.0]))  # Half speed, no steering

        # Step simulation
        self.sim.step()

        # Car should have moved right (positive x)
        self.assertGreater(self.sim.car.position[0], 0)
        self.assertAlmostEqual(self.sim.car.position[1], 0)  # Should not have moved vertically

        # Test turning
        self.sim.car.position = (0, 0)
        self.sim.car.rotation = 0
        self.sim.set_controls(np.array([0.5, 0.5]))  # Half speed, full right turn

        initial_rotation = self.sim.car.rotation
        self.sim.step()

        # Car should have turned
        self.assertNotEqual(self.sim.car.rotation, initial_rotation)

    def test_reset(self):
        """Test simulation reset functionality."""
        # Modify simulation state
        self.sim.car.position = (50, 50)
        self.sim.car.rotation = 2.0
        self.sim.speed = 20.0
        self.sim.steering_angle = 0.3
        self.sim.time = 10.0

        # Reset simulation
        self.sim.reset()

        # Check that values were reset
        self.assertEqual(self.sim.time, 0.0)
        self.assertEqual(self.sim.speed, 0.0)
        self.assertEqual(self.sim.steering_angle, 0.0)
        self.assertEqual(self.sim.car.position, self.sim.start.position)

        # After reset, car should be on a road
        self.assertTrue(self.sim.is_on_road(self.sim.car.position))

    def test_goal_detection(self):
        """Test goal reaching detection."""
        # Place car right at goal
        self.sim.car.position = self.sim.end.position
        self.assertTrue(self.sim.reached_goal())

        # Place car just outside goal radius
        goal_x, goal_y = self.sim.end.position
        self.sim.car.position = (goal_x + 2.1, goal_y)  # Just beyond 2.0 unit tolerance
        self.assertFalse(self.sim.reached_goal())

        # Place car just inside goal radius
        self.sim.car.position = (goal_x + 1.9, goal_y)  # Just within 2.0 unit tolerance
        self.assertTrue(self.sim.reached_goal())

    def test_get_inputs(self):
        """Test neural network input generation."""
        inputs = self.sim.get_inputs()

        # Should return 5 normalized inputs
        self.assertEqual(len(inputs), 5)

        # All inputs should be normalized
        for input_value in inputs:
            self.assertTrue(-1 <= input_value <= 1)

    def test_randomize_goals(self):
        """Test goal randomization."""
        initial_start = self.sim.start.position
        initial_end = self.sim.end.position

        self.sim.randomize_goals()

        # Start and end should be different from initial positions
        self.assertNotEqual(self.sim.start.position, initial_start)
        self.assertNotEqual(self.sim.end.position, initial_end)

        # Start and end should be different from each other
        self.assertNotEqual(self.sim.start.position, self.sim.end.position)

        # Both positions should be on roads
        self.assertTrue(self.sim.is_on_road(self.sim.start.position))
        self.assertTrue(self.sim.is_on_road(self.sim.end.position))

    def test_get_angle_to_goal(self):
        """Test angle calculation to goal."""
        # Place car and goal in known positions
        self.sim.car.position = (0, 0)
        self.sim.end.position = (1, 0)  # Goal directly to right
        self.sim.car.rotation = 0  # Facing right

        # Angle should be 0 when facing directly at goal
        self.assertAlmostEqual(self.sim.get_angle_to_goal(), 0)

        # Rotate car 90 degrees right
        self.sim.car.rotation = np.pi / 2
        self.assertAlmostEqual(self.sim.get_angle_to_goal(), -np.pi / 2)

        # Rotate car 90 degrees left
        self.sim.car.rotation = -np.pi / 2
        self.assertAlmostEqual(self.sim.get_angle_to_goal(), np.pi / 2)


if __name__ == '__main__':
    unittest.main()

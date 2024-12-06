# simulation.py
from dataclasses import dataclass
from typing import List, Tuple
import random
import numpy as np
import torch


@dataclass
class Road:
    position: Tuple[float, float]  # x, y position
    size: Tuple[float, float]      # width, height
    is_vertical: bool


@dataclass
class SimulationObject:
    position: Tuple[float, float]  # x, y position
    rotation: float = 0.0          # rotation in radians


class Simulation:
    def __init__(self):
        # Initialize roads
        self.vertical_roads = [
            Road((0, 0), (7, 15), True),
            Road((30, 0), (7, 15), True),
            Road((60, 0), (7, 15), True),
            Road((90, 0), (7, 15), True),
            Road((120, 0), (7, 15), True),
            Road((150, 0), (7, 15), True),
        ]

        self.horizontal_roads = [
            Road((0, 0), (15, 7), False),
            Road((0, 30), (15, 7), False),
            Road((0, 60), (15, 7), False),
            Road((0, 90), (15, 7), False),
            Road((0, 120), (15, 7), False),
            Road((0, 150), (15, 7), False),
        ]

        self.intersections = [
            Road((30, 30), (19, 19), False),
            Road((60, 60), (19, 19), False),
            Road((90, 90), (19, 19), False),
            Road((120, 120), (19, 19), False),
        ]

        self.all_roads = self.vertical_roads + self.horizontal_roads + self.intersections

        # Initialize simulation objects
        self.car = SimulationObject((0, 0), 0.0)
        self.start = SimulationObject((0, 0))
        self.end = SimulationObject((100, 100))

        # Simulation parameters
        self.time = 0.0
        self.time_step = 0.01  # 10ms
        self.max_speed = 30.0
        self.max_steering_angle = 0.5

        # Current state
        self.speed = 0.0
        self.steering_angle = 0.0
        self.distance_travelled = 0.0
        self.previous_distance = 0.0

    def reset(self):
        """Reset the simulation state."""
        self.randomize_goals()
        self.time = 0.0
        self.speed = 0.0
        self.steering_angle = 0.0
        self.distance_travelled = 0.0

        # Place car at start position
        self.car.position = self.start.position
        self.car.rotation = random.uniform(-np.pi, np.pi)

        self.previous_distance = self.calculate_distance_to_goal()

    def get_valid_road_positions(self) -> List[Tuple[float, float]]:
        """Get list of valid positions on all roads."""
        positions = []
        for road in self.all_roads:
            # For vertical roads
            if road.is_vertical:
                y_positions = np.linspace(road.position[1] - road.size[1] / 2,
                                          road.position[1] + road.size[1] / 2,
                                          5)
                for y in y_positions:
                    positions.append((road.position[0], y))
            # For horizontal roads and intersections
            else:
                x_positions = np.linspace(road.position[0] - road.size[0] / 2,
                                          road.position[0] + road.size[0] / 2,
                                          5)
                for x in x_positions:
                    positions.append((x, road.position[1]))
        return positions

    def randomize_goals(self):
        """Randomize start and end positions on different roads."""
        valid_positions = self.get_valid_road_positions()

        # Ensure we have enough positions
        if len(valid_positions) < 2:
            raise ValueError("Not enough valid road positions")

        # Select random start position
        start_pos = random.choice(valid_positions)
        # Remove start position from available positions
        valid_positions.remove(start_pos)
        # Select random end position from remaining positions
        end_pos = random.choice(valid_positions)

        self.start.position = start_pos
        self.end.position = end_pos

    def is_on_road(self, position: Tuple[float, float]) -> bool:
        """Check if a position is on any road."""
        x, y = position
        epsilon = 1e-10  # Small value for floating-point comparison

        for road in self.all_roads:
            rx, ry = road.position
            rw, rh = road.size
            half_width = rw / 2
            half_height = rh / 2

            # Check if point is within road boundaries with epsilon tolerance
            x_min = rx - half_width - epsilon
            x_max = rx + half_width + epsilon
            y_min = ry - half_height - epsilon
            y_max = ry + half_height + epsilon

            if x_min <= x <= x_max and y_min <= y <= y_max:
                return True
        return False

    def calculate_distance_to_goal(self) -> float:
        """Calculate distance from car to goal."""
        return np.sqrt(
            (self.car.position[0] - self.end.position[0])**2 +
            (self.car.position[1] - self.end.position[1])**2
        )

    def get_angle_to_goal(self) -> float:
        """Calculate angle between car's heading and goal."""
        dx = self.end.position[0] - self.car.position[0]
        dy = self.end.position[1] - self.car.position[1]
        goal_angle = np.arctan2(dy, dx)
        return self.normalize_angle(goal_angle - self.car.rotation)

    def normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        angle = angle % (2 * np.pi)  # First, normalize to [0, 2π]
        if angle > np.pi:
            angle -= 2 * np.pi  # Convert to [-π, π]
        return angle

    def get_inputs(self) -> np.ndarray:
        """Get normalized inputs for the neural network."""
        return np.array([
            self.get_angle_to_goal() / np.pi,
            (self.car.position[0] - self.end.position[0]) / 150,
            (self.car.position[1] - self.end.position[1]) / 150,
            self.speed / self.max_speed,
            self.steering_angle / self.max_steering_angle
        ])

    def set_controls(self, outputs):
        """Set car controls from neural network outputs."""
        # Ensure outputs are regular floats/numpy values
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.detach().numpy()

        self.speed = float(outputs[0] * self.max_speed)
        self.steering_angle = float(outputs[1] * self.max_steering_angle)

    def step(self):
        """Advance simulation by one time step. Returns False if simulation should end."""
        # Convert all values to regular floats for calculations
        speed = float(self.speed)
        angle = float(self.steering_angle)
        rotation = float(self.car.rotation)

        # Update car position based on physics
        dx = speed * np.cos(rotation) * self.time_step
        dy = speed * np.sin(rotation) * self.time_step
        new_position = (
            self.car.position[0] + dx,
            self.car.position[1] + dy
        )

        # Only update if new position is valid
        if self.is_on_road(new_position):
            self.car.position = new_position
            self.car.rotation = self.normalize_angle(
                rotation + speed * np.tan(angle) * self.time_step)
            self.distance_travelled += np.sqrt(dx**2 + dy**2)
        else:
            return False  # End simulation if car goes off road

        self.time += self.time_step
        return True

    def reached_goal(self) -> bool:
        """Check if car has reached the goal."""
        distance = self.calculate_distance_to_goal()
        return distance < 2.0  # 2 units tolerance

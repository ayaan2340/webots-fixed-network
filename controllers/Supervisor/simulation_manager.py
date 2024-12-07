# simulation_manager.py
import numpy as np
from simulation import Simulation
from RecurrentNetwork import RecurrentNetwork
from pathlib import Path
import os


class SimulationManager:
    def __init__(self, process_id=0):
        self.simulation = None
        self.max_simulation_time = 1000  # Maximum frames for simulation
        self.process_id = process_id
        self.num_trials = 3
        self.reset_simulation_state()

    def reset_simulation_state(self):
        """Reset the simulation to initial state."""
        self.simulation = Simulation()
        self.simulation.reset()

    def calculate_distance_from_start(self):
        """Calculate how far the car has traveled from start position."""
        dx = self.simulation.car.position[0] - self.simulation.start.position[0]
        dy = self.simulation.car.position[1] - self.simulation.start.position[1]
        return np.sqrt(dx * dx + dy * dy)

    def calculate_distance_reward(self) -> float:
        """Calculate reward based on distance to goal."""
        current_distance = self.simulation.calculate_distance_to_goal()
        reward = self.simulation.previous_distance - current_distance
        self.simulation.previous_distance = current_distance
        return reward

    def evaluate_genome(self, genome):
        """Evaluate genome using shortest path distance to goal along roads"""
        self.simulation.reset()
        genome.reset()

        total_fitness = 0
        episode_steps = 0
        max_steps = 1000

        # Helper function to estimate path distance along roads
        def estimate_path_distance(pos1, pos2):
            """Estimate shortest path distance between two points along roads"""
            # Find nearest roads to start and end points
            def find_nearest_roads(x, y):
                v_dists = [(abs(road.position[0] - x), road.position[0])
                           for road in self.simulation.vertical_roads]
                h_dists = [(abs(road.position[1] - y), road.position[1])
                           for road in self.simulation.horizontal_roads]
                nearest_v = min(v_dists, key=lambda x: x[0])[1]
                nearest_h = min(h_dists, key=lambda x: x[0])[1]
                return nearest_v, nearest_h

            x1, y1 = pos1
            x2, y2 = pos2
            v1, h1 = find_nearest_roads(x1, y1)
            v2, h2 = find_nearest_roads(x2, y2)

            # Calculate path segments
            # 1. Distance to nearest intersection
            d1 = min(abs(x1 - v1) + abs(y1 - h1),  # Distance to nearest intersection
                     abs(x1 - v2) + abs(y1 - h1),   # Distance to goal's vertical road
                     abs(x1 - v1) + abs(y1 - h2))   # Distance to goal's horizontal road

            # 2. Distance along roads to goal
            d2 = abs(v2 - v1) + abs(h2 - h1)  # Manhattan distance between intersections

            # 3. Distance from last intersection to goal
            d3 = abs(x2 - v2) + abs(y2 - h2)

            return d1 + d2 + d3

        # Initial path distance to goal
        initial_path_dist = estimate_path_distance(
            self.simulation.car.position,
            self.simulation.end.position
        )
        last_path_dist = initial_path_dist

        while episode_steps < max_steps:
            # Get current state and optimal angle
            state = self.simulation.get_inputs()
            optimal_angle = self.simulation.get_optimal_orientation()

            # Get action from network
            speed, angle = genome.select_action(state, optimal_angle)

            # Convert angle to steering
            current_angle = self.simulation.car.rotation
            angle_diff = self.simulation.normalize_angle(angle - current_angle)
            steering = np.clip(angle_diff / self.simulation.max_steering_angle, -1, 1)

            # Set controls and step simulation
            self.simulation.set_controls([speed, steering])
            if not self.simulation.step():  # Car went off road
                break

            # Calculate current path distance to goal
            current_path_dist = estimate_path_distance(
                self.simulation.car.position,
                self.simulation.end.position
            )

            # Calculate reward based on path distance improvement
            path_improvement = last_path_dist - current_path_dist
            total_fitness += path_improvement
            last_path_dist = current_path_dist

            # Check if goal reached
            if self.simulation.reached_goal():
                total_fitness += 1000  # Bonus for reaching goal
                break

            episode_steps += 1

        # Train network using collected experience
        genome.train_episode()

        # Return normalized fitness score
        return total_fitness / initial_path_dist if initial_path_dist > 0 else 0

    def visualize_network(self, genome: RecurrentNetwork, save_path: str = "visualization.html"):
        # Load the HTML template
        with open('simulation_template.html', 'r') as f:
            html_template = f.read()

        # Reset states
        self.reset_simulation_state()
        genome.hidden_state = None
        frames = []
        print("Starting simulation...")  # Debug print

        while self.simulation.time < self.max_simulation_time:
            # Record current frame
            frame_commands = [
                "simulationRenderer.clearCanvas();",
            ]

            # Draw all roads
            for road in self.simulation.all_roads:
                frame_commands.append(
                    f"simulationRenderer.drawRoad({{"
                    f"position: [{road.position[0]}, {road.position[1]}], "
                    f"size: [{road.size[0]}, {road.size[1]}]"
                    f"}});"
                )

            # Draw start and end points
            frame_commands.append(
                f"simulationRenderer.drawStartEnd("
                f"[{self.simulation.start.position[0]}, {self.simulation.start.position[1]}], "
                f"[{self.simulation.end.position[0]}, {self.simulation.end.position[1]}]);"
            )

            # Draw car
            frame_commands.append(
                f"simulationRenderer.drawCar("
                f"[{self.simulation.car.position[0]}, {self.simulation.car.position[1]}], "
                f"{self.simulation.car.rotation});"
            )

            frames.append("\n".join(frame_commands))

            # Run simulation step
            inputs = self.simulation.get_inputs()
            outputs = genome.forward(inputs)
            self.simulation.set_controls(outputs)
            if not self.simulation.step():  # Car went off road
                print("Car went off road")  # Debug print
                break

            if self.simulation.reached_goal():
                print("Reached goal!")  # Debug print
                break

        print(f"Generated {len(frames)} frames")  # Debug print

        # Generate visualization HTML with actual frame data
        js_frames = "[" + ",".join([f"`{frame}`" for frame in frames]) + "]"
        viz_html = html_template.replace(
            'const frames = [];',
            f'const frames = {js_frames};'
        )

        # Save to file
        with open(save_path, 'w') as f:
            f.write(viz_html)
        print(f"Saved visualization to {save_path}")  # Debug print

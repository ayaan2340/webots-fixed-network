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
        self.simulation.create_checkpoints()
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
        """Modified to include REINFORCE training during episode"""
        self.simulation.reset()
        genome.reset()

        total_distance = 0
        episode_steps = 0
        max_steps = 1000

        while episode_steps < max_steps:
            # Get current state and optimal angle
            state = self.simulation.get_inputs()
            optimal_angle = self.simulation.get_optimal_orientation()

            # Get action from network
            angle = genome.select_action(state, optimal_angle)

            # Convert angle to controls (speed and steering)
            speed = self.simulation.max_speed * 0.5  # Constant speed for simplicity
            current_angle = self.simulation.car.rotation
            angle_diff = self.simulation.normalize_angle(angle - current_angle)
            steering = np.clip(angle_diff / self.simulation.max_steering_angle, -1, 1)

            # Set controls and step simulation
            self.simulation.set_controls([speed, steering])
            if not self.simulation.step():
                break

            # Check if goal reached
            if self.simulation.reached_goal():
                break

            episode_steps += 1

        # Train network using collected experience
        genome.train_episode()

        # Final fitness based on distance to goal
        final_distance = self.simulation.calculate_distance_to_goal()
        return -final_distance  # Negative because we want to minimize distance

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

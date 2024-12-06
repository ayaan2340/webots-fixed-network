# simulation_manager.py
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from simulation import Simulation
from RecurrentNetwork import RecurrentNetwork


@dataclass
class SyllabusFrame:
    inputs: np.ndarray
    outputs: np.ndarray
    position: Tuple[float, float]
    rotation: float
    success: bool
    fitness: float  # Store the fitness of the run this frame came from


class SimulationManager:
    def __init__(self, process_id=0, syllabus_manager=None):
        self.simulation = None
        self.max_simulation_time = 1000  # Maximum frames for simulation
        self.process_id = process_id
        self.num_trials = 1
        self.syllabus_manager = syllabus_manager
        self.frame_sample_rate = 10
        self.reset_simulation_state()

    def record_frame(self, genome, inputs, outputs, buffer) -> None:
        """Record current frame if conditions are met."""
        # Only record every Nth frame to reduce memory usage
        if int(self.simulation.time / self.simulation.time_step) % self.frame_sample_rate != 0:
            return

        frame = SyllabusFrame(
            inputs=inputs,
            outputs=outputs,
            position=self.simulation.car.position,
            rotation=self.simulation.car.rotation,
            success=False,  # Will be updated later if run is successful
            fitness=0.0     # Will be updated later with final fitness
        )
        buffer.add_frame(genome.genome_id, frame)

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

    def evaluate_single_trial(self, genome, frame_buffer) -> float:
        """Evaluate a single trial and record frames."""
        genome.hidden_state = None
        start_time = self.simulation.time
        prev_time = 0
        time_on_road = 0
        success = False

        while self.simulation.time < self.max_simulation_time:
            if self.simulation.am_on_road():
                time_on_road += self.simulation.time - prev_time
            prev_time = self.simulation.time

            # Get inputs and run network
            inputs = self.simulation.get_inputs()
            outputs = genome.forward(inputs)

            # Record frame if buffer is provided
            if frame_buffer is not None:
                self.record_frame(genome, inputs, outputs, frame_buffer)

            self.simulation.set_controls(outputs)
            if not self.simulation.step():
                break

            if self.simulation.reached_goal():
                success = True
                break

        # Calculate fitness
        on_road_ratio = time_on_road / (self.simulation.time - start_time)
        distance_fitness = self.calculate_distance_from_start()
        trial_fitness = distance_fitness * on_road_ratio

        # Update success status and fitness for all recorded frames
        if frame_buffer is not None:
            for frame in frame_buffer.frames.get(genome.genome_id, []):
                frame.success = success
                frame.fitness = trial_fitness

        return trial_fitness

    def evaluate_genome(self, genome, frame_buffer=None) -> float:
        """Evaluate a genome over multiple trials."""
        total_fitness = 0.0

        for _ in range(self.num_trials):
            self.reset_simulation_state()
            trial_fitness = self.evaluate_single_trial(genome, frame_buffer)
            total_fitness += trial_fitness

        return total_fitness / self.num_trials

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

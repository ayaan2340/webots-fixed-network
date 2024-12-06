# simulation_manager.py
import numpy as np
from simulation import Simulation
from RecurrentNetwork import RecurrentNetwork
from pathlib import Path
import os


class SimulationManager:
    def __init__(self, port: int = 1234):
        self.port = port
        self.simulation = Simulation()
        self.fitness = 0.0
        self.max_simulation_time = 15  # seconds

    def reset_simulation_state(self):
        """Reset the simulation to initial state."""
        self.simulation.reset()
        self.fitness = 0.0

    def calculate_distance_reward(self) -> float:
        """Calculate reward based on distance to goal."""
        current_distance = self.simulation.calculate_distance_to_goal()
        reward = self.simulation.previous_distance - current_distance
        self.simulation.previous_distance = current_distance
        return reward

    def evaluate_genome(self, genome: RecurrentNetwork) -> float:
        """Evaluate a single genome's fitness."""
        genome.hidden_state = None
        self.simulation.time = 0.0
        self.fitness = 0.0

        # Calculate shortest path (manhattan distance)
        dx = abs(self.simulation.end.position[0] - self.simulation.start.position[0])
        dy = abs(self.simulation.end.position[1] - self.simulation.start.position[1])
        shortest_path = dx + dy
        self.max_simulation_time = max(12, shortest_path / 5.0)

        while self.simulation.time < self.max_simulation_time:
            if self.simulation.reached_goal():
                return 100 / (self.simulation.time + 1e-8)

            inputs = self.simulation.get_inputs()
            outputs = genome.forward(inputs)
            self.simulation.set_controls(outputs)

            if not self.simulation.step():  # Car went off road
                return -1

            self.fitness += self.calculate_distance_reward()

        return (100 - self.simulation.calculate_distance_to_goal()) / \
            (self.max_simulation_time + 1e-8)


def loadall(dirname: str) -> list:
    """Load all genome files from a directory."""
    genome_list = []
    for root, dirs, files in os.walk(dirname):
        for genome in files:
            genome_list.append(RecurrentNetwork.load(Path(root) / genome))
    return genome_list


def run_simulation():
    """Main simulation loop."""
    manager = SimulationManager()

    # Get the genome data directory path
    current_directory = os.path.dirname(__file__)
    grandpa_directory = os.path.abspath(os.path.join(current_directory, "..", ".."))
    genome_data_path = os.path.join(grandpa_directory, "genome_data", f"genome_data{manager.port}")

    # Load and evaluate all genomes
    batch = loadall(genome_data_path)
    for genome in batch:
        manager.reset_simulation_state()
        fitness = manager.evaluate_genome(genome)
        genome.fitness = fitness
        print(genome.fitness)


if __name__ == '__main__':
    run_simulation()

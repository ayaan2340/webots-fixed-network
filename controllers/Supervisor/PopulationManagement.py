import numpy as np
import multiprocessing
import os
import subprocess
import time
import pickle
import signal
from pathlib import Path

from RecurrentNetwork import RecurrentNetwork


class PopulationManager:
    def __init__(self, population_size, input_size, hidden_size, output_size, port,
                 jobs=10):
        self.population_size = population_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.base_port = port

        # Initialize population
        self.population = [
            RecurrentNetwork(input_size, hidden_size, output_size, genome_id)
            for genome_id in range(population_size)
        ]

        self.fitness_scores = [0.0] * population_size
        self.processor_count = jobs
        self.ports = [self.base_port + i for i in range(self.processor_count)]
        self.generation = 0

    @staticmethod
    def run_webots_static(port):
        current_directory = os.path.dirname(__file__)  # Directory of the current script

        # Construct the path to the Webots world file dynamically by going up one
        # directory and then into "worlds"
        world_file_path = os.path.join(
            current_directory,
            "..",  # Move up to the parent directory
            "..",
            "worlds",
            "firstWorld.wbt"
        )
        world_file_path = os.path.abspath(world_file_path)  # Ensure the path is absolute

        # Run Webots with the dynamically constructed path
        subprocess.run([
            "webots",
            f"--port={port}",
            "--mode=fast",
            "--no-rendering",
            "--batch",
            "--minimize",
            world_file_path  # Use the dynamically constructed path
        ], check=False)

    def start_webots_instances(self):
        """Start multiple Webots instances in parallel using static method"""
        processes = []
        for port in self.ports:
            # Use static method instead of instance method
            process = multiprocessing.Process(
                target=PopulationManager.run_webots_static,
                args=(port,)
            )
            processes.append(process)
            process.start()

        # Store processes without storing the whole class instance
        self._processes = processes
        time.sleep(15)

    def distribute_genomes_evenly(self, genomes):
        """
        Distribute genomes across available processors evenly
        """
        total_genomes = len(genomes)
        batch_size = total_genomes // self.processor_count
        extra_size = batch_size + 1  # Number of genomes for batches with one extra genome
        extra_batches = total_genomes % self.processor_count
        batches = []

        genome_index = 0
        # Distribute batches with the extra genome
        for _ in range(extra_batches):
            batch = genomes[genome_index:genome_index + extra_size]
            batches.append(batch)
            genome_index += extra_size

        # Distribute batches with the base genome size
        for _ in range(self.processor_count - extra_batches):
            batch = genomes[genome_index:genome_index + batch_size]
            batches.append(batch)
            genome_index += batch_size

        return batches

    def save_batch(self, batch, port):
        # Prepare genome data for simulation
        dir_path = Path(f"genome_data/genome_data{port}")
        for genome_id, genome in enumerate(batch):
            genome.save(dir_path / f"{genome_id}.h5")

    @staticmethod
    def run_simulation_static(port, ready_event, fitness_dict):
        """Static method to run simulation without class instance"""
        try:
            result = subprocess.run([
                "/Applications/Webots.app/Contents/MacOS/webots-controller",
                f"--port={port}",
                "SimulationManager.py",
                "--port", str(port)
            ], 
                capture_output=True, 
                text=True, 
                check=True,
            )
        except subprocess.CalledProcessError as e:
            # Capture both stdout and stderr
            error_output = e.stderr if e.stderr else e.stdout
            # Log the error if needed
            with open('error.log', 'w') as f:
                f.write(f"Process failed with exit code {e.returncode}\n")
                f.write(f"Error output:\n{error_output}")

            ## Raise a new exception with the detailed error info
            raise RuntimeError(f"Subprocess failed with exit code {e.returncode}. Error: {error_output}") from e
        # Parse fitness scores from output
        fitness_values = list(map(float, result.stdout.splitlines()))

        # Read genomes and update fitness dictionary
        genome_dir = Path(f"genome_data/genome_data{port}")
        for i, genome_path in enumerate(sorted(genome_dir.glob("*.h5"))):
            genome = RecurrentNetwork.load(genome_path)
            fitness_dict[genome.genome_id] = fitness_values[i]
        ready_event.set()

    def evaluate_fitness(self):
        """Parallel fitness evaluation"""
        # Save genomes to temporary files before distributing
        batches = self.distribute_genomes_evenly(self.population)
        for i, batch in enumerate(batches):
            save_dir = Path(f"genome_data/genome_data{self.ports[i]}")
            save_dir.mkdir(parents=True, exist_ok=True)
            for j, genome in enumerate(batch):
                genome.save(save_dir / f"{j}.h5")

        # Run simulations in parallel using static method
        processes = []
        ready_events = []
        manager = multiprocessing.Manager()
        fitness_dict = manager.dict()

        for i, port in enumerate(self.ports):
            ready_event = multiprocessing.Event()
            process = multiprocessing.Process(
                target=self.run_simulation_static,
                args=(port, ready_event, fitness_dict)
            )
            processes.append(process)
            ready_events.append(ready_event)
            process.start()

        # Wait for all processes
        for event in ready_events:
            event.wait()

        for process in processes:
            process.join()

        # Update fitness scores
        highest_score = 0
        best_genome = None
        for i, genome in enumerate(self.population):
            if genome.genome_id in fitness_dict:
                self.fitness_scores[i] = fitness_dict[genome.genome_id]
                if self.fitness_scores[i] > highest_score:
                    highest_score = self.fitness_scores[i]
                    best_genome = genome

        # Save best genome
        if best_genome is not None:
            save_dir = Path("bestBest")
            save_dir.mkdir(exist_ok=True)
            torch.save({
                'fitness': highest_score,
                'genome': best_genome.state_dict()
            }, save_dir / f"lebron{self.generation}.pt")

        return highest_score

    def tournament_selection(self, tournament_size=5):
        """
        Tournament selection to choose parents for crossover.
        Select the best individual from a random subset.
        """
        indices = np.random.choice(len(self.population), tournament_size, replace=False)
        tournament_fitness = [self.fitness_scores[i] for i in indices]
        winner_index = indices[np.argmax(tournament_fitness)]
        return self.population[winner_index]

    def create_next_generation(self):
        """
        Create the next generation through tournament selection, crossover, and mutation.
        """
        new_population = []

        while len(new_population) < self.population_size:
            # Select parents
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()

            # Create child through crossover
            child = parent1.crossover(parent2, len(new_population))

            # Mutate child
            child.mutate()

            new_population.append(child)

        self.population = new_population
        self.fitness_scores = [0.0] * self.population_size

    def train(self, num_generations):
        """
        Train the population over multiple generations.
        Manages Webots instances and evolutionary process.
        """
        # Start Webots instances
        self.start_webots_instances()

        try:
            for self.generation in range(num_generations):
                print(f"Generation {self.generation}")

                # Evaluate fitness in parallel
                best_fitness = self.evaluate_fitness()

                # Create next generation
                self.create_next_generation()

                print(f"Best Fitness: {best_fitness}")

        finally:
            # Terminate all Webots processes
            for process in self._processes:
                os.kill(process.pid, signal.SIGTERM)
            # Wait for processes to terminate
            for process in self._processes:
                process.join()


def main():
    population_manager = PopulationManager(
        population_size=100,
        input_size=5,
        hidden_size=10,
        output_size=2,
        port=10000
    )
    population_manager.train(num_generations=300)


if __name__ == '__main__':
    main()

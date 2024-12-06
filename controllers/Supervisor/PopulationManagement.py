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
    def __init__(self, population_size, input_size, hidden_size, output_size):
        self.population_size = population_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize population
        self.population = [
            RecurrentNetwork(input_size, hidden_size, output_size, genome_id) 
            for genome_id in range(population_size)
        ]
        
        # Initialize fitness scores
        self.fitness_scores = [0.0] * population_size
        
        # Configure parallel processing
        self.processor_count = 10
        self.ports = [10000 + i for i in range(self.processor_count)]
        self.generation = 0
        self.webots_processes = []

    def run_webots(self, port):
        """
        Run a single Webots instance on a specific port
        """
        try:
            # CHANGE FIRST LINE OF SUBPROCESS RUN TO YOUR WEBOTS-CONTROLLER FILE
            subprocess.run([
                "/Applications/Webots.app/Contents/MacOS/webots-controller", 
                f"--port={port}", 
                "SimulationManager.py",
                "--port", str(port)
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error in Webots process on port {port}: {e}")

    def start_webots_instances(self):
        """
        Start multiple Webots instances in parallel
        """
        self.webots_processes = []
        for port in self.ports:
            process = multiprocessing.Process(target=self.run_webots, args=(port,))
            self.webots_processes.append(process)
            process.start()
        
        # Give Webots some time to initialize
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

    def run_simulation(self, port, batch, ready_event, fitness_scores):
        """
        Run simulation for a batch of genomes on a specific port
        """
        self.save_batch(batch, port)
        
        # Run Webots controller
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, "SimulationManager.py")
        
        result = subprocess.run([
            "/Applications/Webots.app/Contents/MacOS/webots-controller", 
            f"--port={port}", 
            file_path,
            "--port", str(port)
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error occurred while running Webots controller on port {port}:")
            print(result.stderr)
        else:
            print(f"Webots controller on port {port} ran successfully.")
        
        # Parse fitness scores
        fitness_values = list(map(float, result.stdout.splitlines()))
        for i, genome in enumerate(batch):
            fitness_scores[genome] = fitness_values[i]

        ready_event.set()

    def evaluate_fitness(self):
        """
        Parallel fitness evaluation using multiprocessing
        """
        # Prepare genomes with their IDs
        genomes_with_ids = [(i, genome) for i, genome in enumerate(self.population)]
        
        # Distribute genomes across processors
        batches = self.distribute_genomes_evenly(genomes_with_ids)

        # Run simulations in parallel
        processes = []
        ready_events = []
        manager = multiprocessing.Manager()
        fitness_scores = manager.dict()
        
        for i in range(self.processor_count):
            ready_event = multiprocessing.Event()
            ready_events.append(ready_event)
            process = multiprocessing.Process(
                target=self.run_simulation, 
                args=(self.ports[i], batches[i], ready_event, fitness_scores)
            )
            processes.append(process)
            process.start()
        
        # Wait for all processes to signal readiness
        for ready_event in ready_events:
            ready_event.wait()

        print("All simulations have completed.")
        
        # Wait for all processes to complete
        for process in processes:
            process.join()
        
        # Update fitness scores and find best genome
        highestScore = 0
        genomeHighest = None
        for i, genome in enumerate(genomes_with_ids):
            fitness = fitness_scores[genome]
            self.fitness_scores[i] = fitness
            if fitness > highestScore:
                highestScore = fitness
                genomeHighest = genome
        
        # Save best genome
        with open(f"bestBest/lebron{self.generation}.pkl", "wb") as f:
            pickle.dump((highestScore, genomeHighest), f)
        
        return highestScore

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
            child = parent1.crossover(parent2)

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
            for process in self.webots_processes:
                os.kill(process.pid, signal.SIGTERM)
            
            # Wait for processes to terminate
            for process in self.webots_processes:
                process.join()

def main():
    population_manager = PopulationManager(
        population_size=100, 
        input_size=5, 
        hidden_size=10, 
        output_size=2
    )
    population_manager.train(num_generations=300)

if __name__ == '__main__':
    main()
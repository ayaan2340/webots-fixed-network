import numpy as np
import multiprocessing
import os
from pathlib import Path
import torch
from typing import List, Dict

from RecurrentNetwork import RecurrentNetwork
from simulation_manager import SimulationManager


class PopulationManager:
    def __init__(self, population_size: int, input_size: int,
                 hidden_size: int, output_size: int, num_workers: int = 10):
        self.population_size = population_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_workers = min(num_workers, multiprocessing.cpu_count())

        # Initialize population
        self.population = [
            RecurrentNetwork(input_size, hidden_size, output_size, genome_id)
            for genome_id in range(population_size)
        ]

        self.fitness_scores = [0.0] * population_size
        self.generation = 0

    def distribute_genomes_evenly(
            self, genomes: List[RecurrentNetwork]) -> List[List[RecurrentNetwork]]:
        """Distribute genomes across available processors evenly."""
        batches = []
        batch_size = len(genomes) // self.num_workers
        extra = len(genomes) % self.num_workers

        start = 0
        for i in range(self.num_workers):
            end = start + batch_size + (1 if i < extra else 0)
            batches.append(genomes[start:end])
            start = end

        return batches

    @staticmethod
    def evaluate_batch(batch: List[RecurrentNetwork], process_id: int,
                       result_dict: Dict, ready_event: multiprocessing.Event):
        """Evaluate a batch of genomes using the simulation."""
        manager = SimulationManager(process_id)

        for genome in batch:
            manager.reset_simulation_state()
            fitness = manager.evaluate_genome(genome)
            result_dict[genome.genome_id] = fitness

        ready_event.set()

    def evaluate_fitness(self) -> float:
        """Parallel fitness evaluation using custom simulation."""
        # Prepare for parallel processing
        batches = self.distribute_genomes_evenly(self.population)
        processes = []
        ready_events = []

        # Create shared memory for results
        manager = multiprocessing.Manager()
        fitness_dict = manager.dict()

        # Start evaluation processes
        for i, batch in enumerate(batches):
            ready_event = multiprocessing.Event()
            process = multiprocessing.Process(
                target=self.evaluate_batch,
                args=(batch, i, fitness_dict, ready_event)
            )
            processes.append(process)
            ready_events.append(ready_event)
            process.start()

        # Wait for all processes to complete
        for event in ready_events:
            event.wait()

        for process in processes:
            process.join()

        # Update fitness scores and track best
        highest_score = float('-inf')
        best_genome = None

        for i, genome in enumerate(self.population):
            if genome.genome_id in fitness_dict:
                self.fitness_scores[i] = fitness_dict[genome.genome_id]
                if self.fitness_scores[i] > highest_score:
                    highest_score = self.fitness_scores[i]
                    best_genome = genome

        # Save best genome
        if best_genome is not None:
            save_dir = Path("best_genomes")
            save_dir.mkdir(exist_ok=True)
            torch.save({
                'fitness': highest_score,
                'genome': best_genome.state_dict(),
                'generation': self.generation
            }, save_dir / f"best_genome_gen_{self.generation}.pt")

        return highest_score

    def tournament_selection(self, tournament_size: int = 5) -> RecurrentNetwork:
        """Select the best individual from a random tournament."""
        indices = np.random.choice(len(self.population), tournament_size, replace=False)
        tournament_fitness = [self.fitness_scores[i] for i in indices]
        winner_index = indices[np.argmax(tournament_fitness)]
        return self.population[winner_index]

    def create_next_generation(self):
        """Create next generation through selection, crossover, and mutation."""
        new_population = []

        # Keep the best individual (elitism)
        best_idx = np.argmax(self.fitness_scores)
        best_genome = self.population[best_idx]
        # Create child of best genome and copy its weights
        elite = best_genome.make_child(0)  # Use genome_id 0 for elite
        for param_elite, param_best in zip(elite.parameters(), best_genome.parameters()):
            param_elite.data.copy_(param_best.data)
        new_population.append(elite)

        # Create rest of population
        while len(new_population) < self.population_size:
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()

            child = parent1.crossover(parent2, len(new_population))
            child.mutate()
            new_population.append(child)

        self.population = new_population
        self.fitness_scores = [0.0] * self.population_size

    def train(self, num_generations: int):
        """Train the population over multiple generations."""
        try:
            for self.generation in range(num_generations):
                print(f"\nGeneration {self.generation}")

                # Evaluate fitness
                best_fitness = self.evaluate_fitness()
                avg_fitness = np.mean(self.fitness_scores)

                print(f"Best Fitness: {best_fitness:.2f}")
                print(f"Average Fitness: {avg_fitness:.2f}")

                # Create next generation
                self.create_next_generation()

        except KeyboardInterrupt:
            print("\nTraining interrupted by user")

        finally:
            # Save final population
            save_dir = Path("final_population")
            save_dir.mkdir(exist_ok=True)
            for genome in self.population:
                genome.save(save_dir / f"genome_{genome.genome_id}.h5")


def main():
    # Initialize and run evolution
    population_manager = PopulationManager(
        population_size=256,
        input_size=6,
        hidden_size=10,
        output_size=2,
        num_workers=8  # Adjust based on your CPU cores
    )

    population_manager.train(num_generations=300)


if __name__ == '__main__':
    main()

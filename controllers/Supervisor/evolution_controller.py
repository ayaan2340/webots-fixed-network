import numpy as np
import multiprocessing
import os
from pathlib import Path
import torch
from typing import List, Dict

from RecurrentNetwork import RecurrentNetwork
from simulation_manager import SimulationManager
from syllabus import SyllabusGenerator, evaluate_batch_with_frames


class PopulationManager:
    def __init__(self, population_size: int, input_size: int,
                 hidden_size: int, output_size: int, num_workers: int = 8):
        self.population_size = population_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_workers = min(num_workers, multiprocessing.cpu_count())

        # Initialize syllabus manager
        self.syllabus_generator = SyllabusGenerator()

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
        """Parallel fitness evaluation with frame collection."""
        batches = self.distribute_genomes_evenly(self.population)
        processes = []
        ready_events = []

        # Create shared memory for results and frame collection
        manager = multiprocessing.Manager()
        fitness_dict = manager.dict()
        frame_queue = multiprocessing.Queue()

        # Start evaluation processes
        for i, batch in enumerate(batches):
            ready_event = multiprocessing.Event()
            process = multiprocessing.Process(
                target=evaluate_batch_with_frames,
                args=(batch, i, fitness_dict, frame_queue, ready_event)
            )
            processes.append(process)
            ready_events.append(ready_event)
            process.start()

        # Collect frames while waiting for processes
        collected_frames = {}
        active_processes = list(range(len(processes)))  # Track indices of active processes

        while active_processes:
            # Check for new frames
            try:
                while True:
                    network_id, frames, fitness = frame_queue.get_nowait()
                    if network_id not in collected_frames:
                        collected_frames[network_id] = []
                    collected_frames[network_id].extend((f, fitness) for f in frames)
            except multiprocessing.queues.Empty:
                pass

            # Check for completed processes
            for process_idx in active_processes[:]:  # Create a copy for safe iteration
                if ready_events[process_idx].is_set():
                    processes[process_idx].join()
                    active_processes.remove(process_idx)

        # Update fitness scores and generate syllabus
        for i, genome in enumerate(self.population):
            self.fitness_scores[i] = fitness_dict.get(genome.genome_id, 0.0)

        # Generate syllabus for next generation
        self.current_syllabus = self.syllabus_generator.generate_syllabus(collected_frames)

        return max(self.fitness_scores)

    def tournament_selection(self, tournament_size: int = 5) -> RecurrentNetwork:
        """Select the best individual from a random tournament."""
        indices = np.random.choice(len(self.population), tournament_size, replace=False)
        tournament_fitness = [self.fitness_scores[i] for i in indices]
        winner_index = indices[np.argmax(tournament_fitness)]
        return self.population[winner_index]

    def evaluate_network_on_syllabus(self, network: RecurrentNetwork) -> np.ndarray:
        """Evaluate a network's responses to the current syllabus questions."""
        responses = []
        # Reset network state before evaluation
        network.hidden_state = None

        for frame in self.current_syllabus:
            output = network.forward(frame.inputs)
            responses.append(output.detach().numpy())

        return np.array(responses)

    def create_next_generation(self):
        """Create next generation using syllabus-based approach."""
        if not hasattr(self, 'current_syllabus'):
            # Fallback to regular evolution if no syllabus available
            return self._create_next_generation_regular()

        # Create expanded population
        expanded_size = self.population_size * 2  # Double the population size
        expanded_population = []

        # Keep top performers (20%)
        sorted_indices = np.argsort(self.fitness_scores)
        top_performers = [self.population[i]
                          for i in sorted_indices[-int(self.population_size * 0.2):]]
        expanded_population.extend(top_performers)

        # Generate new individuals
        while len(expanded_population) < expanded_size:
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            child = parent1.crossover(parent2, len(expanded_population))
            child.mutate()
            expanded_population.append(child)

        # Evaluate expanded population on syllabus
        syllabus_responses = {}
        for network in expanded_population:
            responses = self.evaluate_network_on_syllabus(network)
            syllabus_responses[network.genome_id] = responses

        # Calculate response distances and novelty
        final_population = []
        final_population.extend(top_performers)  # Keep top performers

        remaining = expanded_population[len(top_performers):]
        novelty_scores = []

        for network in remaining:
            responses = syllabus_responses[network.genome_id]
            other_responses = [syllabus_responses[n.genome_id] for n in remaining if n != network]

            # Calculate average distance to other responses
            distances = []
            for other in other_responses:
                dist = np.mean([np.linalg.norm(r1 - r2) for r1, r2 in zip(responses, other)])
                distances.append(dist)

            novelty_scores.append(np.mean(distances))

        # Select most novel individuals to complete population
        novel_indices = np.argsort(novelty_scores)[-int(self.population_size * 0.8):]
        final_population.extend(remaining[i] for i in novel_indices)

        # Update population and reset fitness scores
        self.population = final_population[:self.population_size]
        self.fitness_scores = [0.0] * self.population_size

    def _create_next_generation_regular(self):
        """Fallback method for regular evolution without syllabus."""
        new_population = []

        # Keep best individual (elitism)
        best_idx = np.argmax(self.fitness_scores)
        best_genome = self.population[best_idx]
        elite = best_genome.make_child(0)
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
        population_size=10,
        input_size=6,
        hidden_size=10,
        output_size=2,
        num_workers=5  # Adjust based on your CPU cores
    )

    population_manager.train(num_generations=300)


if __name__ == '__main__':
    main()

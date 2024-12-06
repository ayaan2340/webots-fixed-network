import torch
import torch.nn as nn
import numpy as np
import multiprocessing
import os
import subprocess
import time
import pickle

from pytorch_network import RecurrentNetwork
from population_manager import PopulationManager

generation = 0

def run_simulation(port, batch, ready_event, fitness_scores):
    with open(f"genome_data/genome_data{port}.pkl", "wb") as f:
        for genome_id, genome in batch:
            pickle.dump((genome_id, genome), f)
    f.close()
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, "NEATSupervisor.py")
    # CHANGE FIRST LINE OF SUBPROCESS RUN TO YOUR WEBOTS-CONTROLLER FILE
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
        print(result.stdout)
    
    # Assuming the controller outputs fitness scores line by line
    fitness_values = list(map(float, result.stdout.splitlines()))
    for i, (genome_id, genome) in enumerate(batch):
        fitness_scores[genome_id] = fitness_values[i]

    ready_event.set()
    
def distribute_genomes_evenly(genomes, processor_count):
    total_genomes = len(genomes)
    batch_size = total_genomes // processor_count
    extra_size = batch_size + 1  # Number of genomes for batches with one extra genome
    extra_batches = total_genomes % processor_count
    batches = []
    
    genome_index = 0
    # Distribute batches with the extra genome
    for _ in range(extra_batches):
        batch = genomes[genome_index:genome_index + extra_size]
        batches.append(batch)
        genome_index += extra_size

    # Distribute batches with the base genome size
    for _ in range(processor_count - extra_batches):
        batch = genomes[genome_index:genome_index + batch_size]
        batches.append(batch)
        genome_index += batch_size

    return batches

def eval_genomes(genomes, config):
    global generation
    highestScore = 0
    genomeHighest = 0
    processor_count = 10
    ports = []
    for i in range(processor_count):
        port = 10000 + i
        ports.append(port)

    batches = distribute_genomes_evenly(genomes, processor_count)

    # Run simulations in parallel
    processes = []
    ready_events = []
    manager = multiprocessing.Manager()
    fitness_scores = manager.dict()
    for i in range(processor_count):
        ready_event = multiprocessing.Event()
        ready_events.append(ready_event)
        process = multiprocessing.Process(target=run_simulation, args=(ports[i], batches[i], ready_event, fitness_scores))
        processes.append(process)
        process.start()
    
    # Wait for all processes to signal readiness
    for ready_event in ready_events:
        ready_event.wait()

    print("All simulations have completed.")
    # Wait for all processes to complete
    for process in processes:
        process.join()
    
    for genome_id, genome in genomes:
        genome.fitness = fitness_scores[genome_id]
        if fitness_scores[genome_id] > highestScore:
            highestScore = fitness_scores[genome_id]
            genomeHighest = genome
    
    with open(f"bestBest/lebron{generation}.pkl", "wb") as f:
        pickle.dump((highestScore, genomeHighest), f)
    f.close()
    generation += 1

def run_pytorch_neat():
    population_manager = PopulationManager(
        population_size=100, 
        input_size=5, 
        hidden_size=10, 
        output_size=2
    )
    population_manager.train(num_generations=300)

if __name__ == '__main__':
    webots_processes = []
    
    for i in range(10):
        port = 10000 + i
        process = multiprocessing.Process(target=run_webots, args=(port,))
        webots_processes.append(process)
        process.start()
    
    time.sleep(15)
    
    # Run PyTorch NEAT
    run_pytorch_neat()

    # Wait for all Webots instances to terminate
    for process in webots_processes:
        process.join()

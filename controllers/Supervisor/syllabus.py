from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
import random
from pathlib import Path
import torch
import multiprocessing
from multiprocessing.shared_memory import SharedMemory
import pickle
from RecurrentNetwork import RecurrentNetwork
from simulation_manager import SimulationManager, SyllabusFrame


class ProcessLocalBuffer:
    """Process-local storage for frames."""

    def __init__(self, max_frames_per_network: int = 100):
        self.max_frames = max_frames_per_network
        self.frames: Dict[int, List[SyllabusFrame]] = {}

    def add_frame(self, network_id: int, frame: SyllabusFrame):
        """Add frame to local buffer, maintaining max size per network."""
        if network_id not in self.frames:
            self.frames[network_id] = []

        frames = self.frames[network_id]
        frames.append(frame)

        # If over max size, randomly subsample
        if len(frames) > self.max_frames:
            self.frames[network_id] = random.sample(frames, self.max_frames)

    def get_successful_frames(self, network_id: int) -> List[SyllabusFrame]:
        """Get all successful frames for a network."""
        return [f for f in self.frames.get(network_id, []) if f.success]


def evaluate_batch_with_frames(batch: List[RecurrentNetwork], process_id: int,
                               result_dict: Dict, frame_queue: multiprocessing.Queue,
                               ready_event: multiprocessing.Event):
    """Evaluate a batch of genomes while collecting frames locally."""
    manager = SimulationManager(process_id)
    local_buffer = ProcessLocalBuffer()

    for genome in batch:
        fitness = 0
        for trial in range(manager.num_trials):
            manager.reset_simulation_state()
            trial_fitness = manager.evaluate_single_trial(genome, local_buffer)
            fitness += trial_fitness

        avg_fitness = fitness / manager.num_trials
        result_dict[genome.genome_id] = avg_fitness

        # If this was a good performer, send its frames to the main process
        if avg_fitness > np.mean(list(result_dict.values())):
            successful_frames = local_buffer.get_successful_frames(genome.genome_id)
            if successful_frames:
                # Send frames and fitness as a tuple
                frame_queue.put((genome.genome_id, successful_frames, avg_fitness))

    ready_event.set()


class SyllabusGenerator:
    """Handles syllabus generation in the main process."""

    def __init__(self, syllabus_size: int = 15):
        self.syllabus_size = syllabus_size

    def calculate_frame_novelty(self, frame: SyllabusFrame,
                                frame_set: List[SyllabusFrame]) -> float:
        if not frame_set:
            return 1.0

        novelties = []
        for other in frame_set:
            # Calculate input space difference
            input_diff = np.linalg.norm(frame.inputs - other.inputs)

            # Calculate state space difference (position and rotation)
            pos_diff = np.sqrt((frame.position[0] - other.position[0])**2 +
                               (frame.position[1] - other.position[1])**2)
            rot_diff = min(abs(frame.rotation - other.rotation),
                           2 * np.pi - abs(frame.rotation - other.rotation))

            # Weighted combination
            total_diff = input_diff * 0.4 + pos_diff * 0.4 + rot_diff * 0.2
            novelties.append(total_diff)

        return float(np.mean(novelties))

    def generate_syllabus(
            self, collected_frames: Dict[int, List[Tuple[SyllabusFrame, float]]]) -> List[SyllabusFrame]:
        """Generate syllabus from collected frames."""
        candidate_frames = []

        # First, gather all frames and calculate their novelty
        for network_id, frames in collected_frames.items():
            for frame, fitness in frames:
                novelty = self.calculate_frame_novelty(frame, [f for f, _ in candidate_frames])
                # Combine novelty and fitness for importance
                importance = novelty * 0.6 + (fitness / np.max([f for _, f in frames])) * 0.4
                candidate_frames.append((frame, importance))

        # Select frames for syllabus using weighted sampling
        syllabus = []
        while len(syllabus) < self.syllabus_size and candidate_frames:
            weights = np.array([imp for _, imp in candidate_frames])
            probs = weights / weights.sum()
            chosen_idx = np.random.choice(len(candidate_frames), p=probs)
            syllabus.append(candidate_frames.pop(chosen_idx)[0])

        return syllabus


def evaluate_network_on_syllabus(network: RecurrentNetwork,
                                 syllabus: List[SyllabusFrame]) -> np.ndarray:
    """Evaluate a network's responses to syllabus questions."""
    responses = []
    for frame in syllabus:
        output = network.forward(frame.inputs)
        responses.append(output.detach().numpy())
    return np.array(responses)

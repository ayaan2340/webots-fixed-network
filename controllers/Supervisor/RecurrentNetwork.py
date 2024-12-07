import torch
from torch.distributions import Normal
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RecurrentNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, genome_id,
                 crossover_prob=0.25, mutation_rate=0.25, mutation_strength=0.1,
                 learning_rate=0.001):
        super().__init__()
        self.hidden_size = hidden_size
        self.genome_id = genome_id
        self.crossover_prob = crossover_prob
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength

        # Network architecture
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.recurrent_layer = nn.Linear(hidden_size, hidden_size)

        # Separate output layers for angle and speed
        self.angle_layer = nn.Linear(hidden_size, 1)  # Just mean angle, no std
        self.speed_layer = nn.Linear(hidden_size, 1)  # Speed (evolved via GA)

        self.tanh = nn.Tanh()
        self.hidden_state = None

        # RL components
        self.optimizer = torch.optim.Adam([
            # Only optimize angle-related parameters
            *self.input_layer.parameters(),
            *self.recurrent_layer.parameters(),
            *self.angle_layer.parameters()
        ], lr=learning_rate)

        # For RL training
        self.predicted_angles = []
        self.optimal_angles = []
        self.angle_losses = []

    def __hash__(self):
        return self.genome_id

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)

        if self.hidden_state is None:
            self.hidden_state = torch.zeros(self.hidden_size)

        # Shared feature extraction
        input_transformed = self.tanh(self.input_layer(x))
        recurrent_output = self.tanh(
            self.recurrent_layer(self.hidden_state) + input_transformed
        )
        self.hidden_state = recurrent_output

        # Separate outputs for angle and speed
        angle = torch.tanh(self.angle_layer(recurrent_output)) * np.pi  # [-π, π]
        speed = torch.tanh(self.speed_layer(recurrent_output))  # [-1, 1]

        return angle.squeeze(), speed.squeeze()

    def select_action(self, state, optimal_angle):
        """Get actions and store angle info for RL training"""
        angle, speed = self.forward(state)

        # Store angle information for training
        self.predicted_angles.append(angle)
        self.optimal_angles.append(torch.tensor(optimal_angle))

        return float(speed), float(angle)

    def reset_rl(self):
        """Reset RL-specific components"""
        self.saved_log_probs = []
        self.rewards = []
        self.optimizer.zero_grad()

    def reset(self):
        """Full reset including hidden state and RL components"""
        self.hidden_state = None
        self.reset_rl()

    def train_episode(self):
        """Train angle prediction using collected episode data"""
        if not self.predicted_angles:  # Skip if no data
            return

        # Convert lists to tensors
        predicted = torch.stack(self.predicted_angles)
        optimal = torch.stack(self.optimal_angles)

        # Compute angle differences and loss
        angle_diff = self.normalize_angle(predicted - optimal)
        loss = torch.mean(angle_diff ** 2)  # MSE loss

        # Update network to minimize angle error
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Store loss for monitoring
        self.angle_losses.append(float(loss))

        # Clear episode data
        self.predicted_angles = []
        self.optimal_angles = []

    @staticmethod
    def normalize_angle(angle):
        """Normalize angle difference to [-π, π]"""
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    def make_child(self, genome_id):
        """Create child network with fresh RL components"""
        child = RecurrentNetwork(
            input_size=self.input_layer.in_features,
            hidden_size=self.hidden_size,
            output_size=self.output_layer.out_features,
            genome_id=genome_id,
            crossover_prob=self.crossover_prob
        )
        # Create new optimizer for child
        child.optimizer = torch.optim.Adam(
            child.parameters(),
            lr=child.optimizer.param_groups[0]['lr'])
        return child

    def mutate(self):
        """
        Mutate the network weights with a given probability and strength.
        Args:
            mutation_rate: Probability of a weight being mutated
            mutation_strength: Magnitude of weight changes
        """
        for param in self.parameters():
            mask = torch.rand(param.data.shape) < self.mutation_rate
            mutation = torch.randn(param.data.shape) * self.mutation_strength
            param.data[mask] += mutation[mask]

    def crossover(self, other_network, genome_id):
        """Modified crossover to handle RL components"""
        offspring = self.make_child(genome_id)  # This creates fresh RL components

        # Perform standard crossover for network parameters
        for (name1, param1), (name2, param2) in zip(
            self.named_parameters(),
            other_network.named_parameters()
        ):
            mask = torch.rand(param1.data.shape) < self.crossover_prob
            offspring_param = param1.clone()
            offspring_param[mask] = param2[mask]
            dict(offspring.named_parameters())[name1].data.copy_(offspring_param)

        return offspring

    def save(self, path):
        torch.save(self, path)

    @staticmethod
    def load(path, **kwargs):
        """
        Load a network from a saved file.

        Args:
            path: Path to the saved network file

        Returns:
            RecurrentNetwork: Loaded network with weights restored
        """
        saved_data = torch.load(path)

        # Handle case where saved data is a dictionary with genome params
        if isinstance(saved_data, dict) and 'genome' in saved_data:
            # Create a new network with default parameters
            # Note: This assumes we know the architecture. You might need to
            # store/load these parameters separately if they vary
            network = RecurrentNetwork(**kwargs)

            # Load the state dict from the saved genome parameters
            network.load_state_dict(saved_data['genome'])

            # Store the fitness score as an attribute
            network.fitness_score = saved_data['fitness']

            return network

        # Handle case where saved data is the network itself (old format)
        return saved_data

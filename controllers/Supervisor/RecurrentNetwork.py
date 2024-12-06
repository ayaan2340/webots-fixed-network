import torch
import torch.nn as nn
import numpy as np


class RecurrentNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, genome_id,
                 crossover_prob=0.25, mutation_rate=0.25, mutation_strength=0.1):
        super(RecurrentNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.genome_id = genome_id
        self.crossover_prob = crossover_prob
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength

        # Input to hidden layer
        self.input_layer = nn.Linear(input_size, hidden_size)

        # Hidden to hidden recurrent connection
        self.recurrent_layer = nn.Linear(hidden_size, hidden_size)

        # Hidden to output layer
        self.output_layer = nn.Linear(hidden_size, output_size)

        # Activation functions
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        # Initialize hidden state
        self.hidden_state = None

    def __hash__(self):
        return self.genome_id

    def forward(self, x):
        # Ensure x is a torch tensor
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)

        # If this is the first forward pass, initialize hidden state
        if self.hidden_state is None:
            self.hidden_state = torch.zeros(self.hidden_size)

        with torch.no_grad():
            # Process input through layers
            input_transformed = self.tanh(self.input_layer(x))

            # Combine current input with previous hidden state
            recurrent_output = self.tanh(
                self.recurrent_layer(self.hidden_state) + input_transformed
            )

            # Update hidden state
            self.hidden_state = recurrent_output

            # Generate output
            output = self.relu(self.output_layer(recurrent_output))

        return output.detach().numpy()

    def reset_hidden_state(self):
        """Reset the hidden state to zero."""
        self.hidden_state = torch.zeros(self.hidden_size)

    def make_child(self, genome_id):
        # Create a new network with the same architecture
        return RecurrentNetwork(
            input_size=self.input_layer.in_features,
            hidden_size=self.hidden_size,
            output_size=self.output_layer.out_features,
            crossover_prob=self.crossover_prob,
            genome_id=genome_id,
        )

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
        """
        Perform crossover with another network.

        Args:
            other_network (RecurrentNetwork): The second parent network

        Returns:
            RecurrentNetwork: A new network with weights crossed over from both parents
        """
        # Validate that networks have compatible architectures
        if not all([
            self.input_layer.in_features == other_network.input_layer.in_features,
            self.hidden_size == other_network.hidden_size,
            self.output_layer.out_features == other_network.output_layer.out_features
        ]):
            raise ValueError("Networks must have identical architecture for crossover")

        offspring = self.make_child(genome_id)

        # Perform crossover for each parameter
        for (name1, param1), (name2, param2) in zip(
            self.named_parameters(),
            other_network.named_parameters()
        ):
            # Create a mask based on crossover probability
            mask = torch.rand(param1.data.shape) < self.crossover_prob

            # Create offspring weights by selecting from parents based on mask
            offspring_param = param1.clone()
            offspring_param[mask] = param2[mask]

            # Assign the crossed-over weights to the offspring network
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

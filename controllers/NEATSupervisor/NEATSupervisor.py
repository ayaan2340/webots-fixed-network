import torch
import os
import numpy as np
from controller import Supervisor
import argparse
import random

class PyTorchNEATSupervisor(Supervisor):
    def __init__(self):
        super().__init__()
        parser = argparse.ArgumentParser(description="Parse the Webots controller port argument.")
        parser.add_argument('--port', type=int, required=True, help='Port number for the Webots controller.')
        args = parser.parse_args()
        
        self.port = args.port if args.port else 1234
        self.time_step = int(self.getBasicTimeStep())

        # Same initialization as before
        # ... [rest of the initialization from the previous NEATSupervisor remains the same]

    def evaluate_network(self, network):
        """
        Evaluate a PyTorch network's fitness.
        Maintains similar logic to the previous evaluate_genome method.
        """
        network.reset_hidden_state()
        self.start_time = self.getTime()
        self.fitness = 0.0
        reached = False
        timeCounter = 0.0
        onRoadCounter = 0.0
        rightDirection = 0.0
        rightAngle = 0
        self.max_simulation_time = max(12, (self.shortest_path / 5.0))

        while self.step(self.time_step) != -1:
            current_time = self.getTime()
            if self.is_on_road():
                onRoadCounter += 1
            timeCounter += 1
            
            if current_time - self.start_time > self.max_simulation_time:
                break
            
            if self.reached_end():
                reached = True
                break
            
            inputs = torch.FloatTensor(self.preprocess_inputs())
            outputs = network(inputs)
            
            # Convert torch tensor to numpy for setting controls
            control_outputs = outputs.detach().numpy()
            self.set_controls(control_outputs)
            
            rightDirection += self.calculate_distance()
            rightAngle += self.calculate_angle()

        if reached:
            self.fitness += 5
        
        onRoadPenalty = -10 * ((timeCounter - onRoadCounter) / timeCounter)
        gettingCloser = 3 * rightDirection / self.max_simulation_time
        wrongDirectionPenalty = 30 * (rightAngle / self.max_simulation_time)
        
        self.fitness = onRoadPenalty + gettingCloser + wrongDirectionPenalty
        return self.fitness

def run_simulation(config):
    supervisor = PyTorchNEATSupervisor()
    # Rest of the method remains similar to previous implementation
    # Load PyTorch networks and evaluate them
    # Use similar file loading mechanism as before
```

Key Changes and Notes:
1. Replaced NEAT's genome with a PyTorch Recurrent Neural Network
2. Maintained similar input/output structure
3. Implemented tournament selection and custom mutation
4. Added recurrent connections via hidden state
5. Modified fitness calculation to work with PyTorch networks
6. Kept the overall project structure and simulation workflow intact

Recommendations for Implementation:
1. Ensure you have the required libraries: `torch`, `numpy`
2. Make sure the Webots simulation interface remains consistent
3. The PyTorch implementation allows for more flexible network architectures

Would you like me to elaborate on any part of the implementation or explain the design choices?
'''

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

'''

from controller import Supervisor
import cv2
import random
import pickle
import os
import argparse
import random
from pathlib import Path
from RecurrentNetwork import RecurrentNetwork

class SimulationManager(Supervisor):
    def __init__(self):
        super().__init__()
        parser = argparse.ArgumentParser(description="Parse the Webots controller port argument.")
        parser.add_argument('--port', type=int, required=True, help='Port number for the Webots controller.')
        args = parser.parse_args()
        if args.port:
            self.port = args.port
        else:
            self.port = 1234
        self.time_step = int(self.getBasicTimeStep())

        # Initialize the robot node first
        self.robot_node = self.getFromDef("car")
        if self.robot_node is None:
            raise ValueError("Robot node with DEF name 'car' not found.")

        self.end_obj = self.getFromDef("redEnd")
        if self.end_obj is None:
            raise ValueError("End node with DEF name 'redEnd' not found.")
        self.green_start = self.getFromDef("greenStart")
        if self.green_start is None:
            raise ValueError("Start node with DEF name 'greenStart' not found.")

        self.roads = []
        self.vertical_roads = [
            self.getFromDef('Road_0'),
            self.getFromDef('Road_9'),
            self.getFromDef('Road_10'),
            self.getFromDef('Road_11'),
            self.getFromDef('Road_12'),
            self.getFromDef('Road_13'),
        ]
        self.horizontal_roads = [
            self.getFromDef('Road_1'),
            self.getFromDef('Road_2'),
            self.getFromDef('Road_3'),
            self.getFromDef('Road_4'),
            self.getFromDef('Road_5'),
            self.getFromDef('Road_6'),
            self.getFromDef('Road_7'),
            self.getFromDef('Road_8'),
        ]
        self.road_intersections = [
            self.getFromDef('Road_14'),
            self.getFromDef('Road_15'),
            self.getFromDef('Road_16'),
            self.getFromDef('Road_17'),
        ]
        self.roads.extend(self.vertical_roads)
        self.roads.extend(self.horizontal_roads)
        self.roads.extend(self.road_intersections)
        
        self.previous_position = self.getPos()
        self.time_stuck = 0.0
        self.movement_threshold = 0.01
        self.crash_delay = 1.0
        self.fitness = 0.0
        self.max_simulation_time = 15  # seconds
        self.shortest_path = self.shortestPath(self.previous_position, self.end_obj.getField("translation").getSFVec3f())
        self.speed = 0.0
        self.previous_distance = 0.0
        self.steering_angle = 0.0
        self.distance_travelled = 0.0
        self.angle_to_end = 0.0
        self.initialize_devices()

    def initialize_devices(self):
        self.left_steer = self.getDevice("left_steer")
        self.right_steer = self.getDevice("right_steer")
        self.left_front_wheel = self.getDevice("left_front_wheel")
        self.right_front_wheel = self.getDevice("right_front_wheel")

        self.camera = self.getDevice("camera")
        if not self.camera:
            raise ValueError("Camera device not found.")
        self.camera.enable(self.time_step)

        if not all([self.left_steer, self.right_steer, self.left_front_wheel, self.right_front_wheel]):
            raise ValueError("One or more motor devices not found.")

        self.left_front_wheel.setPosition(float('inf'))
        self.right_front_wheel.setPosition(float('inf'))
        self.left_front_wheel.setVelocity(0)
        self.right_front_wheel.setVelocity(0)

    def preprocess_camera_data(self):
        """Preprocess the camera image and return a flattened grayscale array."""
        width = self.camera.getWidth()
        height = self.camera.getHeight()
        image = np.frombuffer(self.camera.getImage(), dtype=np.uint8).reshape((height, width, 4))
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)  # Convert BGRA to grayscale
        resized_image = cv2.resize(gray_image, (10, 10))  # Resize to reduce dimensionality
        flattened_image = resized_image.flatten() / 255.0  # Normalize pixel values
        return flattened_image

    def preprocess_inputs(self):
        # input_list = self.preprocess_camera_data().tolist()
        input_list = []
        input_list.append(self.get_angle_to_end() / np.pi)
        carPos = self.getPos()
        endPos = self.end_obj.getPosition()
        input_list.append((carPos[0] - endPos[0]) / 150)
        input_list.append((carPos[1] - endPos[1]) / 150)
        input_list.append(self.speed / 30.0)
        input_list.append(self.steering_angle / 0.5)
        # if self.port == 10000:
        #     with open("outputs.txt", "w") as f:
        #         f.write(str(input_list))
        #     f.close()
        return np.array(input_list)
            

    def normalize_angle(self, angle):
        """Normalize angle to the range [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def get_angle_to_end(self):
        # Get the car's current position
        car_pos = np.array(self.getPos()[:2])  # x, y position of the car
        end_pos = np.array(self.end_obj.getField("translation").getSFVec3f()[:2])  # x, y position of the target

        # Calculate the angle to the target position
        angle_to_end = np.arctan2(end_pos[1] - car_pos[1], end_pos[0] - car_pos[0])
        # self.robot_node.getField("rotation").setSFRotation([0, 0, 1, angle_to_end])
        # Get the car's current orientation angle
        car_rotation_field = self.robot_node.getField("rotation")
        car_orientation_angle = car_rotation_field.getSFRotation()[3]  # Extract angle (α) from rotation
        # Calculate the relative angle
        relative_angle = self.normalize_angle(angle_to_end - car_orientation_angle)
        self.angle_to_end = relative_angle / np.pi
        return float(relative_angle)

    def get_random_road_position(self):
        """Selects a random road segment and returns its translation position."""
        if not self.roads:
            return [0, 0, 0]  # Default position to avoid crashing
        selected_road = random.choice(self.roads)
        road_position = selected_road.getField("translation").getSFVec3f()
        return road_position

    def teleport_object(self, obj, position):
        """Teleports a specific object to the given position."""
        if obj:
            obj.getField("translation").setSFVec3f(position)

    def randomize_goals(self):
        # Teleport redEnd and greenStart at the start of each generation
        new_red_end_position = self.get_random_road_position()
        new_green_start_position = self.get_random_road_position()
        while new_red_end_position == new_green_start_position:
            new_red_end_position = self.get_random_road_position()
            new_green_start_position = self.get_random_road_position()

        self.teleport_object(self.end_obj, new_red_end_position)
        self.teleport_object(self.green_start, new_green_start_position)
        
    def reset_simulation_state(self):
        """Manually reset the simulation state."""
        # Reset velocities
        self.left_front_wheel.setVelocity(0)
        self.right_front_wheel.setVelocity(0)

        # Reset internal states
        self.previous_position = self.getPos()
        self.time_stuck = 0.0
        self.fitness = 0.0
        self.speed = 0.0
        self.steering_angle = 0.0
        self.distance_travelled = 0.0
        # Teleport the car back to the green start position
        start_position = self.green_start.getField("translation").getSFVec3f()
        self.teleport_object(self.robot_node, start_position)
        self.robot_node.getField("rotation").setSFRotation([0, 0, 1, 2 * (random.random() - 0.5) * np.pi])
        self.robot_node.resetPhysics()
        self.angle_to_end = 0.0
        self.shortest_path = self.shortestPath(start_position, self.end_obj.getField("translation").getSFVec3f())
        self.previous_distance = np.linalg.norm(
            np.array(self.end_obj.getField("translation").getSFVec3f()[:2]) -
            np.array(start_position[:2])
        )

        for _ in range(10):  # Step multiple times to apply changes
            self.step(self.time_step)

    def getPos(self):
        """Get the current position of the car."""
        return self.robot_node.getPosition()

    def calculate_distance(self):
        """
        Calculate fitness based on:
        - Speed
        - Distance from the goal
        """
        current_pos = self.getPos()
        distance_to_goal = np.linalg.norm(
            np.array(self.end_obj.getField("translation").getSFVec3f()[:2]) -
            np.array(current_pos[:2])
        )
        add = 0
        if distance_to_goal < self.previous_distance:
            add = -distance_to_goal + self.previous_distance
        # elif -0.01 <= distance_to_goal - self.previous_distance <= 0.001:
        #     add = -0.1
        else: 
            add = (-distance_to_goal + self.previous_distance) / 2
        self.previous_distance = distance_to_goal
        return add

    def set_controls(self, outputs):
        max_speed = 30.0
        max_steering_angle = 0.5
        self.speed = outputs[0] * max_speed
        self.steering_angle = outputs[1] * max_steering_angle

        self.left_front_wheel.setVelocity(self.speed)
        self.right_front_wheel.setVelocity(self.speed)

        self.left_steer.setPosition(self.steering_angle)
        self.right_steer.setPosition(self.steering_angle)
    
    def reached_end(self):
        carPos = self.getPos()
        endPos = self.end_obj.getField("translation").getSFVec3f()
        if (endPos[0] - 2 <= carPos[0] <= endPos[0] + 2 and
           endPos[1] - 2 <= carPos[1] <= endPos[1] + 2):
           return True
        return False
    
    def calculate_angle(self):
        if self.angle_to_end <= -0.7 and self.angle_to_end >= 0.7:
            return -abs(self.angle_to_end)
        else:
            return 0


    def evaluate_genome(self, genome: RecurrentNetwork):
        """Evaluate a single genome's fitness."""
        genome.hidden_state = None
        self.start_time = self.getTime()
        self.fitness = 0.0
        self.max_simulation_time = max(12, (self.shortest_path / 5.0))
        while self.step(self.time_step) != -1:
            current_time = self.getTime()
            if current_time - self.start_time > self.max_simulation_time:
                break
            if self.reached_end():
                self.fitness = 100 / (self.getTime() - self.start_time)
                return self.fitness
            # if not self.is_on_road():
            #     return -1
            inputs = self.preprocess_inputs()
            outputs = genome.forward(inputs)
            self.set_controls(outputs)

        return (100 - self.calculate_distance()) / self.max_simulation_time
        
    def shortestPath(self, carPos, endPos):
        if carPos[0] == endPos[0]:
            for road in self.road_intersections:
                if carPos[0] == road.getField("translation").getSFVec3f()[0]:
                    return abs(carPos[0] - endPos[0]) + abs(carPos[1] - endPos[1])
            return abs(carPos[0] - endPos[0]) + abs(carPos[1] - endPos[1]) + 30
        if carPos[1] == endPos[1]:
            for road in self.road_intersections:
                if carPos[1] == road.getField("translation").getSFVec3f()[1]:
                    return abs(carPos[0] - endPos[0]) + abs(carPos[1] - endPos[1])
            return abs(carPos[0] - endPos[0]) + abs(carPos[1] - endPos[1]) + 30
        return abs(carPos[0] - endPos[0]) + abs(carPos[1] - endPos[1])
        
    def is_on_road(self):
        car_position = self.getPos()  # [x, y, z]
        for road in self.roads:
            center = road.getField("translation").getSFVec3f()
            # Define road boundaries based on your world setup
            # Example for horizontal roads:
            if road in self.horizontal_roads:
                road_length = 15  # Adjust as per your world
                road_width = 7
                if (center[0] - road_length / 2 <= car_position[0] <= center[0] + road_length / 2 and
                    center[1] - road_width / 2 <= car_position[1] <= center[1] + road_width / 2):
                    return True
            # Example for vertical roads:
            elif road in self.vertical_roads:
                road_length = 15
                road_width = 7
                if (center[0] - road_width / 2 <= car_position[0] <= center[0] + road_width / 2 and
                    center[1] - road_length / 2 <= car_position[1] <= center[1] + road_length / 2):
                    return True
            # Example for intersections:
            elif road in self.road_intersections:
                road_length = 19
                road_width = 19
                if (center[0] - road_length / 2 <= car_position[0] <= center[0] + road_length / 2 and
                    center[1] - road_width / 2 <= car_position[1] <= center[1] + road_width / 2):
                    return True
        return False

def loadall(dirname):
    genome_list = []
    for root, dirs, files in os.walk(dirname):
        for genome in files:
            genome_list.append(RecurrentNetwork.load(Path(root) / genome))
    return genome_list

def run_simulation():
    supervisor = SimulationManager()
    # Teleport redEnd and greenStart at the start of each generation
    supervisor.randomize_goals()
    # Get the parent directory dynamically
    current_directory = os.path.dirname(__file__)  # Directory of the current script
    grandpa_directory = os.path.abspath(os.path.join(current_directory, "..", ".."))  # Navigate to parent directory
    # Construct the path to the genome data file
    genome_data_path = os.path.join(grandpa_directory, "genome_data/" f"genome_data{supervisor.port}")

    # Load all the genome data
    batch = loadall(genome_data_path)
    for genome in batch:
        supervisor.reset_simulation_state()
        fitness = supervisor.evaluate_genome(genome)
        genome.fitness = fitness
        print(genome.fitness)

if __name__ == '__main__':
    run_simulation()
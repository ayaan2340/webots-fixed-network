import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
from simulation_manager import SimulationManager
from RecurrentNetwork import RecurrentNetwork


def render_frame(draw, simulation, width=800, height=800, scale=4):
    """Render a single frame of the simulation using PIL"""
    # Clear background to white
    draw.rectangle([0, 0, width, height], fill='white')

    # Draw all roads
    for road in simulation.all_roads:
        # Convert simulation coordinates to image coordinates
        x = road.position[0] * scale + width // 2
        y = road.position[1] * scale + height // 2
        w = road.size[0] * scale
        h = road.size[1] * scale

        # Draw road as rectangle
        draw.rectangle([
            x - w // 2, y - h // 2,
            x + w // 2, y + h // 2
        ], fill='darkgray')

    # Draw start point
    start_x = simulation.start.position[0] * scale + width // 2
    start_y = simulation.start.position[1] * scale + height // 2
    draw.ellipse([start_x - 5, start_y - 5, start_x + 5, start_y + 5], fill='green')

    # Draw end point
    end_x = simulation.end.position[0] * scale + width // 2
    end_y = simulation.end.position[1] * scale + height // 2
    draw.ellipse([end_x - 5, end_y - 5, end_x + 5, end_y + 5], fill='blue')

    # Draw car as triangle
    car_x = simulation.car.position[0] * scale + width // 2
    car_y = simulation.car.position[1] * scale + height // 2

    # Create triangle points for car
    angle = simulation.car.rotation
    car_size = 10
    points = [
        (car_x + car_size * np.sin(angle),
         car_y - car_size * np.cos(angle)),
        (car_x + car_size * np.sin(angle + 2.6),
         car_y - car_size * np.cos(angle + 2.6)),
        (car_x + car_size * np.sin(angle - 2.6),
         car_y - car_size * np.cos(angle - 2.6))
    ]
    draw.polygon(points, fill='red')


def create_simulation_video(genome_path, output_path="simulation.mp4", fps=30):
    """Create a video of the simulation running"""
    # Initialize simulation
    manager = SimulationManager()
    genome = RecurrentNetwork.load(
        genome_path,
        input_size=5,
        hidden_size=10,
        output_size=2,
        genome_id=0,
    )
    manager.reset_simulation_state()
    genome.hidden_state = None

    # Initialize video writer
    width, height = 800, 800
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("Starting video creation...")
    frame_count = 0

    while manager.simulation.time < manager.max_simulation_time:
        # Create new image for each frame
        image = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(image)

        # Render frame
        render_frame(draw, manager.simulation, width, height)

        # Convert PIL image to OpenCV format
        opencv_frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        video.write(opencv_frame)
        frame_count += 1

        # Run simulation step
        inputs = manager.simulation.get_inputs()
        outputs = genome.forward(inputs)
        manager.simulation.set_controls(outputs)

        if not manager.simulation.step():  # Car went off road
            print("Car went off road")
            break

        if manager.simulation.reached_goal():
            print("Reached goal!")
            # Add a few extra frames at the end
            for _ in range(fps):
                video.write(opencv_frame)
            break

    # Release video writer
    video.release()
    print(f"Created video with {frame_count} frames at {output_path}")


if __name__ == "__main__":
    # Get the path to the best genome
    current_dir = os.path.dirname(os.path.abspath(__file__))
    genome_path = os.path.join(current_dir, "best_genomes", "best_genome_gen_1.pt")

    if not os.path.exists(genome_path):
        raise FileNotFoundError(f"Could not find genome file at {genome_path}")

    # Create video
    create_simulation_video(genome_path)

from simulation import Simulation
import numpy as np
import time


def test_visualization():
    # Create simulation with visualization enabled
    sim = Simulation(visualize=True)
    sim.reset()

    # Place car in the middle of an intersection (60, 60)
    sim.car.position = (60, 60)
    sim.car.rotation = 0.0

    try:
        # Spin the car around slowly
        while True:
            # Set minimal forward speed and constant turning
            sim.set_controls([0.1, 0.5])  # 10% of max speed, max steering angle

            if not sim.step():
                print("Off road!")
                break

            # Add a small delay to make the visualization easier to follow
            time.sleep(0.01)

            # Check for goal (we won't reach it in this test)
            if sim.reached_goal():
                print("Reached goal!")
                break

    except KeyboardInterrupt:
        print("\nTest stopped by user")
    finally:
        sim.close()


if __name__ == "__main__":
    test_visualization()

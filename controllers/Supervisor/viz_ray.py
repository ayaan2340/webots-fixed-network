import numpy as np
import matplotlib.pyplot as plt
from simulation import Simulation

# Create simulation instance
sim = Simulation()

# Choose an intersection (using the second one at (60, 60))
intersection_pos = (60, 60)

# Set up test parameters
num_angles = 360  # Test every degree
angles = np.linspace(0, 2 * np.pi, num_angles)
ray_distances = []

# Collect ray distances at each angle
for angle in angles:
    # Position car at intersection
    sim.car.position = intersection_pos
    sim.car.rotation = angle

    # Get ray distance
    distance = sim.calculate_ray_distance()
    ray_distances.append(distance)

# Convert to numpy array
ray_distances = np.array(ray_distances)

# Create polar plot
plt.figure(figsize=(10, 10))
ax = plt.subplot(111, projection='polar')

# Plot the data
ax.plot(angles, ray_distances)

# Customize the plot
ax.set_title('Ray Distances at Different Angles\nIntersection Position: (60, 60)', pad=20)
ax.set_rticks([5, 10, 15, 20, 25, 30])
ax.set_rlabel_position(45)
ax.grid(True)

# Add labels
plt.text(0, -5, 'Distance (units)', rotation=90, ha='center', va='bottom')

# Save the plot
plt.savefig('out.png', bbox_inches='tight', dpi=300)
plt.close()

# Print some statistics
print(f"Maximum ray distance: {np.max(ray_distances):.2f}")
print(f"Minimum ray distance: {np.min(ray_distances):.2f}")
print(f"Average ray distance: {np.mean(ray_distances):.2f}")

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

print(Path.cwd())

left_PCs = np.load('data/PCA_data/left_PCs.npy')
stop_left_PCs = np.load('data/PCA_data/stop_left_PCs.npy')
right_PCs = np.load('data/PCA_data/right_PCs.npy')
stop_right_PCs = np.load('data/PCA_data/stop_right_PCs.npy')

print(stop_left_PCs.shape, left_PCs.shape)

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot left trajectory (180°)
ax.plot(
    left_PCs[0, :], left_PCs[1, :], left_PCs[2, :], 
    color='blue', linewidth=2, label='Left (180°)'
)

ax.scatter(
    left_PCs[0, 0], left_PCs[1, 0], left_PCs[2, 0], 
    marker='^', s=200, color='blue', edgecolors='black', linewidths=2, 
    label='Left Start', zorder=5
)

# Plot right trajectory (0°)
ax.plot(
    right_PCs[0, :], right_PCs[1, :], right_PCs[2, :], 
    color='green', linewidth=2, label='Right (0°)'
)

ax.scatter(
    right_PCs[0, 0], right_PCs[1, 0], right_PCs[2, 0], 
    marker='^', s=200, color='green', edgecolors='black', linewidths=2, 
    label='Right Start', zorder=5
)

# Plot left STOP trajectory (180°)
ax.plot(
    stop_left_PCs[0, :], stop_left_PCs[1, :], stop_left_PCs[2, :], 
    color='red', linewidth=2, label='STOP Left (180°)', linestyle='dashed'
)

ax.scatter(
    stop_left_PCs[0, 0], stop_left_PCs[1, 0], stop_left_PCs[2, 0], 
    marker='*', s=200, color='red', edgecolors='black', linewidths=2, 
    label='Stop Left Start', zorder=5
)


ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('3D Trajectory Plot')
ax.legend()
plt.show()

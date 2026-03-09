"""
Plot multi-session PCA results in 3D.

Loads PCA results from data/PCA_data/multi_session/ and creates
3D trajectory visualizations for GO and STOP trials.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

# Set path relative to repository root
repo_root = Path(__file__).parent.parent
pca_data_dir = repo_root / 'data' / 'PCA_data' / 'multi_session'

print(f"Loading multi-session PCA results from: {pca_data_dir}")

# Load PC trajectories
go_left_PCs = np.load(pca_data_dir / 'go_left_PCs.npy')
go_right_PCs = np.load(pca_data_dir / 'go_right_PCs.npy')
stop_left_PCs = np.load(pca_data_dir / 'stop_left_PCs.npy')
stop_right_PCs = np.load(pca_data_dir / 'stop_right_PCs.npy')
time = np.load(pca_data_dir / 'time.npy')

# Load metadata
with open(pca_data_dir / 'metadata.json', 'r') as f:
    metadata = json.load(f)

print(f"\nDataset info:")
print(f"  Sessions: {metadata['n_sessions']}")
print(f"  Total cells: {metadata['n_cells_total']}")
print(f"  Epoch: {metadata['epok']} ms")
print(f"  PC variance: {[f'{v*100:.1f}%' for v in metadata['explained_variance_ratio'][:3]]}")
print(f"  Cumulative (PC1-3): {metadata['cumulative_variance'][2]*100:.1f}%")

print(f"\nPC trajectory shapes:")
print(f"  GO Left: {go_left_PCs.shape}")
print(f"  GO Right: {go_right_PCs.shape}")
print(f"  STOP Left: {stop_left_PCs.shape}")
print(f"  STOP Right: {stop_right_PCs.shape}")

# Create 3D plot
fig = plt.figure(figsize=(14, 11))
ax = fig.add_subplot(111, projection='3d')

# Plot GO trajectories
ax.plot(
    go_left_PCs[0, :], go_left_PCs[1, :], go_left_PCs[2, :],
    color='blue', linewidth=2.5, label='GO Left (180°)', alpha=0.9
)

ax.plot(
    go_right_PCs[0, :], go_right_PCs[1, :], go_right_PCs[2, :],
    color='green', linewidth=2.5, label='GO Right (0°)', alpha=0.9
)

# Plot STOP trajectories
ax.plot(
    stop_left_PCs[0, :], stop_left_PCs[1, :], stop_left_PCs[2, :],
    color='red', linewidth=2.5, label='STOP Left (180°)',
    linestyle='dashed', alpha=0.9
)

ax.plot(
    stop_right_PCs[0, :], stop_right_PCs[1, :], stop_right_PCs[2, :],
    color='orange', linewidth=2.5, label='STOP Right (0°)',
    linestyle='dashed', alpha=0.9
)

# Mark start points
ax.scatter(
    go_left_PCs[0, 0], go_left_PCs[1, 0], go_left_PCs[2, 0],
    marker='^', s=200, color='blue', edgecolors='black', linewidths=2,
    label='GO Start', zorder=5
)

ax.scatter(
    go_right_PCs[0, 0], go_right_PCs[1, 0], go_right_PCs[2, 0],
    marker='^', s=200, color='green', edgecolors='black', linewidths=2,
    zorder=5
)

ax.scatter(
    stop_left_PCs[0, 0], stop_left_PCs[1, 0], stop_left_PCs[2, 0],
    marker='*', s=250, color='red', edgecolors='black', linewidths=2,
    label='STOP Start', zorder=5
)

ax.scatter(
    stop_right_PCs[0, 0], stop_right_PCs[1, 0], stop_right_PCs[2, 0],
    marker='*', s=250, color='orange', edgecolors='black', linewidths=2,
    zorder=5
)

# Axis labels with variance explained
var1, var2, var3 = metadata['explained_variance_ratio'][:3]
ax.set_xlabel(f'PC1 ({var1*100:.1f}%)', fontsize=14, fontweight='bold')
ax.set_ylabel(f'PC2 ({var2*100:.1f}%)', fontsize=14, fontweight='bold')
ax.set_zlabel(f'PC3 ({var3*100:.1f}%)', fontsize=14, fontweight='bold')

# Title with dataset info
cumvar = metadata['cumulative_variance'][2] * 100
title = (f'Multi-Session PCA Trajectories\n'
         f'{metadata["n_sessions"]} sessions, {metadata["n_cells_total"]} cells | '
         f'PC1-3: {cumvar:.1f}% variance | '
         f'Epoch: {metadata["epok"][0]} to {metadata["epok"][1]} ms')
ax.set_title(title, fontsize=15, fontweight='bold', pad=20)

# Legend
ax.legend(fontsize=11, loc='upper left', framealpha=0.9)

# Grid
ax.grid(True, alpha=0.3)

# Adjust viewing angle for better visualization
# ax.view_init(elev=20, azim=45)

plt.tight_layout()

# Save figure
output_path = pca_data_dir / 'trajectory_3d_script.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved figure to: {output_path}")

plt.show()

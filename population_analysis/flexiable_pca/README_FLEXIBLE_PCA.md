# Flexible PCA Analysis System

A modular PCA framework for neural population analysis that allows:
1. **Fitting PCA on specific trial conditions and epochs**
2. **Projecting arbitrary conditions onto the fitted principal components**
3. **Comparing how different conditions use the same neural space**

---

## Quick Start

### 1. Run the quick-start script

```bash
cd population_analysis/xie_style
../../.conda/bin/python flexible_pca_quickstart.py
```

This will:
- Fit PCA on GO trials (both directions)
- Project STOP and CONT trials onto the GO-defined PC space
- Generate 3D and time series visualizations
- Save results to `data/flexible_pca_results/`

### 2. Or use Jupyter notebook

```bash
jupyter notebook flexible_pca_demo.ipynb
```

The demo notebook includes multiple examples:
- Example 1: Fit on GO, project STOP/CONT
- Example 2: Fit on early epoch, project later epochs
- Example 3: Using standard 6-condition specs

---

## Core Concepts

### TrialSpec: Specifying Trial Conditions

`TrialSpec` defines a single trial condition with all necessary parameters:

```python
from flexible_pca import TrialSpec

# Single direction
spec = TrialSpec(
    trial_type='GO',        # 'GO', 'STOP', or 'CONT'
    direction=0,            # 0 (right), 180 (left), or None (both)
    epoch=[-50, 300],       # Time window in ms
    alignment='go_cue',     # Alignment event
    ssd_number=None,        # SSD for STOP/CONT (ignored for GO)
    label='GO_R'            # Optional custom label
)

# Both directions (will create separate conditions internally)
spec_both = TrialSpec(
    trial_type='STOP',
    direction=None,         # Will process both 0 and 180
    epoch=[-50, 300],
    alignment='stop_cue',
    ssd_number=2
)
```

### FlexiblePCA: The Main Class

```python
from flexible_pca import FlexiblePCA
from sklearn.decomposition import TruncatedSVD

# Initialize
fpca = FlexiblePCA(
    cell_df,                      # Your cell trial database
    bin_size=1,                   # PSTH bin size (ms)
    smooth_ker_size=25,           # Smoothing kernel (ms)
    success_only=True,            # Use only successful trials
    n_components=5,               # Number of PCs
    pca_function=TruncatedSVD,    # PCA or TruncatedSVD
    random_state=42,
    verbose=True
)
```

---

## Usage Examples

### Example 1: Fit on GO trials, project STOP/CONT

**Use case:** See how STOP and CONT signals affect the neural space defined by GO trials.

```python
# Fit PCA on GO trials only
fit_specs = [
    TrialSpec('GO', direction=0, epoch=[-50, 300], alignment='go_cue'),
    TrialSpec('GO', direction=180, epoch=[-50, 300], alignment='go_cue'),
]

fpca.fit(fit_specs)

# Get GO trajectories in PC space
go_trajectories = fpca.get_fit_trajectories()
# Returns: {'GO_R': array(5, 350), 'GO_L': array(5, 350)}

# Project STOP and CONT trials onto GO-defined PCs
proj_specs = [
    TrialSpec('STOP', direction=0, epoch=[-50, 300], alignment='go_cue', ssd_number=2),
    TrialSpec('STOP', direction=180, epoch=[-50, 300], alignment='go_cue', ssd_number=2),
    TrialSpec('CONT', direction=0, epoch=[-50, 300], alignment='go_cue', ssd_number=2),
    TrialSpec('CONT', direction=180, epoch=[-50, 300], alignment='go_cue', ssd_number=2),
]

projections = fpca.project(proj_specs)
# Returns: {'STOP_R': array(5, 350), 'STOP_L': array(5, 350), ...}
```

### Example 2: Fit on early epoch, project later epochs

**Use case:** Understand how neural activity evolves in a PC space defined by an early time window.

```python
# Fit on early epoch (initialization period)
early_specs = [
    TrialSpec('GO', direction=0, epoch=[-50, 100], alignment='go_cue'),
    TrialSpec('GO', direction=180, epoch=[-50, 100], alignment='go_cue'),
]

fpca.fit(early_specs)

# Project later epoch (execution period)
late_specs = [
    TrialSpec('GO', direction=0, epoch=[100, 300], alignment='go_cue'),
    TrialSpec('GO', direction=180, epoch=[100, 300], alignment='go_cue'),
]

late_projections = fpca.project(late_specs)
```

### Example 3: Compare different alignments

**Use case:** See how the same trials look when aligned to different events.

```python
# Fit on go_cue aligned data
go_aligned_specs = [
    TrialSpec('STOP', direction=0, epoch=[-200, 200], alignment='go_cue', ssd_number=2),
]

fpca.fit(go_aligned_specs)

# Project same trials aligned to stop_cue
stop_aligned_specs = [
    TrialSpec('STOP', direction=0, epoch=[-200, 200], alignment='stop_cue', ssd_number=2),
]

stop_projections = fpca.project(stop_aligned_specs)
```

### Example 4: Standard 6-condition analysis

**Use case:** Recreate the condition-concatenated PCA approach.

```python
from flexible_pca import create_standard_specs

# Create all 6 conditions (GO/STOP/CONT × Left/Right)
all_specs = create_standard_specs(
    go_epok=[-50, 300],
    stop_epok=[-50, 300],
    cont_epok=[-50, 300],
    go_align='go_cue',
    stop_cont_align='go_cue',
    ssd_number=2,
    include_both_dirs=True  # Creates 6 conditions
)

fpca.fit(all_specs)
trajectories = fpca.get_fit_trajectories()
# Returns all 6 trajectories: GO_R, GO_L, STOP_R, STOP_L, CONT_R, CONT_L
```

---

## Visualization

### 3D Trajectory Plot

```python
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Get trajectories
fit_traj = fpca.get_fit_trajectories()
proj_traj = fpca.project(proj_specs)

# Combine
all_traj = {**fit_traj, **proj_traj}

# Plot
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')

colors = {'GO_R': 'cyan', 'GO_L': 'blue', 'STOP_R': 'orange',
          'STOP_L': 'red', 'CONT_R': 'lime', 'CONT_L': 'green'}

for label, traj in all_traj.items():
    color = colors.get(label, 'gray')
    ax.plot(traj[0, :], traj[1, :], traj[2, :],
            label=label, color=color, linewidth=2, alpha=0.7)

var_ratios = fpca.pca_model.explained_variance_ratio_
ax.set_xlabel(f'PC1 ({var_ratios[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({var_ratios[1]*100:.1f}%)')
ax.set_zlabel(f'PC3 ({var_ratios[2]*100:.1f}%)')
ax.legend()
plt.show()
```

### Time Series Plot

```python
# Get time axis
time_axis = fpca.get_time_axis(fit_specs[0])

# Plot
fig, axes = plt.subplots(3, 1, figsize=(16, 12))

for i in range(3):  # First 3 PCs
    ax = axes[i]

    for label, traj in all_traj.items():
        color = colors.get(label, 'gray')
        ax.plot(time_axis, traj[i, :], label=label, color=color, linewidth=2)

    ax.set_ylabel(f'PC{i+1}')
    ax.axvline(0, color='black', linestyle=':', alpha=0.5)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
    ax.legend()

axes[-1].set_xlabel('Time (ms)')
plt.tight_layout()
plt.show()
```

---

## Key Methods

### FlexiblePCA Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `fit(trial_specs)` | Fit PCA on specified conditions | self (for chaining) |
| `project(trial_specs)` | Project conditions onto fitted PCs | dict of {label: trajectory} |
| `get_fit_trajectories()` | Get trajectories for fit conditions | dict of {label: trajectory} |
| `get_time_axis(spec)` | Get time axis for a TrialSpec | ndarray of time values |

### Helper Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `create_standard_specs(...)` | Create standard 6 conditions | list of TrialSpec |

---

## Data Shapes

Understanding the data flow:

```
1. Raw PSTHs per condition:
   Shape: (n_neurons, n_time_bins)
   Example: (1202, 350) for 1202 neurons, 350ms epoch with 1ms bins

2. Concatenated PSTHs for fitting:
   Shape: (n_neurons, n_total_bins)
   Example: (1202, 700) for 2 conditions × 350 bins each

3. Normalized data:
   Shape: (n_neurons, n_total_bins)
   Z-scored across all bins for each neuron

4. PCA components:
   Shape: (n_components, n_total_bins)
   Example: (5, 700) for 5 PCs

5. Individual trajectories:
   Shape: (n_components, n_time_bins)
   Example: (5, 350) for 5 PCs, 350 time bins
```

---

## Implementation Details

### Normalization

Each neuron is z-scored across **all bins in the fit data**:
```python
for each neuron:
    mean = mean(all_bins_in_fit_conditions)
    std = std(all_bins_in_fit_conditions)
    normalized = (data - mean) / std
```

**Important:** Projection uses the **same normalization stats** from fitting, ensuring consistency.

### PCA Fitting

PCA is fitted on the **transposed** normalized data:
```python
# Data: (n_neurons, n_features)
# Transpose to: (n_features, n_neurons)
# This treats each time point as a sample, each neuron as a feature
pca_model.fit(X_normalized.T)
```

### Projection

Projections are computed as:
```python
# For each condition:
# 1. Extract PSTHs: (n_neurons, n_time_bins)
# 2. Normalize using fit stats
# 3. Project: PCA_components @ normalized_data
# Result: (n_components, n_time_bins)
```

---

## Comparison with condition_concatenated_pca.ipynb

| Feature | condition_concatenated_pca | FlexiblePCA |
|---------|---------------------------|-------------|
| **Conditions** | Fixed 6 conditions | Any conditions |
| **Epochs** | Fixed epochs | Any epochs |
| **Alignment** | Fixed alignment | Any alignment |
| **Projection** | Only fit data | Fit + project anything |
| **Reusability** | One analysis | Reuse fit for multiple projections |
| **Code style** | Notebook cells | Modular class |

**When to use which:**
- Use `condition_concatenated_pca.ipynb` for: Standard 6-condition analysis
- Use `FlexiblePCA` for: Exploring specific hypotheses, comparing conditions, temporal dynamics

---

## File Structure

```
population_analysis/xie_style/
├── flexible_pca.py                    # Main module
├── flexible_pca_demo.ipynb           # Demo notebook (3 examples)
├── flexible_pca_quickstart.py        # Quick start script
├── README_FLEXIBLE_PCA.md            # This file
└── condition_concatenated_pca.ipynb  # Original implementation
```

---

## Advanced Usage

### Custom preprocessing

You can access raw data before PCA:

```python
# Fit PCA
fpca.fit(fit_specs)

# Access raw and normalized data
X_raw = fpca.X_fit_raw              # (n_neurons, n_features)
X_normalized = fpca.X_fit_normalized  # (n_neurons, n_features)

# Access normalization stats
means = fpca.normalization_stats['mean']  # List of neuron means
stds = fpca.normalization_stats['std']    # List of neuron stds

# Access PCA model directly
components = fpca.pca_model.components_           # (n_components, n_features)
variance_ratio = fpca.pca_model.explained_variance_ratio_
```

### Multiple projections

Reuse fitted PCA for multiple projections:

```python
# Fit once
fpca.fit(fit_specs)

# Project multiple different conditions
proj1 = fpca.project(specs_set1)
proj2 = fpca.project(specs_set2)
proj3 = fpca.project(specs_set3)
```

### Extract specific neurons

```python
# Filter to specific cells before creating FlexiblePCA
cell_subset = cell_df[cell_df['cell_ID'].isin(selected_cell_ids)]

fpca = FlexiblePCA(cell_subset, ...)
```

---

## Tips and Best Practices

1. **Start simple:** Begin with `flexible_pca_quickstart.py` to understand the basics

2. **Consistent epochs:** When comparing conditions, use the same epoch length for fair comparison

3. **Alignment matters:** Think carefully about alignment points:
   - Use `go_cue` to compare trial initiation
   - Use `stop_cue` to focus on signal processing

4. **Check trial counts:** The verbose output shows trial counts - ensure sufficient data

5. **Normalization:** All conditions (fit + project) use the same normalization from fit data

6. **Variance explained:** Check explained variance ratios to ensure PCs capture meaningful structure

7. **Missing data:** Neurons without data in all fit conditions are excluded

---

## Troubleshooting

**Q: "Must call fit() before project()"**
A: You need to call `fpca.fit(specs)` before projecting.

**Q: Different epoch lengths in fit specs**
A: This is allowed but creates concatenated vectors of different lengths per condition. Ensure this is intentional.

**Q: Low variance explained**
A: Try increasing `n_components` or check if your conditions have sufficient structure.

**Q: Missing neurons in projection**
A: Projection uses the same neurons as fit. If a neuron had no data during fit, it's excluded from projection.

**Q: Projection looks different from fit trajectories**
A: This is expected! Projection uses the same PC space but different time windows/conditions.

---

## Contact

For questions or issues:
- Check `flexible_pca_demo.ipynb` for examples
- Review `condition_concatenated_pca.ipynb` for original implementation
- Consult project documentation in `docs/`

---

**Author:** Claude & Barak
**Date:** December 2024
**Version:** 1.0

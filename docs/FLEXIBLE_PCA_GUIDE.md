# FlexiblePCA Guide

## Overview

**FlexiblePCA** is an optimized PCA analysis tool that allows:
1. **Flexible condition selection**: Fit PCA on arbitrary trial conditions and epochs
2. **Flexible projection**: Project different conditions onto fitted principal components
3. **High performance**: Parallel processing and vectorized operations for fast computation
4. **Flexible normalization**: Choose between z-scoring or centering only

**Location**: `population_analysis/flexiable_pca/flexible_pca.py`

---

## Key Features

### 1. **Arbitrary Condition Fitting**
Unlike traditional approaches that fit on all conditions together, FlexiblePCA lets you:
- Fit on GO trials only, then project STOP/CONT
- Fit on early epochs, project later epochs
- Fit on one direction, project the other
- Any combination you can imagine

### 2. **Performance Optimizations**
- **Parallel processing**: Use `n_jobs=-1` to utilize all CPU cores (~5-7x speedup)
- **Vectorized normalization**: No Python for-loops, pure numpy operations
- **Smart data handling**: Pre-filters data to minimize serialization overhead

### 3. **Flexible Normalization**
- **z_score=True** (default): Full z-scoring (subtract mean, divide by std)
- **z_score=False**: Centering only (subtract mean, preserve variance)

---

## Quick Start

### Basic Usage

```python
from flexible_pca import FlexiblePCA, TrialSpec, create_standard_specs
import pandas as pd

# Load data
cell_df = pd.read_pickle('data/unified_cell_trial_data/msn_fiona_cell_trial_data.pkl')

# Create FlexiblePCA instance
fpca = FlexiblePCA(
    cell_df,
    bin_size=1,
    smooth_ker_size=25,
    n_components=5,
    pca_function=TruncatedSVD,
    n_jobs=-1,        # Use all CPU cores
    z_score=False,    # Only center, don't z-score
    verbose=True
)

# Define conditions for fitting
fit_specs = [
    TrialSpec('GO', direction=0, epoch=[-50, 300], alignment='go_cue'),
    TrialSpec('GO', direction=180, epoch=[-50, 300], alignment='go_cue'),
]

# Fit PCA
fpca.fit(fit_specs)

# Project other conditions
proj_specs = [
    TrialSpec('STOP', direction=0, epoch=[-50, 300], alignment='go_cue', ssd_number=2),
    TrialSpec('STOP', direction=180, epoch=[-50, 300], alignment='go_cue', ssd_number=2),
    TrialSpec('CONT', direction=0, epoch=[-50, 300], alignment='go_cue', ssd_number=2),
    TrialSpec('CONT', direction=180, epoch=[-50, 300], alignment='go_cue', ssd_number=2),
]

projections = fpca.project(proj_specs)

# Get fit trajectories
fit_trajectories = fpca.get_fit_trajectories()
```

---

## TrialSpec Class

**Purpose**: Specify a trial condition for PCA fitting or projection

### Parameters

```python
TrialSpec(
    trial_type: str,              # 'GO', 'STOP', or 'CONT'
    direction: int or None,       # 0 (right), 180 (left), or None (both)
    epoch: List[int],             # [start_ms, end_ms]
    alignment: str,               # 'go_cue', 'stop_cue', etc.
    ssd_number: int or None,      # SSD index for STOP/CONT (ignored for GO)
    label: str or None            # Custom label (auto-generated if None)
)
```

### Examples

```python
# GO trial, right direction, aligned to go_cue
TrialSpec('GO', direction=0, epoch=[-50, 300], alignment='go_cue')

# STOP trial with specific SSD
TrialSpec('STOP', direction=180, epoch=[-50, 300], alignment='stop_cue', ssd_number=2)

# Early epoch with custom label
TrialSpec('GO', direction=0, epoch=[-150, 10], alignment='go_cue', label='GO_R_early')

# Both directions combined
TrialSpec('GO', direction=None, epoch=[-50, 300], alignment='go_cue')
```

---

## FlexiblePCA Class

### Initialization Parameters

```python
FlexiblePCA(
    cell_df,                      # DataFrame with cell trial data
    bin_size=1,                   # Bin size for PSTHs (ms)
    smooth_ker_size=25,           # Smoothing kernel size (ms)
    success_only=True,            # Use only successful trials
    n_components=5,               # Number of principal components
    pca_function=TruncatedSVD,    # PCA or TruncatedSVD
    random_state=42,              # Random seed
    verbose=True,                 # Print progress
    n_jobs=-1,                    # Parallel workers (-1 = all cores)
    z_score=False                 # True: z-score, False: center only
)
```

### Methods

#### `fit(trial_specs)`
Fit PCA on specified conditions

**Parameters**:
- `trial_specs`: List of TrialSpec objects defining conditions to fit

**Returns**: `self` (for method chaining)

**Example**:
```python
fit_specs = [
    TrialSpec('GO', direction=0, epoch=[-50, 300], alignment='go_cue'),
    TrialSpec('GO', direction=180, epoch=[-50, 300], alignment='go_cue'),
]
fpca.fit(fit_specs)
```

#### `project(trial_specs)`
Project specified conditions onto fitted PCs

**Parameters**:
- `trial_specs`: List of TrialSpec objects defining conditions to project

**Returns**: Dictionary mapping condition labels to PC projections
- Each projection: shape `(n_components, n_time_bins)`

**Example**:
```python
proj_specs = [
    TrialSpec('STOP', direction=0, epoch=[-50, 300], alignment='go_cue', ssd_number=2),
]
projections = fpca.project(proj_specs)
# projections['STOP_R'] -> shape (5, 350)
```

#### `get_fit_trajectories()`
Get PC trajectories for conditions used in fitting

**Returns**: Dictionary mapping condition labels to PC trajectories
- Each trajectory: shape `(n_components, n_time_bins)`

**Example**:
```python
fit_traj = fpca.get_fit_trajectories()
# fit_traj['GO_R'] -> shape (5, 350)
```

#### `get_time_axis(spec)`
Get time axis for a trial specification

**Parameters**:
- `spec`: TrialSpec object

**Returns**: numpy array of time values (ms)

**Example**:
```python
spec = TrialSpec('GO', direction=0, epoch=[-50, 300], alignment='go_cue')
time_axis = fpca.get_time_axis(spec)
# array([-50, -49, -48, ..., 297, 298, 299])
```

---

## Helper Functions

### `create_standard_specs()`

Creates standard set of trial specifications (GO, STOP, CONT × 2 directions)

```python
create_standard_specs(
    go_epok=[-50, 300],
    stop_epok=[-50, 300],
    cont_epok=[-50, 300],
    go_align='go_cue',
    stop_cont_align='go_cue',
    ssd_number=2,
    include_both_dirs=True
)
```

**Returns**: List of 6 TrialSpec objects (3 types × 2 directions)

**Example**:
```python
specs = create_standard_specs()
# Returns:
# [GO_R, STOP_R, CONT_R, GO_L, STOP_L, CONT_L]
```

---

## Common Use Cases

### 1. Fit on GO, Project STOP/CONT

**Goal**: See how STOP/CONT signals affect neural space defined by GO trials

```python
# Fit on GO trials
fit_specs = [
    TrialSpec('GO', direction=0, epoch=[-50, 300], alignment='go_cue'),
    TrialSpec('GO', direction=180, epoch=[-50, 300], alignment='go_cue'),
]
fpca.fit(fit_specs)

# Project STOP and CONT
proj_specs = [
    TrialSpec('STOP', direction=0, epoch=[-50, 300], alignment='go_cue', ssd_number=2),
    TrialSpec('STOP', direction=180, epoch=[-50, 300], alignment='go_cue', ssd_number=2),
    TrialSpec('CONT', direction=0, epoch=[-50, 300], alignment='go_cue', ssd_number=2),
    TrialSpec('CONT', direction=180, epoch=[-50, 300], alignment='go_cue', ssd_number=2),
]
projections = fpca.project(proj_specs)
```

### 2. Fit on Early Epoch, Project Later

**Goal**: See how neural activity evolves in PC space defined by early time window

```python
# Fit on early epoch
early_specs = [
    TrialSpec('GO', direction=0, epoch=[-150, 10], alignment='go_cue', label='GO_R_early'),
    TrialSpec('GO', direction=180, epoch=[-150, 10], alignment='go_cue', label='GO_L_early'),
]
fpca.fit(early_specs)

# Project later epoch
late_specs = [
    TrialSpec('GO', direction=0, epoch=[100, 300], alignment='go_cue', label='GO_R_late'),
    TrialSpec('GO', direction=180, epoch=[100, 300], alignment='go_cue', label='GO_L_late'),
]
late_projections = fpca.project(late_specs)
```

### 3. Standard 6-Condition PCA

**Goal**: Traditional PCA on all conditions

```python
specs = create_standard_specs(
    go_epok=[-50, 300],
    stop_epok=[-50, 300],
    cont_epok=[-50, 300],
    go_align='go_cue',
    stop_cont_align='go_cue',
    ssd_number=2
)
fpca.fit(specs)
trajectories = fpca.get_fit_trajectories()
```

---

## Performance Optimization

### Parallel Processing

**Recommended**: Use `n_jobs=-1` to utilize all CPU cores

```python
fpca = FlexiblePCA(cell_df, n_jobs=-1)  # ~5-7x speedup
```

**When it helps**:
- Large number of cells (>500)
- Multiple conditions
- Long epochs

**Benchmark results** (1213 neurons, 4 conditions):
- Sequential (n_jobs=1): ~28-36 seconds
- Parallel (n_jobs=-1, 20 cores): ~4-6 seconds
- **Speedup**: 5-7x

### Normalization Options

**z_score=True** (default):
- Full z-scoring: (X - mean) / std
- Each neuron scaled to mean=0, std=1
- Recommended for most analyses

**z_score=False**:
- Centering only: X - mean
- Preserves variance differences between neurons
- Useful when relative firing rates matter

Both options use **vectorized numpy operations** for speed.

---

## Data Requirements

### Input DataFrame

Same format as used by Session/MultiSessionPCA:

**Required columns**:
- `cell_ID`: Unique neuron identifier
- `trial_type`: 'GO', 'STOP', or 'CONT'
- `dir`: Direction (0 or 180)
- `trial_failed`: Boolean
- `trial_session`: Session identifier
- Trial event timestamps (for alignment)
- Spike times

**Filtering**:
```python
# Exclude poor quality sessions
excluded_sessions = ['fi210628', 'fi210629', 'fi210704']
cell_df = cell_df[~cell_df['trial_session'].isin(excluded_sessions)]

# Success only
cell_df = cell_df[cell_df['trial_failed'] == False]
```

---

## Visualization

### 3D Trajectory Plot

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Get trajectories
trajectories = fpca.get_fit_trajectories()
var_ratios = fpca.pca_model.explained_variance_ratio_

# Plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

for label, traj in trajectories.items():
    ax.plot(traj[0, :], traj[1, :], traj[2, :], label=label, linewidth=2)

ax.set_xlabel(f'PC1 ({var_ratios[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({var_ratios[1]*100:.1f}%)')
ax.set_zlabel(f'PC3 ({var_ratios[2]*100:.1f}%)')
ax.legend()
plt.show()
```

### Time Series Plot

```python
fig, axes = plt.subplots(3, 1, figsize=(14, 10))
time_axis = fpca.get_time_axis(fit_specs[0])

for i in range(3):
    for label, traj in trajectories.items():
        axes[i].plot(time_axis, traj[i, :], label=label)

    axes[i].set_ylabel(f'PC{i+1} ({var_ratios[i]*100:.1f}%)')
    axes[i].axvline(0, color='black', linestyle='--', alpha=0.3)
    axes[i].legend()

axes[2].set_xlabel('Time from go_cue (ms)')
plt.tight_layout()
plt.show()
```

---

## Comparison with Session-based PCA

| Feature | FlexiblePCA | Session/MultiSessionPCA |
|---------|-------------|------------------------|
| **Condition Selection** | Arbitrary combinations | All conditions together |
| **Parallel Processing** | ✅ Both fit() and project() | ✅ PSTH extraction only |
| **Normalization** | Vectorized, configurable | For-loop based |
| **Flexibility** | High - fit/project separately | Medium - fit on all |
| **Speed** | Very fast (~5-7x) | Fast |
| **Use Case** | Exploratory, hypothesis-driven | Standard population analysis |

---

## Example Notebook

See `population_analysis/flexiable_pca/flexible_pca_demo.ipynb` for:
- Complete examples of all use cases
- 3D and time series visualizations
- Performance benchmarks
- Step-by-step walkthrough

---

## Tips and Best Practices

1. **Start with small n_jobs**: Test with `n_jobs=4` before using `-1` to verify no issues
2. **Filter data first**: Remove poor quality sessions before creating FlexiblePCA
3. **Choose epochs carefully**: Shorter epochs = faster computation
4. **Use create_standard_specs()**: For traditional 6-condition PCA
5. **Check variance explained**: Ensure top PCs capture sufficient variance
6. **Custom labels**: Use meaningful labels for complex analyses

---

## Troubleshooting

### "Not enough data" warnings
- Some neurons may not have trials for all conditions
- FlexiblePCA automatically excludes neurons with missing conditions
- Check trial counts in fit output

### Slow performance despite parallel processing
- Verify `n_jobs` is set correctly
- Check if system has enough RAM (workers need memory)
- Try smaller `n_jobs` value

### Different results from traditional PCA
- Check normalization setting (`z_score`)
- Verify epoch alignment and timing
- Compare trial counts between methods

---

**Last Updated**: December 2024
**Author**: Claude & Barak
**Related**: [PCA_GUIDE.md](PCA_GUIDE.md), [API_REFERENCE.md](API_REFERENCE.md)

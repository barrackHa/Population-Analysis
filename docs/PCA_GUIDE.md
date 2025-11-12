# PCA Analysis Guide

## Overview

**Purpose**: Analyze population dynamics in a low-dimensional space using Principal Component Analysis (PCA)

**Key Concept**: Neural populations traverse trajectories in a high-dimensional state space during task execution. PCA identifies the dominant dimensions (principal components) of population activity, revealing:
- Common patterns across neurons
- Trial-type-specific dynamics
- Direction selectivity in neural state space
- Temporal evolution of population activity

**Workflow**: Train/test split to ensure PCA captures generalizable population dynamics

---

## Epoch Selection

The optimal epoch for PCA is **[-50, 150] ms** relative to go_cue (t=0):

- **t < 0**: Fixation period - subject fixates at center waiting for target to appear
- **t = 0 (go_cue)**: Target appears (stimulus onset)
- **Reaction Time (RT)**: Time from go_cue to movement onset (`first_relevant_saccade[0]`), typically 100-150 ms
- **~100-150 ms**: Movement onset (saccade begins)

### What This Epoch Captures:
- A bit before target appearance (end of fixation)
- Target processing and decision period
- The entire reaction time window
- Movement onset and shortly after

This epoch differentiates trials with movement onset (GO, successful CONT, failed STOP) from trials without (successful STOP, failed CONT).

---

## PCA Pipeline

### Step 1: Data Preparation

```python
from cell_analysis import Cell, PopulationAnalyzer
from session_class import Session
import pandas as pd

# Load session data
cell_df = pd.read_pickle('data/unified_cell_trial_data/msn_fiona_cell_trial_data.pkl')
session_data = cell_df[cell_df['trial_session'] == 'fi211110a']
session = Session(session_data, verbose=True)

# Remove cells with incomplete data
session.drop_cells_with_missing_trial_type_or_dir_data()

# Split into train/test
train_session, test_session = session.split_to_train_test(
    test_fraction=0.5,
    random_state=42
)
```

---

### Step 2: Calculate Average PSTH (Baseline)

```python
# Calculate average activity across all conditions (training set)
# Note: Epoch is currently being optimized - [-50, 150] focuses on critical period
avg_psth = train_session.get_population_PSTH_single_condition(
    epok=[-50, 150],
    bin_size=1,
    alignment_point='go_cue',
    trial_type='GO',
    smooth=True,
    smooth_ker_size=25,
    delta=True,  # Mean-center the PSTH
    normalize_bins=False,
    normalize=True,
    sort_by_peak=False
)
```

---

### Step 3: Get Condition-Specific Data

```python
import numpy as np

def get_data_matrix_for_PCA(session, epok=[-50, 150]):
    """
    Get PSTH matrix for PCA with left and right trials concatenated.

    Returns:
        np.ndarray: Matrix of shape (n_cells, 2*n_bins)
                   First n_bins columns: left direction
                   Last n_bins columns: right direction
    """
    left_right_data = session.get_population_PSTHs_left_right(
        epok=epok,
        bin_size=1,
        alignment_point='go_cue',
        trial_type='GO',  # or 'STOP', 'CONT'
        success_only=True,
        smooth=True,
        smooth_ker_size=15,
        delta=True,
        normalize_bins=False,
        normalize=True
    )

    # Subtract average PSTH to isolate condition-specific activity
    left = left_right_data['left']['psth_matrix'] - avg_psth['psth_matrix']
    right = left_right_data['right']['psth_matrix'] - avg_psth['psth_matrix']

    # Concatenate left and right
    psth_matrix = np.concatenate([left, right], axis=1)
    return psth_matrix
```

---

### Step 4: Fit PCA on Training Data

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def pca_mat(mat, pca_components=5):
    """
    Fit PCA on neural population matrix.

    Parameters:
    -----------
    mat : np.ndarray
        Matrix of shape (n_cells, n_time_bins*2)
    pca_components : int
        Number of principal components to extract

    Returns:
    --------
    sklearn.decomposition.PCA : Fitted PCA object
    sklearn.preprocessing.StandardScaler : Fitted scaler object
    """
    # Standardize across time bins (each column has mean=0, std=1)
    scaler = StandardScaler()
    mat_for_pca = scaler.fit_transform(mat.T)

    # Fit PCA
    pca = PCA(n_components=pca_components)
    pca.fit(mat_for_pca)

    return pca, scaler

# Train PCA
train_matrix = get_data_matrix_for_PCA(train_session, epok=[-50, 150])
pca, scaler = pca_mat(train_matrix, pca_components=5)

# Check explained variance
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Cumulative variance: {np.cumsum(pca.explained_variance_ratio_)}")
```

---

### Step 5: Transform Test Data

```python
# Get test data
test_matrix = get_data_matrix_for_PCA(test_session, epok=[-50, 150])

# Transform to PC space
mat_scaled = scaler.transform(test_matrix.T)
pc_scores = pca.transform(mat_scaled).T  # Shape: (n_components, n_time_bins*2)

# Split into left and right directions
epok = [-50, 150]
cutoff = epok[1] - epok[0]  # 200 time bins
left_PCs = pc_scores[:, :cutoff]   # First half: left direction
right_PCs = pc_scores[:, cutoff:]  # Second half: right direction
```

---

### Step 6: Visualize 3D Trajectories

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3D_PC_trajectories(left_PCs, right_PCs, time):
    """
    Plot neural trajectories in 3D PC space.

    Parameters:
    -----------
    left_PCs : np.ndarray
        PC scores for left direction, shape (n_components, n_time_bins)
    right_PCs : np.ndarray
        PC scores for right direction, shape (n_components, n_time_bins)
    time : np.ndarray
        Time vector
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot left trajectory (180°)
    ax.plot(left_PCs[0, :], left_PCs[1, :], left_PCs[2, :],
            color='blue', linewidth=2, label='Left (180°)')

    # Plot right trajectory (0°)
    ax.plot(right_PCs[0, :], right_PCs[1, :], right_PCs[2, :],
            color='green', linewidth=2, label='Right (0°)')

    # Mark start points
    ax.scatter(left_PCs[0, 0], left_PCs[1, 0], left_PCs[2, 0],
               marker='^', s=200, color='blue', edgecolors='black',
               linewidths=2, label='Left Start', zorder=5)

    ax.scatter(right_PCs[0, 0], right_PCs[1, 0], right_PCs[2, 0],
               marker='^', s=200, color='green', edgecolors='black',
               linewidths=2, label='Right Start', zorder=5)

    # Labels
    ax.set_xlabel('PC1', fontsize=14)
    ax.set_ylabel('PC2', fontsize=14)
    ax.set_zlabel('PC3', fontsize=14)
    ax.set_title('Left vs Right 3D PC Trajectories', fontsize=16)
    ax.legend(fontsize=12)

    return fig, ax

# Plot
time = np.arange(epok[0], epok[1])
fig, ax = plot_3D_PC_trajectories(left_PCs, right_PCs, time)
plt.show()
```

---

### Step 7: Compare GO vs STOP/CONT Trials

```python
# Get STOP trial data in PC space
stop_data = test_session.get_population_PSTHs_left_right(
    epok=[-50, 150],  # Use same epoch as training
    bin_size=1,
    alignment_point='go_cue',  # or 'stop_cue' for alignment to stop signal
    trial_type='STOP',
    success_only=True,
    smooth=True,
    smooth_ker_size=25,
    delta=True,
    normalize_bins=False,
    normalize=False
)

# Subtract baseline
left_stop = stop_data['left']['psth_matrix'] - avg_psth['psth_matrix']
right_stop = stop_data['right']['psth_matrix'] - avg_psth['psth_matrix']
stop_matrix = np.concatenate([left_stop, right_stop], axis=1)

# Transform to PC space
stop_scaled = scaler.transform(stop_matrix.T)
stop_pc_scores = pca.transform(stop_scaled).T
stop_left_PCs = stop_pc_scores[:, :cutoff]
stop_right_PCs = stop_pc_scores[:, cutoff:]

# Visualize GO vs STOP trajectories
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# GO trials
ax.plot(left_PCs[0, :], left_PCs[1, :], left_PCs[2, :],
        color='blue', linewidth=2, label='GO Left')
ax.plot(right_PCs[0, :], right_PCs[1, :], right_PCs[2, :],
        color='green', linewidth=2, label='GO Right')

# STOP trials
ax.plot(stop_left_PCs[0, :], stop_left_PCs[1, :], stop_left_PCs[2, :],
        color='red', linewidth=2, linestyle='dashed', label='STOP Left')
ax.plot(stop_right_PCs[0, :], stop_right_PCs[1, :], stop_right_PCs[2, :],
        color='orange', linewidth=2, linestyle='dashed', label='STOP Right')

# Start markers
ax.scatter(left_PCs[0, 0], left_PCs[1, 0], left_PCs[2, 0],
           marker='^', s=200, color='blue', edgecolors='black', linewidths=2)
ax.scatter(stop_left_PCs[0, 0], stop_left_PCs[1, 0], stop_left_PCs[2, 0],
           marker='*', s=200, color='red', edgecolors='black', linewidths=2)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('GO vs STOP Trajectories in PC Space')
ax.legend()
plt.show()
```

---

## Key Insights from PCA

1. **Trajectory Separation**: GO left and GO right trajectories diverge in PC space, indicating direction-selective population activity

2. **STOP Signal Effect**: STOP trial trajectories may:
   - Start similarly to GO trials
   - Diverge after stop signal presentation
   - Terminate earlier or follow different path

3. **Explained Variance**: First 3 PCs typically capture 60-80% of population variance

4. **Temporal Dynamics**: Time evolution along PC trajectories reveals:
   - Initiation phase (similar across conditions)
   - Decision phase (divergence based on trial type)
   - Execution phase (direction-specific patterns)

---

## Best Practices for PCA

1. **Always use train/test split**: Prevents overfitting to specific trial noise
2. **Subtract average PSTH**: Isolates condition-specific activity patterns
3. **Standardize before PCA**: Ensures each time bin contributes equally
4. **Use mean-centered PSTHs** (`delta=True`): Removes cell-specific baseline differences
5. **Smooth PSTHs**: Reduces high-frequency noise (typical `smooth_ker_size=15-25`)
6. **Epoch selection** (`[-50, 150]`): Capture from just before go_cue to just after movement onset
   - Includes end of fixation, target processing, reaction time, and movement initiation
   - Differentiates movement vs. no-movement trials
7. **Check explained variance**: Ensure first few PCs capture substantial variance
8. **Save PCA results**: Store PC scores as numpy arrays for later analysis

---

## Data Storage

```python
# Save PC scores for later use
np.save('data/PCA_data/go_left_PCs.npy', left_PCs)
np.save('data/PCA_data/go_right_PCs.npy', right_PCs)
np.save('data/PCA_data/stop_left_PCs.npy', stop_left_PCs)
np.save('data/PCA_data/stop_right_PCs.npy', stop_right_PCs)
np.save('data/PCA_data/time_vector.npy', time)

# Save PCA model and scaler
import pickle
with open('data/PCA_data/pca_model.pkl', 'wb') as f:
    pickle.dump({'pca': pca, 'scaler': scaler}, f)

# Load later
left_PCs = np.load('data/PCA_data/go_left_PCs.npy')
with open('data/PCA_data/pca_model.pkl', 'rb') as f:
    models = pickle.load(f)
    pca = models['pca']
    scaler = models['scaler']
```

---

## Multi-Session PCA

**Purpose**: Combine neural populations across multiple recording sessions for larger-scale PCA analysis.

**Files**:
- `population_analysis/multi_session_PCA_analysis.ipynb` - Main analysis notebook
- `population_analysis/pca_helpers.py` - Worker functions for parallel processing

### Key Differences from Single-Session:
- Combines 1000+ cells from 40+ sessions (vs. 10-50 cells from one session)
- Uses ProcessPoolExecutor for parallel PSTH extraction
- No train/test split (all data used, cross-session variability provides validation)
- Memory-efficient: discards Session objects after extracting PSTH matrices
- Tracks cell-to-session mapping for post-hoc analysis

### Configuration:

```python
EPOK = [-50, 150]  # ms relative to go_cue
BIN_SIZE = 1  # ms
NORMALIZE = 'by_baseline_FR'  # Subtract baseline FR from each cell
N_PCA_COMPONENTS = 5
MIN_CELLS_PER_SESSION = 10  # Validation threshold
MIN_TRIALS_PER_CONDITION = 5  # Validation threshold
```

### Workflow:

1. **Validate sessions** (sufficient cells and trials per condition)
2. **Extract PSTH matrices in parallel** using `pca_helpers.extract_session_psth_worker()`
3. **Concatenate matrices** across all sessions
4. **Fit PCA** on combined population
5. **Project GO and STOP data** to PC space
6. **Visualize trajectories** and save results to `data/PCA_data/multi_session/`

### Example Results (Fiona, 40 sessions, 1,322 cells):
- PC1-3: 81.2% variance explained
- PC1-5: 89.7% variance explained
- Clear GO left/right trajectory separation
- STOP trajectories diverge from GO after signal onset

### Important Note:
Worker function must be in separate `.py` module (not notebook) for ProcessPoolExecutor pickling.

---

**Related Documentation**:
- [API Reference](API_REFERENCE.md)
- [Workflows & Usage Patterns](WORKFLOWS.md)
- [Coding Standards](STANDARDS.md)

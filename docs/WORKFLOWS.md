# Workflows & Usage Patterns

## Setup and Imports

```python
# Standard imports
from cell_analysis import Cell, PopulationAnalyzer
from session_class import Session
import pandas as pd
import numpy as np

# Load data
cell_df = pd.read_pickle('data/unified_cell_trial_data/msn_fiona_cell_trial_data.pkl')
```

---

## Pattern 1: Single Cell Analysis

```python
# Load data
cell_df = pd.read_pickle('msn_fiona_cell_trial_data.pkl')

# Create analyzer
analyzer = PopulationAnalyzer(cell_df)

# Get specific cell
cell = analyzer.get_cell(cell_id)

# Create visualizations
rasters = cell.plot_raster_by_type_direction(alignment_point='go_cue')
psth = cell.plot_psth_by_type_direction(separate_ssd=True, smooth=True)

# Display specific condition
rasters[0]['STOP']  # STOP trials, right direction (0°)
rasters[1]['GO']    # GO trials, left direction (180°)

# Check baseline firing rate
print(f"Baseline FR: {cell.baseline_FR:.2f} spikes/sec")
```

---

## Pattern 2: Population Analysis

```python
# Select best session
session_data = cell_df[cell_df['trial_session'] == 'fi211110a']

# Create session object
session = Session(session_data)

# Single condition heatmap
heatmap = session.plot_population_PSTH_heatmap(
    epok=[-200, 700],
    bin_size=10,
    alignment_point='stop_cue',
    trial_type='STOP',
    direction=0,
    success_only=True,
    smooth=True,
    normalize=True,
    sort_by_peak=True
)
```

---

## Pattern 3: Data Separation & Reuse

```python
# Generate data once
data = session.get_population_PSTH_single_condition(
    epok=[-200, 700],
    bin_size=10,
    alignment_point='go_cue',
    trial_type='STOP',
    direction=0,
    success_only=True,
    smooth=True,
    normalize=True,
    sort_by_peak=True
)

# Reuse for multiple plots
plot1 = session.plot_population_PSTH_heatmap(data=data)

# Or analyze the data directly
psth_matrix = data['psth_matrix']  # Shape: (n_cells, n_bins)
bin_centers = data['bin_centers']  # Time points
cell_ids = data['cell_ids']        # Ordered cell IDs

# Save for later
np.save('stop_condition_psth.npy', psth_matrix)
```

---

## Pattern 4: SSD-Specific Analysis

```python
# Compare STOP trials across SSDs
stop_comparison = session.plot_trial_type_PSTH_by_ssd(
    trial_type='STOP',
    epok_stop=[-200, 700],
    bin_size=10,
    success_only=True,
    smooth=True,
    normalize=True
)

# Then CONT trials
cont_comparison = session.plot_trial_type_PSTH_by_ssd(
    trial_type='CONT',
    epok_stop=[-200, 700],
    bin_size=10,
    success_only=True,
    smooth=True,
    normalize=True
)
```

---

## Pattern 5: Multi-Condition Comparison

```python
# Compare left vs right for a trial type
left_right_comparison = session.plot_left_right_PSTH_comparison(
    epok=[-200, 700],
    bin_size=10,
    alignment_point='go_cue',
    trial_type='GO',
    success_only=True,
    smooth=True,
    normalize=True
)

# Compare all trial types in 3×2 grid
all_types_comparison = session.plot_trial_type_PSTH_comparison(
    epok_go=[-200, 700],
    epok_stop=[-200, 700],
    bin_size=10,
    direction=0,
    ssd_number=None,
    success_only=True,
    smooth=True,
    normalize=True
)
```

---

## Pattern 6: Cell Filtering for Analysis

```python
# Create session
session = Session(session_data, verbose=True)

# Remove cells with incomplete data
session.drop_cells_with_missing_trial_type_or_dir_data()

# Check for cells with no spikes
n_empty = session.get_cells_with_no_spikes()
print(f"Cells with no spikes: {n_empty}")

# Get percentage
pct_empty = session.get_cells_with_no_spikes(as_percentage=True)
print(f"Percentage empty: {pct_empty:.1f}%")

# Now proceed with analysis
# All cells have data for all conditions
```

---

## Pattern 7: Alignment Point Selection

### GO Trials
```python
# GO trials: Always use go_cue
go_data = session.get_population_PSTH_single_condition(
    epok=[-200, 700],
    bin_size=10,
    alignment_point='go_cue',  # Only valid option for GO
    trial_type='GO',
    direction=0
)
```

### STOP Trials
```python
# Option 1: Align to go_cue (compare initiation across conditions)
stop_go_aligned = session.get_population_PSTH_single_condition(
    epok=[-200, 1000],  # Longer epoch to capture stop signal
    alignment_point='go_cue',
    trial_type='STOP',
    direction=0
)

# Option 2: Align to stop_cue (focus on inhibition process)
stop_signal_aligned = session.get_population_PSTH_single_condition(
    epok=[-200, 700],
    alignment_point='stop_cue',  # Stop signal appears at t=0
    trial_type='STOP',
    direction=0
)
```

### CONT Trials
```python
# Similar to STOP - choose based on research question
# Note: stop_cue contains CONTINUE SIGNAL time for CONT trials

# Option 1: Align to go_cue
cont_go_aligned = session.get_population_PSTH_single_condition(
    epok=[-200, 1000],
    alignment_point='go_cue',
    trial_type='CONT',
    direction=0
)

# Option 2: Align to continue signal
cont_signal_aligned = session.get_population_PSTH_single_condition(
    epok=[-200, 700],
    alignment_point='stop_cue',  # Continue signal (green) appears at t=0
    trial_type='CONT',
    direction=0
)
```

---

## Pattern 8: Normalization Options

```python
# No normalization (raw firing rates)
# Use for PCA and quantitative comparisons
raw_data = session.get_all_cells_psth(
    epok=[-200, 700],
    bin_size=10,
    trial_type='GO',
    normalize=False
)

# Normalize by max (each cell to [0, 1])
# Better for visualizing weak and strong cells together
normalized_data = session.get_all_cells_psth(
    epok=[-200, 700],
    bin_size=10,
    trial_type='GO',
    normalize=True  # or 'by_max'
)

# Normalize by baseline FR (subtract baseline)
# Centers each cell around its baseline firing rate
baseline_centered = session.get_all_cells_psth(
    epok=[-200, 700],
    bin_size=10,
    trial_type='GO',
    normalize='by_baseline_FR'
)
```

---

## Pattern 9: Working with Individual Cells from Session

```python
# Get specific cell as Cell instance
cell = session.get_cell(cell_id, verbose=True)

# Now use Cell class methods
psth = cell.plot_psth_by_type_direction(
    epok=[-200, 700],
    smooth=True,
    delta=False,
    smooth_ker_size=25
)

# Check baseline
print(f"Cell {cell_id} baseline FR: {cell.baseline_FR:.2f} spikes/sec")
```

---

## Pattern 10: Session Selection

### Example Session: fi211110a
One of the best recording sessions with high cell count and good trial distribution.

```python
# Load specific session
session_data = cell_df[cell_df['trial_session'] == 'fi211110a']
session = Session(session_data, verbose=True)

# Check session statistics
print(f"Number of cells: {len(session.data['cell_ID'].unique())}")
print(f"Number of trials: {len(session.data)}")
print(f"Trial types: {session.data['type'].value_counts()}")
```

### Finding Good Sessions
```python
# Count cells per session
session_cell_counts = cell_df.groupby('trial_session')['cell_ID'].nunique()
good_sessions = session_cell_counts[session_cell_counts >= 10].index.tolist()

print(f"Sessions with 10+ cells: {len(good_sessions)}")
print(good_sessions)
```

---

## Common Epochs

```python
# Full trial
epok = [-500, 1500]

# Standard analysis
epok = [-200, 700]

# Pre-stimulus baseline
epok = [-200, 0]

# Post-stimulus response
epok = [0, 700]

# PCA epoch (go_cue to movement onset)
epok = [-50, 150]
```

---

## Quick Reference

```python
# Load data
cell_df = pd.read_pickle('data/unified_cell_trial_data/msn_fiona_cell_trial_data.pkl')

# Single cell
cell = Cell(cell_df[cell_df['cell_ID'] == cell_id], verbose=True)
cell.plot_psth_by_type_direction(smooth=True, delta=False, smooth_ker_size=25)

# Population
session = Session(cell_df[cell_df['trial_session'] == 'fi211110a'])
session.drop_cells_with_missing_trial_type_or_dir_data()
session.plot_trial_type_PSTH_comparison(epok_go=[-200, 700], bin_size=10)

# SSD analysis
session.plot_trial_type_PSTH_by_ssd(trial_type='STOP', epok_stop=[-200, 700])
```

---

**Related Documentation**:
- [API Reference](API_REFERENCE.md)
- [PCA Analysis Guide](PCA_GUIDE.md)
- [Data Structures](DATA_REFERENCE.md)

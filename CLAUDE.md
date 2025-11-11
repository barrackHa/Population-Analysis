# MSN Population Analysis Project - Developer Reference

## Table of Contents
1. [Project Overview](#project-overview)
2. [Stop-Signal Task Structure](#stop-signal-task-structure)
3. [Data Structures](#data-structures)
4. [Architecture & Classes](#architecture--classes)
5. [File Organization](#file-organization)
6. [Key Methods Reference](#key-methods-reference)
7. [Workflow & Usage Patterns](#workflow--usage-patterns)
8. [PCA Analysis](#pca-analysis) **(NEW)**
9. [Coding Guidelines](#coding-guidelines)
10. [Visualization Standards](#visualization-standards)
11. [Important Implementation Notes](#important-implementation-notes)

---

## Project Overview

This project analyzes neuronal recordings from the caudate nucleus during a countermanding stop-signal task (CSST), replicating and extending the methodology from **Pani et al. (2022)**: "Neuronal Activity in the Primate Caudate Nucleus Relates to Visual Salience and Action Selection During Stop-Signal Task".

### Current Scope
- **Cell Type**: Currently analyzing Medium Spiny Neurons (MSN), but architecture is designed to generalize to other cell types
- **Subject**: Fiona (Female Macaque, ~7-8 kg) - primary focus
- **Screen Configuration**: `screen_rotation = 0` (standard horizontal configuration). For Yasmin, in some trials `screen_rotation ≠ 0`, coordinate transformations will be needed.

### Database Statistics (Fiona - Grade <= 8 cells only)
- **Recording period**: Multiple sessions over several months
- **Total cell-trial combinations**: 844,694 rows
- **Recording sessions**: 59 sessions (3 sessions excluded due to data quality issues)
- **Quality threshold**: Grade <= 8 (scale 5-11, based on waveform quality and isolation) 

**Excluded sessions** (Fiona):
- `fi210628`, `fi210629`, `fi210704` - excluded due to data quality issues

**Note**: The 'msn_{monkey}_cell_trial_data.pkl' database contains cell-trial combinations, not unique trials. Each row represents one cell's activity during one trial, so multiple cells recorded simultaneously in the same trial create multiple rows. The original 

### Trial Distribution & Performance
**Trial type breakdown (all sessions)**:
- **GO trials**: 467,602 cell-trials (55.4%)
  - Success rate: 96.6% (executed saccade as instructed)
  - Failed trials: 15,682 (no saccade or late saccade)
  
- **CONT trials**: 196,548 cell-trials (23.3%)
  - Success rate: 85.1% (executed saccade despite continue signal)
  - Failed trials: 29,343 (inappropriately inhibited)
  
- **STOP trials**: 180,544 cell-trials (21.4%)
  - Success rate: 54.5% (successfully inhibited saccade)
  - Failed trials: 82,173 (failed to inhibit - these could be analyzed as "error STOP" trials)

**Key Observations**:
- STOP trials have ~50% success rate by design (adaptive staircase procedure)
- CONT trials have high success rate, validating they're easier than STOP
- GO trials have very high success rate (baseline performance)

### Data Quality Control
**Saccade Amplitude Filtering**:
- Trial failures for STOP trials are validated by saccade amplitude
- Method: `update_stop_trial_failures()` recalculates failures based on whether saccade amplitude exceeds threshold
- This ensures that "failed" STOP trials actually had executed saccades, not just noise

**Session Exclusions**:
- Some sessions excluded due to data quality issues (e.g., recording artifacts, insufficient trials)
- Fiona excluded sessions: `fi210628`, `fi210629`, `fi210704`

---

## Future Development & Scope

### Planned Expansions

#### 1. Yasmin Data Integration
- **Status**: Data collected but not yet analyzed
- **Challenge**: Yasmin has `screen_rotation ≠ 0` (rotated display configuration)
- **Blocker**: Need to develop coordinate transformation methods to align with Fiona's standard orientation
- **Impact**: Will require updates to:
  - Direction encoding (currently 0° and 180°)
  - Saccade direction calculations
  - Target position mappings

#### 2. Cell Type Generalization
- **Current**: MSN (Medium Spiny Neurons) only
- **Future**: Extend to other cell types recorded simultaneously
- **Architecture**: Code is already designed to be cell-type agnostic
- **Required Changes**:
  - Cell type filtering options in analysis classes
  - Cell type-specific visualization palettes
  - Comparative analyses across cell types

#### 3. Multi-Subject Comparisons
- Once Yasmin data is integrated:
  - Cross-subject population comparisons
  - Individual difference analyses
  - Subject as factor in statistical models

---

## Stop-Signal Task Structure

### Task Description
A countermanding stop-signal task (CSST) where the subject performs visually-guided saccades with three trial types:

**Basic Task Flow:**
1. **Fixation**: Subject fixates on a central point
2. **Go Cue**: Peripheral target appears, signaling the subject to make a saccade
3. **Signal Cue** (STOP/CONT trials only): Secondary visual cue appears after a variable delay
   - **Stop Signal**: Red cue instructs subject to cancel the saccade
   - **Continue Signal**: Green cue instructs subject to proceed with the saccade (control condition)
4. **Response**: Subject either executes or inhibits the saccade depending on the trial type

### Trial Types

#### 1. GO Trials
- **Description**: No secondary signal presented
- **Expected behavior**: Execute saccade to target
- **Purpose**: Baseline condition measuring normal saccadic response
- **Success criterion**: Saccade executed within time window
- **Data markers**: 
  - `type = 'GO'`
  - `stop_cue = NaN` (no signal)
  - `ssd_number = NaN`
- **Alignment**: `go_cue`

#### 2. STOP Trials (Successful Inhibition)
- **Description**: **Red stop signal** presented after variable delay
- **Expected behavior**: Cancel/inhibit the planned saccade
- **Outcome**: `trial_failed = False` (successfully inhibited)
- **Purpose**: Measure inhibitory control capacity
- **Success criterion**: No saccade made after stop signal
- **Data markers**:
  - `type = 'STOP'`
  - `stop_cue = timestamp` (time of red stop signal)
  - `ssd_number = 1.0-4.0` (stop signal delay level)
  - `first_relevant_saccade = NaN` (no saccade)
- **Alignment**: `go_cue` or `stop_cue`

#### 3. CONT (Continue) Trials (Control Condition)
- **Description**: **Green continue signal** presented after variable delay
- **Expected behavior**: Proceed with the saccade despite the visual cue
- **Outcome**: `trial_failed = False` if saccade executed, `True` if inhibited
- **Purpose**: Control condition to match visual stimulation of STOP trials without inhibitory demand
- **Success criterion**: Saccade executed after continue signal
- **Data markers**:
  - `type = 'CONT'`
  - `stop_cue = timestamp` (time of **green continue signal**, NOT a stop signal)
  - `ssd_number = 1.0-4.0` (continue signal delay level)
  - `first_relevant_saccade = timestamp` (saccade executed)
- **Alignment**: `go_cue` or `stop_cue`

**IMPORTANT**: The `stop_cue` column contains:
- Time of **stop signal** (red) for STOP trials
- Time of **continue signal** (green) for CONT trials
- `NaN` for GO trials

This naming convention can be confusing but reflects the experimental design where both signals appear at matched delays (CSD/SSD).

### Directional Targets
- **0°**: Right direction (rightward saccade)
- **180°**: Left direction (leftward saccade)
- Targets appear at equal eccentricity on opposite sides of the screen

### Signal Delay Levels
- **CSD (Continue Signal Delay)**: Time between go cue and continue signal
- **SSD (Stop Signal Delay)**: Time between go cue and stop signal
- **Levels**: 4 discrete values encoded as `ssd_number = 1.0, 2.0, 3.0, 4.0`
- **Effect**: 
  - Shorter delays → easier inhibition (for STOP) / faster continue response (for CONT)
  - Longer delays → harder inhibition (for STOP) / approaching natural GO RT (for CONT)
- **Adaptive staircase**: Delays adjusted during session to maintain ~50% stop success rate

### Key Temporal Events
1. **Fixation period** (t < 0): Subject fixates at center waiting for target to appear
2. **go_cue** (t = 0): Visual target appears, signals saccade initiation (all trials)
3. **stop_cue**: Secondary signal appears (STOP/CONT trials only)
   - Red stop signal for STOP trials
   - Green continue signal for CONT trials
4. **first_relevant_saccade**: Array [start, end] of saccade times (if executed)
   - `first_relevant_saccade[0]`: Saccade start time (movement onset)
   - `first_relevant_saccade[1]`: Saccade end time (fixation on target)
5. **reaction_time**: Interval from `go_cue` to `first_relevant_saccade[0]`
   - Typical RT: 100-150 ms

### Experimental Rationale
The CONT trials serve as a critical control condition:
- Match the visual stimulation of STOP trials (a secondary cue appears)
- Control for visual distraction effects
- Isolate inhibitory processes specific to stopping vs. general cue processing
- Enable comparison: successful STOP vs. successful CONT (both inhibit vs. both execute)

---

## Data Structures

### Main DataFrame (`cell_df`)
Unified cell-trial database stored as pickle file: `msn_fiona_cell_trial_data.pkl`

#### Key Columns
```python
# Cell identification
'cell_ID'           # Unique cell identifier (int)
'cell_type'         # Cell type classification
'trial_session'     # Session identifier (e.g., 'fi211110a')

# Trial classification
'type'              # 'GO', 'STOP', 'CONT'
'dir'               # Direction: 0 (right) or 180 (left)
'ssd_number'        # SSD level: 1.0-4.0 (NaN for GO trials)
'trial_failed'      # Boolean: Success depends on trial type
                    #   GO: False=saccade made, True=no saccade
                    #   STOP: False=successfully inhibited, True=failed to inhibit
                    #   CONT: False=saccade made, True=inappropriately inhibited

# Temporal events (timestamps in ms)
'go_cue'            # Go signal time
'stop_cue'          # Stop signal (STOP) OR Continue signal (CONT) time (NaN for GO trials)
'first_relevant_saccade'  # Array [start, end]: saccade start time (movement onset) and end time (target fixation)
'reaction_time'     # RT from go_cue to first_relevant_saccade[0]

# Neural data
'neural_data'       # Array of spike times (absolute timestamps)

# Trial metadata
'trial_number'      # Trial number within session
'trial_length'      # Total trial duration
'screen_rotation'   # Screen orientation (0 for Fiona, varies for Yasmin)
'saccades'          # All saccade events
'blinks'            # Blink events
'grade'             # Cell quality score (1-10, ≥8 recommended)
```

**Important Notes**:
- **screen_rotation**: Currently only analyzing `screen_rotation = 0` (Fiona). Yasmin has rotated display requiring coordinate transformations.
- **grade**: Quality metric for cell isolation. Threshold of <= 8 filters out poorly isolated units.

---

## Architecture & Classes

### Class Hierarchy

```
PopulationAnalyzer (cell_analysis.py)
├── Manages full cell database
└── Creates → Cell objects

Cell (cell_analysis.py)
├── Single neuron analysis
├── Trial-level operations
├── Baseline firing rate calculation
└── Individual cell visualizations

Session (session_class.py)
├── Session-level population analysis
├── Multi-cell population operations
├── PCA-ready data preparation
├── Train/test data splitting
└── Population visualizations
```

### Cell Class
**Location**: `population_analysis/cell_analysis.py` (moved from analysis.ipynb)

**Purpose**: Single-cell analysis with trial-level visualizations

**Key Features**:
- Spike alignment to task events
- Trial filtering and sorting
- **NEW**: Baseline firing rate property
- Raster plots (color-coded by condition)
- Histograms (spike counts)
- PSTH (firing rate with smoothing)
- **NEW**: Mean-centered firing rate (delta parameter)
- **NEW**: Configurable smoothing kernel size

**Properties**:
```python
baseline_FR  # Baseline firing rate (spikes/sec) calculated from -500 to 0 ms before go_cue
```

**Core Methods**:
```python
# Data operations
align_spikes_to_event(alignment_point)
filter_trials(trial_type, direction, ssd_number, success_only, failed_only)
aggregate_spikes_by_bins(epok, bin_size, normalize=False, ...)
calculate_psth(epok, bin_size, smooth=True, delta=False,
               smooth_ker_size=25, normalize_bins=False, ...)

# Visualizations
plot_raster_by_type_direction(epok, alignment_point, show_legend)
plot_histogram_by_type_direction(epok, bin_size, separate_ssd, normalize)
plot_psth_by_type_direction(epok, bin_size, separate_ssd, smooth,
                             delta=False, smooth_ker_size=25,
                             normalize_bins=False)
```

**NEW Parameters**:
- `delta` (bool): If True, center PSTH to mean firing rate (subtract mean)
- `smooth_ker_size` (int): Gaussian smoothing kernel size (default: 25)
- `normalize_bins` (bool): If True, z-score normalize spike counts before PSTH calculation
- `verbose` (bool): Print initialization details

### Session Class
**Location**: `population_analysis/session_class.py`

**Purpose**: Population-level analysis across all cells in a session

**Architecture**: Separated data generation from plotting
- **Data methods**: Return dictionaries with results
- **Plot methods**: Accept pre-generated data or generate on-the-fly

**Key Features**:
- Global normalization (preserves relative cell differences)
- **NEW**: Baseline FR normalization option
- Peak-based cell ordering
- Multi-condition comparisons
- Grid layouts for comprehensive views
- **NEW**: Cell filtering and validation
- **NEW**: Train/test data splitting for PCA

**Properties**:
```python
cells  # Generator yielding Cell instances for all cells in session
```

**Core Data Methods**:
```python
# Cell access and filtering (NEW)
get_cell_data(cell_id)  # Get DataFrame for specific cell
get_cell(cell_id, verbose=False)  # Get Cell instance for specific cell
drop_cell_from_session(cell_id)  # Remove cell from session
drop_cells_with_missing_trial_type_or_dir_data()  # Remove incomplete cells
get_cells_with_no_spikes(as_percentage=False)  # Count cells with no activity

# PSTH generation (UPDATED)
get_cell_psth(cell_id, epok, bin_size, alignment_point, trial_type,
              direction, ssd_number, success_only, smooth, delta=False,
              smooth_ker_size=25, normalize_bins=False, normalize=False)
              # normalize options: False, True, 'by_max', 'by_baseline_FR'

get_all_cells_psth(epok, bin_size, ..., normalize=False)

# Spike counts (for PCA)
get_cell_spike_counts(cell_id, epok, bin_size, alignment_point, ...)
get_all_cells_spike_counts(epok, bin_size, ..., normalize=True)

# Population data generation
get_population_spike_counts_data(epok, bin_size, alignment_point,
                                  trial_type, direction, ssd_number,
                                  success_only, normalize, sort_by_peak)

get_population_PSTH_single_condition(epok, bin_size, alignment_point,
                                      trial_type, direction, ssd_number,
                                      success_only, smooth, delta=False,
                                      smooth_ker_size=25, normalize_bins=False,
                                      normalize=False, sort_by_peak=True)

get_population_PSTHs_left_right(epok, bin_size, alignment_point,
                                trial_type, success_only, smooth, delta=False,
                                smooth_ker_size=25, normalize_bins=False,
                                normalize=False, sort_by_peak=True)

get_population_PSTHs_trial_types(epok_go, epok_stop, bin_size, direction,
                                  ssd_number, success_only, smooth, normalize)

# PCA support (NEW)
split_to_train_test(test_fraction=0.5, random_state=None)
  # Returns: (train_session, test_session) - both Session instances

# Plotting methods
plot_population_PSTH_heatmap(data, **kwargs)  # Single condition, 800×600px
plot_population_spike_counts_heatmap(data, **kwargs)  # Spike counts heatmap
plot_left_right_PSTH_comparison(data, **kwargs)  # 2 columns, 400×600px each
plot_trial_type_PSTH_comparison(data_left, data_right, **kwargs)  # 3×2 grid
plot_trial_type_PSTH_by_ssd(trial_type, ssd_numbers, **kwargs)  # 4×2 grid
```

---

## File Organization

### Repository Structure
```
population_analysis/
├── CLAUDE.md                    # This file
├── README.md                    # Project overview
├── requirements.txt             # Python dependencies
├── data/
│   ├── unified_cell_trial_data/
│   │   └── msn_fiona_cell_trial_data.pkl  # Main database
│   └── PCA_data/                # Saved PCA results (numpy arrays)
├── population_analysis/
│   ├── cell_analysis.py         # Cell & PopulationAnalyzer classes
│   ├── session_class.py         # Session class for population analysis
│   ├── plot_PCA_in_3D.py        # 3D PCA trajectory plotting utilities
│   ├── analysis.ipynb           # Legacy single-cell analysis notebook
│   ├── msn_analysis.ipynb       # MSN-specific analysis examples
│   ├── session_analysis.ipynb   # Session class demonstrations
│   ├── session_PCA_analysis.ipynb  # Single-session PCA analysis workflow
│   ├── multi_session_PCA_analysis.ipynb  # Multi-session PCA analysis **(NEW)**
│   ├── pca_helpers.py           # Worker functions for parallel PCA processing **(NEW)**
│   ├── test_cell_class.ipynb    # Cell class testing notebook
│   ├── test_session_class.ipynb # Session class testing notebook
│   ├── test_session_PSTH.ipynb  # PSTH method testing notebook
│   ├── pytest_test_cell_class.py # Pytest unit tests for Cell class
│   ├── maestro_file.py          # Maestro file parsing utilities
│   └── pre_proc_helper.py       # Preprocessing helpers
├── data_pre_proc/               # Data preprocessing notebooks
├── exploratory_data_analysis/   # EDA notebooks
└── simulation/                  # Neural analysis applications
```

### Key Files

#### 1. `cell_analysis.py` (NEW - formerly in analysis.ipynb)
- **Purpose**: Single-cell and population-level analysis classes
- **Size**: ~1000 lines
- **Main classes**: `Cell`, `PopulationAnalyzer`
- **Dependencies**: pandas, numpy, holoviews, scipy.ndimage, scipy.stats
- **Key Features**:
  - Baseline firing rate calculation
  - PSTH with delta (mean-centered) option
  - Configurable smoothing kernel size
  - Normalization by baseline FR or max FR

#### 2. `session_class.py`
- **Purpose**: Session-level population analysis
- **Size**: ~1220 lines
- **Main class**: `Session`
- **Dependencies**: pandas, numpy, holoviews, sklearn.model_selection
- **New Features**:
  - Cell filtering and data validation methods
  - Train/test data splitting for PCA
  - Enhanced normalization options
  - Baseline FR normalization support

#### 3. `plot_PCA_in_3D.py` (NEW)
- **Purpose**: 3D visualization of PCA trajectories
- **Size**: ~60 lines
- **Functionality**: Plots neural population trajectories in PC space
- **Uses**: matplotlib 3D plotting for GO vs STOP/CONT comparison

#### 4. `session_PCA_analysis.ipynb` (NEW)
- **Purpose**: Complete PCA analysis workflow
- **Key Sections**:
  - Data preparation and normalization
  - Train/test split for cross-validation
  - PCA fitting and transformation
  - 3D trajectory visualization
  - GO vs STOP/CONT trial comparison in PC space
  - Explained variance analysis

#### 5. `session_analysis.ipynb`
- **Purpose**: Demonstration notebook for Session class
- **Sections**:
  - Setup and imports
  - Example 1: Single condition heatmap
  - Example 2: Left vs right comparison
  - Example 3: Trial type comparison (3×2)
  - Example 4a: STOP trials by SSD (4×2)
  - Example 4b: CONT trials by SSD (4×2)

#### 6. `analysis.ipynb`
- **Purpose**: Legacy single-cell analysis notebook
- **Note**: Cell class now in `cell_analysis.py`
- **Includes**: Comprehensive single-cell visualizations

#### 7. `pytest_test_cell_class.py` (NEW)
- **Purpose**: Unit tests for Cell class
- **Framework**: pytest
- **Tests**: Baseline FR calculation, PSTH generation, spike alignment

---

## Key Methods Reference

### Spike Alignment
```python
def align_spikes_to_event(self, alignment_point='go_cue'):
    """
    Align spike times to a specific event.
    
    Parameters:
    - alignment_point: 'go_cue', 'stop_cue', 'first_relevant_saccade'
    
    Creates new column: 'spikes_aligned_to_{alignment_point}'
    Returns: pd.Series of aligned spike times
    """
```

### PSTH Calculation (UPDATED)
```python
def calculate_psth(self, epok=[-200, 700], bin_size=10,
                   alignment_point='go_cue', trial_type=None,
                   direction=None, ssd_number=None,
                   success_only=True, smooth=True,
                   delta=False, smooth_ker_size=25,
                   normalize_bins=False):
    """
    Calculate firing rate with Gaussian smoothing.

    NEW PARAMETERS:
    - delta (bool): If True, subtract mean firing rate to center around zero
    - smooth_ker_size (int): Gaussian smoothing kernel size (default: 25)
                            Previous versions used bin_size as sigma
    - normalize_bins (bool): If True, z-score normalize spike counts before
                            converting to firing rate

    Smoothing: sigma=smooth_ker_size, truncate=2
    - For smooth_ker_size=25ms: ±50ms smoothing window (2*sigma)
    - Smoothing applied BEFORE epoch trimming to avoid edge effects

    Returns: (bin_centers, firing_rate, n_trials)
    """
```

### Baseline Firing Rate (NEW)
```python
@property
def baseline_FR(self):
    """
    Calculate baseline firing rate from -500 to 0 ms before go_cue.

    Calculated once and cached in _baseline_FR.

    Returns: Firing rate in spikes/sec
    """
```

### Global Normalization (UPDATED)
```python
def get_all_cells_psth(self, ..., normalize=False):
    """
    Flexible normalization options for population PSTHs.

    normalize options:
    - False: No normalization (raw firing rates)
    - True or 'by_max': Divide each cell by its max firing rate
    - 'by_baseline_FR': Subtract baseline firing rate from each cell

    When normalize=False (NEW default):
    - Preserves actual firing rate magnitudes
    - Use for PCA and quantitative comparisons

    When normalize=True:
    - Each cell normalized to [0, 1] range
    - Better for visualizing weak and strong cells together

    Result: Population matrix (n_cells × n_bins)
    """
```

### Cell Filtering (NEW)
```python
def drop_cells_with_missing_trial_type_or_dir_data(self):
    """
    Remove cells that lack data for all trial types or directions.

    Criteria:
    - Must have both 0° and 180° direction trials
    - Must have all 3 trial types (GO, STOP, CONT)

    Essential for PCA and population analysis to ensure
    balanced comparisons across conditions.
    """
```

### Train/Test Split (NEW)
```python
def split_to_train_test(self, test_fraction=0.5, random_state=None):
    """
    Split session data into training and testing sets for PCA.

    Uses sklearn.model_selection.train_test_split on trial level.

    Parameters:
    - test_fraction: Fraction of trials for testing (default: 0.5)
    - random_state: Random seed for reproducibility

    Returns: (train_session, test_session) - both Session instances

    Use case:
    - Fit PCA on train_session
    - Validate generalization on test_session
    """
```

### Cell Ordering
```python
def get_population_data_single_condition(self, ..., sort_by_peak=True):
    """
    Sort cells by peak activity time (argmax).
    
    Process:
    1. Calculate PSTH for each cell
    2. Find time of peak firing (argmax)
    3. Sort cells in ascending order by peak time
    
    Maintains consistent ordering across related plots.
    """
```

---

## Workflow & Usage Patterns

### Pattern 1: Single Cell Analysis
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

# Display
rasters[0]['STOP']  # STOP trials, right direction
```

### Pattern 2: Population Analysis
```python
# Select best session
session_data = cell_df[cell_df['trial_session'] == 'fi211110a']

# Create session object
session = Session(session_data)

# Single condition heatmap
heatmap = session.plot_population_heatmap(
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

### Pattern 3: Data Separation & Reuse
```python
# Generate data once
data = session.get_population_data_single_condition(
    epok=[-200, 700],
    trial_type='STOP',
    direction=0,
    normalize=True
)

# Reuse for multiple plots
plot1 = session.plot_population_heatmap(data=data)
# Can also save data for later analysis
```

### Pattern 4: SSD-Specific Analysis
```python
# Compare STOP trials across SSDs
stop_comparison = session.plot_trial_type_by_ssd(
    trial_type='STOP',
    epok_stop=[-200, 700],
    bin_size=10,
    success_only=True,
    smooth=True,
    normalize=True
)

# Then CONT trials
cont_comparison = session.plot_trial_type_by_ssd(
    trial_type='CONT',
    epok_stop=[-200, 700],
    bin_size=10,
    success_only=True,
    smooth=True,
    normalize=True
)
```

### Pattern 5: PCA Population Dynamics Analysis (NEW)
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. Prepare session and split data
session_data = cell_df[cell_df['trial_session'] == 'fi211110a']
session = Session(session_data, verbose=True)
session.drop_cells_with_missing_trial_type_or_dir_data()

train_session, test_session = session.split_to_train_test(
    test_fraction=0.5, random_state=42
)

# 2. Get average baseline activity (epoch being optimized, currently [-50, 150])
avg_psth = train_session.get_population_PSTH_single_condition(
    epok=[-50, 150], bin_size=1, alignment_point='go_cue',
    trial_type='GO', smooth=True, smooth_ker_size=25,
    delta=True, normalize=True
)

# 3. Get condition-specific data (left + right concatenated)
def get_pca_matrix(session, trial_type='GO'):
    data = session.get_population_PSTHs_left_right(
        epok=[-50, 150], bin_size=1, alignment_point='go_cue',
        trial_type=trial_type, smooth=True, smooth_ker_size=15,
        delta=True, normalize=True
    )
    left = data['left']['psth_matrix'] - avg_psth['psth_matrix']
    right = data['right']['psth_matrix'] - avg_psth['psth_matrix']
    return np.concatenate([left, right], axis=1)

# 4. Fit PCA on training data
train_matrix = get_pca_matrix(train_session, 'GO')
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_matrix.T)
pca = PCA(n_components=5)
pca.fit(train_scaled)

print(f"Explained variance: {pca.explained_variance_ratio_}")

# 5. Transform test data
test_matrix = get_pca_matrix(test_session, 'GO')
test_scaled = scaler.transform(test_matrix.T)
pc_scores = pca.transform(test_scaled).T

# 6. Split and visualize
cutoff = 200  # epok[1] - epok[0] = 150 - (-50)
left_PCs = pc_scores[:, :cutoff]
right_PCs = pc_scores[:, cutoff:]

# 7. Plot 3D trajectories
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot(left_PCs[0], left_PCs[1], left_PCs[2], 'b-', lw=2, label='Left')
ax.plot(right_PCs[0], right_PCs[1], right_PCs[2], 'g-', lw=2, label='Right')
ax.legend()
plt.show()
```

---

## PCA Analysis

### Overview

**Purpose**: Analyze population dynamics in a low-dimensional space using Principal Component Analysis (PCA)

**Key Concept**: Neural populations traverse trajectories in a high-dimensional state space during task execution. PCA identifies the dominant dimensions (principal components) of population activity, revealing:
- Common patterns across neurons
- Trial-type-specific dynamics
- Direction selectivity in neural state space
- Temporal evolution of population activity

**Workflow**: Train/test split to ensure PCA captures generalizable population dynamics

**Epoch Selection**: The optimal epoch for PCA is **[-50, 150] ms** relative to go_cue (t=0):
- **t < 0**: Fixation period - subject fixates at center waiting for target to appear
- **t = 0 (go_cue)**: Target appears (stimulus onset)
- **Reaction Time (RT)**: Time from go_cue to movement onset (`first_relevant_saccade[0]`), typically 100-150 ms
- **~100-150 ms**: Movement onset (saccade begins)

The **[-50, 150] epoch captures**:
- A bit before target appearance (end of fixation)
- Target processing and decision period
- The entire reaction time window
- Movement onset and shortly after

This epoch differentiates trials with movement onset (GO, successful CONT, failed STOP) from trials without (successful STOP, failed CONT).

### PCA Pipeline

#### Step 1: Data Preparation
```python
# Load session data
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

#### Step 2: Calculate Average PSTH (Baseline)
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

#### Step 3: Get Condition-Specific Data
```python
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

#### Step 4: Fit PCA on Training Data
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
    """
    # Standardize across time bins (each column has mean=0, std=1)
    scaler = StandardScaler()
    mat_for_pca = scaler.fit_transform(mat.T)

    # Fit PCA
    pca = PCA(n_components=pca_components)
    pca.fit(mat_for_pca)

    return pca

# Train PCA
train_matrix = get_data_matrix_for_PCA(train_session, epok=[-50, 150])
pca = pca_mat(train_matrix, pca_components=5)

# Check explained variance
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
```

#### Step 5: Transform Test Data
```python
# Get test data
test_matrix = get_data_matrix_for_PCA(test_session, epok=[-50, 150])

# Transform to PC space
pc_scores = pca.transform(test_matrix.T).T  # Shape: (n_components, n_time_bins*2)

# Split into left and right directions
epok = [-50, 150]
cutoff = epok[1] - epok[0]  # 200 time bins
left_PCs = pc_scores[:, :cutoff]   # First half: left direction
right_PCs = pc_scores[:, cutoff:]  # Second half: right direction
```

#### Step 6: Visualize 3D Trajectories
```python
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
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

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

#### Step 7: Compare GO vs STOP/CONT Trials
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
stop_pc_scores = pca.transform(stop_matrix.T).T
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

ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('PC3')
ax.set_title('GO vs STOP Trajectories in PC Space')
ax.legend()
plt.show()
```

### Key Insights from PCA

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

### Best Practices for PCA

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

### Data Storage

```python
# Save PC scores for later use
np.save('data/PCA_data/go_left_PCs.npy', left_PCs)
np.save('data/PCA_data/go_right_PCs.npy', right_PCs)
np.save('data/PCA_data/stop_left_PCs.npy', stop_left_PCs)
np.save('data/PCA_data/stop_right_PCs.npy', stop_right_PCs)
np.save('data/PCA_data/time_vector.npy', time)

# Load later
left_PCs = np.load('data/PCA_data/go_left_PCs.npy')
```

### Multi-Session PCA **(NEW)**

**Purpose**: Combine neural populations across multiple recording sessions for larger-scale PCA analysis.

**Files**:
- `population_analysis/multi_session_PCA_analysis.ipynb` - Main analysis notebook
- `population_analysis/pca_helpers.py` - Worker functions for parallel processing

**Key Differences from Single-Session**:
- Combines 1000+ cells from 40+ sessions (vs. 10-50 cells from one session)
- Uses ProcessPoolExecutor for parallel PSTH extraction
- No train/test split (all data used, cross-session variability provides validation)
- Memory-efficient: discards Session objects after extracting PSTH matrices
- Tracks cell-to-session mapping for post-hoc analysis

**Configuration**:
```python
EPOK = [-50, 150]  # ms relative to go_cue
BIN_SIZE = 1  # ms
NORMALIZE = 'by_baseline_FR'  # Subtract baseline FR from each cell
N_PCA_COMPONENTS = 5
MIN_CELLS_PER_SESSION = 10  # Validation threshold
MIN_TRIALS_PER_CONDITION = 5  # Validation threshold
```

**Workflow**:
1. Validate sessions (sufficient cells and trials per condition)
2. Extract PSTH matrices in parallel using `pca_helpers.extract_session_psth_worker()`
3. Concatenate matrices across all sessions
4. Fit PCA on combined population
5. Project GO and STOP data to PC space
6. Visualize trajectories and save results to `data/PCA_data/multi_session/`

**Example Results (Fiona, 40 sessions, 1,322 cells)**:
- PC1-3: 81.2% variance explained
- PC1-5: 89.7% variance explained
- Clear GO left/right trajectory separation
- STOP trajectories diverge from GO after signal onset

**Important**: Worker function must be in separate `.py` module (not notebook) for ProcessPoolExecutor pickling.

---

## Coding Guidelines

### Data Processing

#### 1. Always Check for Missing Data
```python
# Example: SSD numbers are NaN for GO trials
if trial_type in ['STOP', 'CONT']:
    ssd_numbers = sorted(data['ssd_number'].dropna().unique())
```

#### 2. Spike Time Handling
```python
# Spikes are stored as numpy arrays
spikes = np.array(row['neural_data'], dtype=float)

# Always filter by epoch after alignment
aligned_spikes = spikes - alignment_time
spikes_in_epoch = aligned_spikes[(aligned_spikes >= epok[0]) & 
                                  (aligned_spikes <= epok[1])]
```

#### 3. Trial Filtering Best Practices
```python
# Chain filters explicitly
filtered = data.copy()
if trial_type is not None:
    filtered = filtered[filtered['type'] == trial_type]
if direction is not None:
    filtered = filtered[filtered['dir'] == direction]
if success_only:
    filtered = filtered[filtered['trial_failed'] == False]
```

### Visualization

#### 1. Consistent Color Schemes
```python
# Use defined palettes
SSD_COLORS = {1: '#e377c2', 2: '#8c564b', 3: '#bcbd22', 4: '#17becf'}
TYPE_COLORS = {'GO': '#2ca02c', 'STOP': '#d62728', 'CONT': '#9467bd'}
DIRECTION_COLORS = {0: '#1f77b4', 180: '#ff7f0e'}
```

#### 2. Holoviews Image Kdims Order
```python
# CORRECT: Time on X-axis, Neurons on Y-axis
img = hv.Image(
    psth_matrix,
    kdims=['Time', 'Neurons'],  # Order matters!
    vdims='Firing Rate',
    bounds=(epok[0], 0, epok[1], n_cells)
)
```

#### 3. Bounds vs Transpose
```python
# Use bounds parameter, NOT transpose
# bounds=(x_min, y_min, x_max, y_max)
bounds=(epok[0], 0, epok[1], n_cells)

# Set invert_yaxis=False (default behavior we want)
.opts(invert_yaxis=False)
```

#### 4. Plot Sizing for Grids
```python
# Single plots
width=800, height=600

# 2-column grids
width=400, height=300

# 3-column grids
width=350, height=600

# 4-column grids (SSD analysis)
width=300, height=250
```

### Performance

#### 1. Avoid Repeated Alignment
```python
# Check before aligning
col_name = f'spikes_aligned_to_{alignment_point}'
if col_name not in self.data.columns:
    self.align_spikes_to_event(alignment_point)
```

#### 2. Reuse Computed Data
```python
# Generate once
data_left = self.get_population_data_trial_types(direction=180, ...)
data_right = self.get_population_data_trial_types(direction=0, ...)

# Use multiple times
plot1 = self.plot_trial_type_comparison(data_left, data_right)
# Can analyze data_left['go']['psth_matrix'] separately
```

#### 3. SSD Analysis Separation
```python
# Instead of 8×2 grid (heavy):
plot_all_ssd()  # 16 heatmaps

# Use 4×2 grids (lighter):
plot_trial_type_by_ssd(trial_type='STOP')  # 8 heatmaps
plot_trial_type_by_ssd(trial_type='CONT')  # 8 heatmaps
```

---

## Visualization Standards

### Default Parameters
```python
# Temporal
EPOCH = [-200, 700]  # ms
BIN_SIZE = 10  # ms
ALIGNMENT = 'go_cue' or 'stop_cue'

# Processing
SMOOTH = True  # Gaussian smoothing
SIGMA = bin_size  # Same as bin size
TRUNCATE = 2  # ±2σ window
NORMALIZE = True  # Global normalization
SUCCESS_ONLY = True  # Exclude failed trials

# Visual
COLORMAP = 'Plasma'  # Yellow (high) to purple (low)
COLORBAR = True
TOOLS = ['hover']
SHOW_GRID = True
```

### Plasma Colormap Interpretation
- **Yellow/Bright**: High firing rate
- **Orange**: Moderate-high firing rate
- **Pink/Purple**: Moderate firing rate
- **Dark purple/Black**: Low/no firing rate

With global normalization:
- **1.0 (yellow)**: Highest firing rate across all cells
- **0.0 (dark purple)**: Lowest firing rate or no activity

### Grid Layout Guidelines

#### 3×2 Grid (Trial Type Comparison)
```
Row 1: GO-Left    | GO-Right
Row 2: STOP-Left  | STOP-Right
Row 3: CONT-Left  | CONT-Right

All use same cell ordering (from GO-left peaks)
```

#### 4×2 Grid (SSD Comparison)
```
Row 1: SSD1-Left  | SSD1-Right
Row 2: SSD2-Left  | SSD2-Right
Row 3: SSD3-Left  | SSD3-Right
Row 4: SSD4-Left  | SSD4-Right

All use same cell ordering (from GO-left peaks)
```

---

## Important Implementation Notes

### 1. Normalization Philosophy Change
**Old approach** (per-cell):
```python
firing_rate = (firing_rate - firing_rate.min()) / 
              (firing_rate.max() - firing_rate.min())
```
- Made all cells appear equally strong
- Lost relative magnitude information

**New approach** (global):
```python
psth_matrix = np.array(psth_list)
global_max = psth_matrix.max()
psth_matrix = psth_matrix / global_max
```
- Preserves relative firing rate differences
- Shows which cells are dominant contributors
- Critical for population analysis

### 2. Cell Ordering Consistency
**Key principle**: Use ONE reference condition for ordering
```python
# Example: Use GO-left for all plots
data_go = get_population_data_single_condition(
    trial_type='GO', direction=180, sort_by_peak=True
)
cell_order = data_go['cell_ids']

# Then reorder all other conditions to match
for other_condition in [stop_left, stop_right, cont_left, cont_right]:
    reorder_to_match(cell_order)
```

### 3. Alignment Point Selection

**GO trials**: Always use `go_cue`
- No secondary signal present
- `stop_cue` is NaN

**STOP trials**: Choose based on question
- `go_cue`: Compare initiation phase across conditions
- `stop_cue`: Focus on inhibition process (stop signal appears)
- Typical epoch with go_cue: [-200, 1000] to see stop signal effect
- Typical epoch with stop_cue: [-200, 700] to focus on inhibition

**CONT trials**: Choose based on question
- `go_cue`: Compare initiation phase across conditions
- `stop_cue`: Focus on continue signal processing (**green cue, not stop**)
- Note: `stop_cue` contains **continue signal time** for CONT trials
- Same epoch recommendations as STOP trials

### 4. SSD Effects on Analysis
```python
# GO trials: No SSD
trial_type='GO' → ssd_number=None (always)

# STOP/CONT: Must handle SSD
if trial_type in ['STOP', 'CONT']:
    if ssd_number is None:
        # Combine all SSDs
    else:
        # Filter specific SSD
        data = data[data['ssd_number'] == ssd_number]
```

### 5. Module Reloading During Development (UPDATED)
```python
# In Jupyter notebooks - reload both modules
import importlib
import session_class
import cell_analysis  # NEW: Cell class now in separate module

importlib.reload(cell_analysis)
importlib.reload(session_class)

from cell_analysis import Cell, PopulationAnalyzer
from session_class import Session

# Then recreate objects
session = Session(session_data)
cell = Cell(cell_data)
```

### 6. HoloViews Layout Management
```python
# Single plot
plot

# Horizontal layout (+ operator)
plot1 + plot2

# Vertical layout (.cols(1))
(plot1 + plot2 + plot3).cols(1)

# Grid layout
(plot1 + plot2 + plot3 + plot4).cols(2)  # 2×2 grid

# NdOverlay for multiple curves
overlay = hv.NdOverlay({
    'curve1': hv.Curve(data1),
    'curve2': hv.Curve(data2)
})
```

### 7. Data Dictionary Structure
```python
# Returned by get_population_data_* methods
{
    'bin_centers': np.array,      # Time points
    'psth_matrix': np.array,      # (n_cells × n_bins)
    'cell_ids': list,             # Ordered cell IDs
    'params': {                   # Metadata
        'epok': [-200, 700],
        'bin_size': 10,
        'alignment_point': 'go_cue',
        'trial_type': 'STOP',
        'direction': 0,
        'ssd_number': None,
        'success_only': True,
        'smooth': True,
        'normalize': True
    }
}
```

### 8. Data Quality and Filtering

**Trial Failure Validation**:
- STOP trial failures are validated by saccade amplitude
- Ensures that "failed to stop" means an actual saccade was executed
- Method: `update_stop_trial_failures()` in behavioral analysis
- Prevents false failures from noise or small eye movements

**Session Quality**:
- Not all recording sessions are suitable for analysis
- Some sessions excluded due to:
  - Recording artifacts
  - Insufficient trial counts
  - Equipment issues
  - Behavioral anomalies
- Always check for session exclusion lists before analysis

**Grade Filtering**:
- Cell quality rated 1-10 based on:
  - Spike waveform consistency
  - Signal-to-noise ratio
  - Unit isolation quality
- Standard threshold: Grade ≥ 8
- Lower grades may have contamination from other units

---

## Future Development Notes

### Immediate Priorities
1. **Validate Population Analyses**: Test with more sessions
2. **Error Handling**: Add robust error checking for edge cases
3. **Documentation**: Add docstrings to all methods
4. **Unit Tests**: Create test suite for core functionality

### Planned Enhancements
1. **Statistical Testing**: Add significance tests between conditions
2. **Clustering**: Group cells by response profiles
3. **Latency Analysis**: Measure response onset times
4. **Cross-correlation**: Population synchrony measures
5. **Dimensionality Reduction**: PCA/t-SNE on population responses
6. **Export**: Save plots and data in publication-ready formats

### Future Expansions (see Project Overview)
1. **Yasmin Data**: Integrate second subject (requires screen rotation handling)
2. **Cell Type Generalization**: Extend beyond MSN to other recorded cell types
3. **Multi-Subject Analysis**: Cross-subject comparisons and population statistics

### Known Limitations
1. **Current Scope**: Fiona only, MSN cells only, screen_rotation=0 only
2. **Normalization**: Global normalization may not be ideal for all analyses
3. **Peak-based Sorting**: Assumes unimodal responses
4. **Smoothing**: Gaussian smoothing parameters fixed (could be adaptive)
5. **Grid Layouts**: May be too small for many cells (>200)
6. **Session Exclusions**: Manual exclusion list (could be automated with quality metrics)

### Git Workflow
```bash
# Current branch
git branch  # cell_db

# Commit pattern
git add session_class.py session_analysis.ipynb
git commit -m "Add feature: description"

# Check status frequently
git status
```

---

## Quick Reference Card

### Most Common Operations
```python
# Load data
from cell_analysis import Cell, PopulationAnalyzer
from session_class import Session
cell_df = pd.read_pickle('data/unified_cell_trial_data/msn_fiona_cell_trial_data.pkl')

# Single cell (NEW: from cell_analysis.py)
cell_data = cell_df[cell_df['cell_ID'] == cell_id]
cell = Cell(cell_data, verbose=True)
cell.plot_psth_by_type_direction(smooth=True, delta=False, smooth_ker_size=25)
print(f"Baseline FR: {cell.baseline_FR:.2f} spikes/sec")

# Population
session = Session(cell_df[cell_df['trial_session'] == 'fi211110a'])
session.drop_cells_with_missing_trial_type_or_dir_data()
session.plot_trial_type_PSTH_comparison(epok_go=[-200, 700], bin_size=10)

# SSD analysis
session.plot_trial_type_PSTH_by_ssd(trial_type='STOP', epok_stop=[-200, 700])

# PCA analysis (NEW)
train, test = session.split_to_train_test(test_fraction=0.5, random_state=42)
# ... (see PCA Analysis section for full workflow)
```

### Common Epochs
- **Full trial**: [-500, 1500]
- **Standard**: [-200, 700]
- **Pre-stimulus**: [-200, 0]
- **Post-stimulus**: [0, 700]
- **PCA epoch**: [-50, 150] (from before go_cue to after movement onset - captures fixation end, target processing, reaction time, and movement initiation)

### Example Session: fi211110a
One of the best recording sessions with high cell count and good trial distribution.
- **Multiple cells recorded simultaneously**: Check session data for exact count
- **Rich multi-cell population data**: Ideal for population-level analyses
- **Good trial counts across all conditions**: Balanced for comparisons

**Note**: Exact statistics vary by filtering (grade threshold, alignment success, etc.)

---

## References

1. **Pani et al. (2022)**: Original methodology paper
2. **HoloViews Documentation**: http://holoviews.org/
3. **Bokeh Documentation**: https://docs.bokeh.org/
4. **SciPy ndimage**: Gaussian filtering documentation

---

**Last Updated**: November 2025
**Project Lead**: Barak
**AI Assistant**: Claude (Anthropic)
**Repository**: Population-Analysis (cell_db branch)

### Recent Changes (November 2025):
- **Cell class** moved from `analysis.ipynb` to `cell_analysis.py`
- Added **baseline firing rate** calculation and property
- Added **PCA analysis** workflow and methods
- Implemented **train/test split** for cross-validation
- Enhanced **normalization options** (by_max, by_baseline_FR)
- Added **delta parameter** for mean-centered PSTHs
- Configurable **smoothing kernel size** (smooth_ker_size)
- New **cell filtering methods** for data quality control
- Added **pytest unit tests** for Cell class
- Comprehensive **3D trajectory visualization** in PC space

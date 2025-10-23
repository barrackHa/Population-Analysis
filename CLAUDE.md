# MSN Population Analysis Project - Developer Reference

## Table of Contents
1. [Project Overview](#project-overview)
2. [Stop-Signal Task Structure](#stop-signal-task-structure)
3. [Data Structures](#data-structures)
4. [Architecture & Classes](#architecture--classes)
5. [File Organization](#file-organization)
6. [Key Methods Reference](#key-methods-reference)
7. [Workflow & Usage Patterns](#workflow--usage-patterns)
8. [Coding Guidelines](#coding-guidelines)
9. [Visualization Standards](#visualization-standards)
10. [Important Implementation Notes](#important-implementation-notes)

---

## Project Overview

This project analyzes neuronal recordings from the caudate nucleus during a countermanding stop-signal task (CSST), replicating and extending the methodology from **Pani et al. (2022)**: "Neuronal Activity in the Primate Caudate Nucleus Relates to Visual Salience and Action Selection During Stop-Signal Task".

### Current Scope
- **Cell Type**: Currently analyzing Medium Spiny Neurons (MSN), but architecture is designed to generalize to other cell types
- **Subject**: Fiona (Female Macaque, ~7-8 kg) - primary focus
- **Screen Configuration**: `screen_rotation = 0` (standard horizontal configuration)

### Database Statistics (Fiona - Grade ≥ 8 cells only)
- **Recording period**: Multiple sessions over several months
- **Total cell-trial combinations**: 844,694 rows
- **Recording sessions**: 59 sessions (3 sessions excluded due to data quality issues)
- **Quality threshold**: Grade ≥ 8 (scale 1-10, based on waveform quality and isolation)

**Excluded sessions** (Fiona):
- `fi210628`, `fi210629`, `fi210704` - excluded due to data quality issues

**Note**: The database contains cell-trial combinations, not unique trials. Each row represents one cell's activity during one trial, so multiple cells recorded simultaneously in the same trial create multiple rows.

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
1. **go_cue**: Visual target appears, signals saccade initiation (all trials)
2. **stop_cue**: Secondary signal appears (STOP/CONT trials only)
   - Red stop signal for STOP trials
   - Green continue signal for CONT trials
3. **first_relevant_saccade**: Actual saccade onset time (if executed)
4. **reaction_time**: Interval from `go_cue` to `first_relevant_saccade`

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
'first_relevant_saccade'  # Saccade onset time
'reaction_time'     # RT from go_cue to saccade

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
- **grade**: Quality metric for cell isolation. Threshold of ≥8 filters out poorly isolated units.

---

## Architecture & Classes

### Class Hierarchy

```
PopulationAnalyzer (analysis.ipynb)
├── Manages full cell database
└── Creates → Cell objects

Cell (analysis.ipynb)
├── Single neuron analysis
├── Trial-level operations
└── Individual cell visualizations

Session (session_class.py)
├── Session-level population analysis
├── Multi-cell population operations
└── Population visualizations
```

### Cell Class
**Location**: `population_analysis/analysis.ipynb`

**Purpose**: Single-cell analysis with trial-level visualizations

**Key Features**:
- Spike alignment to task events
- Trial filtering and sorting
- Raster plots (color-coded by condition)
- Histograms (spike counts)
- PSTH (firing rate with smoothing)

**Core Methods**:
```python
# Data operations
align_spikes_to_event(alignment_point)
filter_trials(trial_type, direction, ssd_number, success_only, failed_only)
aggregate_spikes_by_bins(epok, bin_size, ...)
calculate_psth(epok, bin_size, smooth=True, ...)

# Visualizations
plot_raster_by_type_direction(epok, alignment_point, show_legend)
plot_histogram_by_type_direction(epok, bin_size, separate_ssd, normalize)
plot_psth_by_type_direction(epok, bin_size, separate_ssd, smooth)
```

### Session Class
**Location**: `population_analysis/session_class.py`

**Purpose**: Population-level analysis across all cells in a session

**Architecture**: Separated data generation from plotting
- **Data methods**: Return dictionaries with results
- **Plot methods**: Accept pre-generated data or generate on-the-fly

**Key Features**:
- Global normalization (preserves relative cell differences)
- Peak-based cell ordering
- Multi-condition comparisons
- Grid layouts for comprehensive views

**Core Methods**:
```python
# Data generation methods
get_cell_psth(cell_id, epok, bin_size, alignment_point, trial_type, 
              direction, ssd_number, success_only, smooth)
get_all_cells_psth(epok, bin_size, ..., normalize=True)
get_population_data_single_condition(epok, bin_size, alignment_point, 
                                      trial_type, direction, ssd_number, 
                                      success_only, smooth, normalize, 
                                      sort_by_peak)
get_population_data_left_right(epok, bin_size, alignment_point, 
                                trial_type, success_only, smooth, normalize)
get_population_data_trial_types(epok_go, epok_stop, bin_size, direction, 
                                 ssd_number, success_only, smooth, normalize)

# Plotting methods
plot_population_heatmap(data, **kwargs)  # Single condition, 800×600px
plot_left_right_comparison(data, **kwargs)  # 2 columns, 400×600px each
plot_trial_type_comparison(data_left, data_right, **kwargs)  # 3×2 grid
plot_trial_type_by_ssd(trial_type, ssd_numbers, **kwargs)  # 4×2 grid
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
│   └── unified_cell_trial_data/
│       └── msn_fiona_cell_trial_data.pkl  # Main database
├── population_analysis/
│   ├── analysis.ipynb           # Cell class & single-cell analysis
│   ├── session_class.py         # Session class for population analysis
│   ├── session_analysis.ipynb   # Session class demonstrations
│   ├── maestro_file.py          # Maestro file parsing utilities
│   └── pre_proc_helper.py       # Preprocessing helpers
├── data_pre_proc/               # Data preprocessing notebooks
├── exploratory_data_analysis/   # EDA notebooks
└── simulation/                  # Neural analysis applications
```

### Key Files

#### 1. `session_class.py`
- **Purpose**: Session-level population analysis
- **Size**: ~780 lines
- **Main class**: `Session`
- **Dependencies**: pandas, numpy, holoviews, scipy.ndimage

#### 2. `session_analysis.ipynb`
- **Purpose**: Demonstration notebook for Session class
- **Sections**: 
  - Setup and imports
  - Example 1: Single condition heatmap
  - Example 2: Left vs right comparison
  - Example 3: Trial type comparison (3×2)
  - Example 4a: STOP trials by SSD (4×2)
  - Example 4b: CONT trials by SSD (4×2)

#### 3. `analysis.ipynb`
- **Purpose**: Cell class and single-cell analysis
- **Size**: ~2000+ lines
- **Main classes**: `Cell`, `PopulationAnalyzer`
- **Includes**: Comprehensive single-cell visualizations

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

### PSTH Calculation
```python
def calculate_psth(self, epok=[-200, 700], bin_size=10,
                   alignment_point='go_cue', trial_type=None, 
                   direction=None, ssd_number=None, 
                   success_only=True, smooth=True):
    """
    Calculate firing rate with Gaussian smoothing.
    
    Smoothing: sigma=bin_size, truncate=2
    - For bin_size=10ms: ±20ms smoothing window
    
    Returns: (bin_centers, firing_rate, n_trials)
    """
```

### Global Normalization
```python
def get_all_cells_psth(self, ..., normalize=True):
    """
    CRITICAL: Global normalization across all cells.
    
    Process:
    1. Collect all PSTHs without normalization
    2. Find global_max across entire population
    3. Divide all values by global_max
    
    Result: Preserves relative firing rate differences
    NOT per-cell [0,1] normalization
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

### 5. Module Reloading During Development
```python
# In Jupyter notebooks
import importlib
import session_class
importlib.reload(session_class)
from session_class import Session

# Then recreate objects
session = Session(session_data)
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
cell_df = pd.read_pickle('msn_fiona_cell_trial_data.pkl')

# Single cell
cell = Cell(cell_df[cell_df['cell_ID'] == cell_id])
cell.plot_psth_by_type_direction(smooth=True)

# Population
session = Session(cell_df[cell_df['trial_session'] == 'fi211110a'])
session.plot_trial_type_comparison(epok_go=[-200, 700], bin_size=10)

# SSD analysis
session.plot_trial_type_by_ssd(trial_type='STOP', epok_stop=[-200, 700])
```

### Common Epochs
- **Full trial**: [-500, 1500]
- **Standard**: [-200, 700]
- **Pre-stimulus**: [-200, 0]
- **Post-stimulus**: [0, 700]

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

**Last Updated**: October 2025  
**Project Lead**: Barak  
**AI Assistant**: Claude (Anthropic)  
**Repository**: Population-Analysis (cell_db branch)

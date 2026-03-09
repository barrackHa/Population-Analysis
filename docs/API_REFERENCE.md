# API Reference

## Architecture Overview

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

### File Organization

```
population_analysis/
├── cell_analysis.py         # Cell & PopulationAnalyzer classes
├── session_class.py         # Session class for population analysis
├── plot_PCA_in_3D.py        # 3D PCA trajectory plotting utilities
├── pca_helpers.py           # Worker functions for parallel PCA processing
├── session_PCA_analysis.ipynb  # Single-session PCA analysis workflow
├── multi_session_PCA_analysis.ipynb  # Multi-session PCA analysis
├── maestro_file.py          # Maestro file parsing utilities
└── pre_proc_helper.py       # Preprocessing helpers
```

---

## Cell Class

**Location**: `population_analysis/cell_analysis.py`

**Purpose**: Single-cell analysis with trial-level visualizations

### Key Features
- Spike alignment to task events
- Trial filtering and sorting
- Baseline firing rate property
- Raster plots (color-coded by condition)
- Histograms (spike counts)
- PSTH (firing rate with smoothing)
- Mean-centered firing rate (delta parameter)
- Configurable smoothing kernel size

### Properties

```python
baseline_FR  # Baseline firing rate (spikes/sec) calculated from -500 to 0 ms before go_cue
```

### Core Methods

#### Data Operations

```python
def align_spikes_to_event(self, alignment_point='go_cue'):
    """
    Align spike times to a specific event.

    Parameters:
    - alignment_point: 'go_cue', 'stop_cue', 'first_relevant_saccade'

    Creates new column: 'spikes_aligned_to_{alignment_point}'
    Returns: pd.Series of aligned spike times
    """

def filter_trials(trial_type, direction, ssd_number, success_only, failed_only):
    """
    Filter trials based on multiple criteria.

    Parameters:
    - trial_type: 'GO', 'STOP', 'CONT', or None
    - direction: 0, 180, or None
    - ssd_number: 1.0-4.0 or None
    - success_only: bool
    - failed_only: bool
    """

def aggregate_spikes_by_bins(epok, bin_size, normalize=False, ...):
    """
    Bin spike counts across trials.

    Returns: (bin_centers, spike_counts_per_bin, n_trials)
    """

def calculate_psth(self, epok=[-200, 700], bin_size=10,
                   alignment_point='go_cue', trial_type=None,
                   direction=None, ssd_number=None,
                   success_only=True, smooth=True,
                   delta=False, smooth_ker_size=25,
                   normalize_bins=False):
    """
    Calculate firing rate with Gaussian smoothing.

    Parameters:
    - epok: Time window [start, end] in ms
    - bin_size: Bin width in ms
    - alignment_point: 'go_cue', 'stop_cue', or 'first_relevant_saccade'
    - trial_type: 'GO', 'STOP', 'CONT', or None
    - direction: 0, 180, or None
    - ssd_number: 1.0-4.0 or None
    - success_only: Filter successful trials only
    - smooth: Apply Gaussian smoothing
    - delta: If True, subtract mean firing rate to center around zero
    - smooth_ker_size: Gaussian smoothing kernel size (default: 25)
    - normalize_bins: If True, z-score normalize spike counts before conversion

    Smoothing: sigma=smooth_ker_size, truncate=2
    - For smooth_ker_size=25ms: ±50ms smoothing window (2*sigma)
    - Smoothing applied BEFORE epoch trimming to avoid edge effects

    Returns: (bin_centers, firing_rate, n_trials)
    """
```

#### Visualization Methods

```python
def plot_raster_by_type_direction(epok, alignment_point, show_legend):
    """Plot raster plots grouped by trial type and direction."""

def plot_histogram_by_type_direction(epok, bin_size, separate_ssd, normalize):
    """Plot histograms of spike counts."""

def plot_psth_by_type_direction(epok, bin_size, separate_ssd, smooth,
                                 delta=False, smooth_ker_size=25,
                                 normalize_bins=False):
    """Plot PSTHs for all trial types and directions."""
```

---

## Session Class

**Location**: `population_analysis/session_class.py`

**Purpose**: Population-level analysis across all cells in a session

### Key Features
- Global normalization (preserves relative cell differences)
- Baseline FR normalization option
- Peak-based cell ordering
- Multi-condition comparisons
- Grid layouts for comprehensive views
- Cell filtering and validation
- Train/test data splitting for PCA

### Properties

```python
cells  # Generator yielding Cell instances for all cells in session
```

### Core Data Methods

#### Cell Access and Filtering

```python
def get_cell_data(cell_id):
    """Get DataFrame for specific cell."""

def get_cell(cell_id, verbose=False):
    """Get Cell instance for specific cell."""

def drop_cell_from_session(cell_id):
    """Remove cell from session."""

def drop_cells_with_missing_trial_type_or_dir_data(self):
    """
    Remove cells that lack data for all trial types or directions.

    Criteria:
    - Must have both 0° and 180° direction trials
    - Must have all 3 trial types (GO, STOP, CONT)

    Essential for PCA and population analysis to ensure
    balanced comparisons across conditions.
    """

def get_cells_with_no_spikes(as_percentage=False):
    """Count cells with no activity."""
```

#### PSTH Generation

```python
def get_cell_psth(cell_id, epok, bin_size, alignment_point, trial_type,
                  direction, ssd_number, success_only, smooth, delta=False,
                  smooth_ker_size=25, normalize_bins=False, normalize=False):
    """
    Get PSTH for a single cell.

    normalize options:
    - False: No normalization (raw firing rates)
    - True or 'by_max': Divide each cell by its max firing rate
    - 'by_baseline_FR': Subtract baseline firing rate from each cell
    """

def get_all_cells_psth(epok, bin_size, ..., normalize=False):
    """
    Get PSTHs for all cells in session.

    When normalize=False (default):
    - Preserves actual firing rate magnitudes
    - Use for PCA and quantitative comparisons

    When normalize=True:
    - Each cell normalized to [0, 1] range
    - Better for visualizing weak and strong cells together

    Result: Population matrix (n_cells × n_bins)
    """
```

#### Spike Counts (for PCA)

```python
def get_cell_spike_counts(cell_id, epok, bin_size, alignment_point, ...):
    """Get spike counts for a single cell."""

def get_all_cells_spike_counts(epok, bin_size, ..., normalize=True):
    """Get spike counts for all cells in session."""
```

#### Population Data Generation

```python
def get_population_spike_counts_data(epok, bin_size, alignment_point,
                                      trial_type, direction, ssd_number,
                                      success_only, normalize, sort_by_peak):
    """Generate population spike count matrix."""

def get_population_PSTH_single_condition(epok, bin_size, alignment_point,
                                          trial_type, direction, ssd_number,
                                          success_only, smooth, delta=False,
                                          smooth_ker_size=25, normalize_bins=False,
                                          normalize=False, sort_by_peak=True):
    """
    Generate PSTH matrix for a single condition.

    Sort cells by peak activity time (argmax).

    Process:
    1. Calculate PSTH for each cell
    2. Find time of peak firing (argmax)
    3. Sort cells in ascending order by peak time

    Maintains consistent ordering across related plots.
    """

def get_population_PSTHs_left_right(epok, bin_size, alignment_point,
                                    trial_type, success_only, smooth, delta=False,
                                    smooth_ker_size=25, normalize_bins=False,
                                    normalize=False, sort_by_peak=True):
    """Generate PSTH matrices for left and right directions."""

def get_population_PSTHs_trial_types(epok_go, epok_stop, bin_size, direction,
                                      ssd_number, success_only, smooth, normalize):
    """Generate PSTH matrices for GO, STOP, and CONT trial types."""
```

#### PCA Support

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

### Plotting Methods

```python
def plot_population_PSTH_heatmap(data, **kwargs):
    """Single condition heatmap, 800×600px."""

def plot_population_spike_counts_heatmap(data, **kwargs):
    """Spike counts heatmap."""

def plot_left_right_PSTH_comparison(data, **kwargs):
    """2 columns, 400×600px each."""

def plot_trial_type_PSTH_comparison(data_left, data_right, **kwargs):
    """3×2 grid comparing GO/STOP/CONT."""

def plot_trial_type_PSTH_by_ssd(trial_type, ssd_numbers, **kwargs):
    """4×2 grid showing SSD effects."""
```

---

## Module Reloading During Development

```python
# In Jupyter notebooks - reload both modules
import importlib
import session_class
import cell_analysis

importlib.reload(cell_analysis)
importlib.reload(session_class)

from cell_analysis import Cell, PopulationAnalyzer
from session_class import Session

# Then recreate objects
session = Session(session_data)
cell = Cell(cell_data)
```

---

**Related Documentation**:
- [Workflows & Usage Patterns](WORKFLOWS.md)
- [PCA Analysis Guide](PCA_GUIDE.md)
- [Coding Standards](STANDARDS.md)

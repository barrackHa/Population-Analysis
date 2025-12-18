# MSN Population Analysis Project - Developer Guide

**Quick 30-Second Pitch**: Analysis of neuronal recordings from the caudate nucleus during a countermanding stop-signal task (CSST), following **Pani et al. (2022)** methodology. Currently analyzing Medium Spiny Neurons (MSN) from Fiona (macaque), with ~845K cell-trial combinations across 59 sessions.

---

## Documentation Index

### Core Documentation
- **[Project Overview](docs/PROJECT.md)** - Scope, database statistics, future plans, limitations
- **[Stop-Signal Task Reference](docs/TASK_REFERENCE.md)** - Task structure, trial types, temporal events
- **[Data Structures](docs/DATA_REFERENCE.md)** - DataFrame structure, column descriptions, data quality

### Development Resources
- **[API Reference](docs/API_REFERENCE.md)** - Cell & Session classes, methods, parameters
- **[Workflows & Usage Patterns](docs/WORKFLOWS.md)** - Common analysis patterns with code examples
- **[PCA Analysis Guide](docs/PCA_GUIDE.md)** - Complete PCA workflow for population dynamics
- **[Coding & Visualization Standards](docs/STANDARDS.md)** - Guidelines, best practices, conventions

---

## Quick Start

**For Python use the local conda env at <project_dir>/.conda**

### Setup
```python
from cell_analysis import Cell, PopulationAnalyzer
from session_class import Session
import pandas as pd

# Load data
cell_df = pd.read_pickle('data/unified_cell_trial_data/msn_fiona_cell_trial_data.pkl')
```

### Single Cell Analysis
```python
# Get specific cell
cell_data = cell_df[cell_df['cell_ID'] == cell_id]
cell = Cell(cell_data, verbose=True)

# Visualize
cell.plot_psth_by_type_direction(smooth=True, delta=False, smooth_ker_size=25)
print(f"Baseline FR: {cell.baseline_FR:.2f} spikes/sec")
```

### Population Analysis
```python
# Create session
session = Session(cell_df[cell_df['trial_session'] == 'fi211110a'])
session.drop_cells_with_missing_trial_type_or_dir_data()

# Visualize population
session.plot_trial_type_PSTH_comparison(epok_go=[-200, 700], bin_size=10)
```

### PCA Analysis
```python
# Single-session PCA
train, test = session.split_to_train_test(test_fraction=0.5, random_state=42)

# Multi-session PCA
from multi_session_pca import MultiSessionPCA
analyzer = MultiSessionPCA(config)
analyzer.load_data('data/unified_cell_trial_data/msn_fiona_cell_trial_data.pkl')
analyzer.validate_all_sessions()
analyzer.extract_all_sessions_parallel(split_train_test=True, test_fraction=0.5, random_state=42)
analyzer.concatenate_sessions()
analyzer.subtract_average_PSTH()
analyzer.fit_and_project(n_components=5)
analyzer.plot_3d_trajectory()  # Shows test data by default

# See PCA_GUIDE.md for complete workflow
```

---

## Architecture Overview

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

MultiSessionPCA (multi_session_pca.py)
├── Multi-session population PCA
├── Parallel PSTH extraction across sessions
├── Train/test split support (optional)
├── PCA fitting and projection
└── 3D/2D trajectory visualizations
```

---

## Common Operations Cheat Sheet

### Data Loading
```python
# Main database
cell_df = pd.read_pickle('data/unified_cell_trial_data/msn_fiona_cell_trial_data.pkl')

# Select session
session_data = cell_df[cell_df['trial_session'] == 'fi211110a']
```

### Trial Filtering
```python
# Filter by trial type
go_trials = cell_df[cell_df['type'] == 'GO']
stop_trials = cell_df[cell_df['type'] == 'STOP']

# Filter by direction
right_trials = cell_df[cell_df['dir'] == 0]
left_trials = cell_df[cell_df['dir'] == 180]

# Filter by success
successful = cell_df[cell_df['trial_failed'] == False]
```

### Common Epochs
```python
# Standard analysis
epok = [-200, 700]

# PCA epoch (go_cue to movement onset)
epok = [-50, 150]

# Full trial
epok = [-500, 1500]
```

### Alignment Points
```python
# GO trials: always 'go_cue'
alignment_point = 'go_cue'

# STOP/CONT trials: 'go_cue' or 'stop_cue'
# - 'go_cue': Compare initiation across conditions
# - 'stop_cue': Focus on signal processing
alignment_point = 'stop_cue'
```

---

## Key Concepts

### Trial Types
- **GO**: No secondary signal, execute saccade (baseline)
- **STOP**: Red stop signal, inhibit saccade (test inhibitory control)
- **CONT**: Green continue signal, execute saccade (control condition)

### Directions
- **0°**: Right direction (rightward saccade)
- **180°**: Left direction (leftward saccade)

### Data Quality
- **Grade threshold**: ≤ 8 (filters poorly isolated units)
- **Excluded sessions**: fi210628, fi210629, fi210704
- **Success rates**: GO 96.6%, CONT 85.1%, STOP 54.5%

---

## File Organization

```
population_analysis/
├── CLAUDE.md (this file)
├── docs/
│   ├── PROJECT.md
│   ├── TASK_REFERENCE.md
│   ├── DATA_REFERENCE.md
│   ├── API_REFERENCE.md
│   ├── WORKFLOWS.md
│   ├── PCA_GUIDE.md
│   └── STANDARDS.md
├── data/
│   ├── unified_cell_trial_data/
│   │   └── msn_fiona_cell_trial_data.pkl
│   └── PCA_data/
├── population_analysis/
│   ├── cell_analysis.py          # Cell & PopulationAnalyzer classes
│   ├── session_class.py           # Session class
│   ├── multi_session_pca.py       # MultiSessionPCA class
│   ├── pca_helpers.py             # Parallel processing workers
│   ├── plot_PCA_in_3D.py
│   ├── session_PCA_analysis.ipynb
│   ├── multi_session_PCA_analysis.ipynb
│   └── ...
```

---

## Example Session: fi211110a

One of the best recording sessions for population analysis:
- Multiple cells recorded simultaneously
- Rich multi-cell population data
- Good trial counts across all conditions
- Ideal for population-level analyses and PCA

---

## Quick Links by Task

**Learning the project**:
1. Start with [Project Overview](docs/PROJECT.md)
2. Read [Stop-Signal Task Reference](docs/TASK_REFERENCE.md)
3. Review [Data Structures](docs/DATA_REFERENCE.md)

**Starting analysis**:
1. Check [Workflows & Usage Patterns](docs/WORKFLOWS.md)
2. Reference [API Reference](docs/API_REFERENCE.md) as needed
3. Follow [Coding Standards](docs/STANDARDS.md)

**Advanced analysis**:
1. Follow [PCA Analysis Guide](docs/PCA_GUIDE.md)
2. Review [Coding Standards](docs/STANDARDS.md) for best practices

---

**Last Updated**: November 2025
**Project Lead**: Barak
**AI Assistant**: Claude (Anthropic)
**Repository**: Population-Analysis (pca branch)
- to memorize that there's a local conda env you need to activate. From the root of the project run:
  ```bash
  conda activate $PWD/.conda
  ```
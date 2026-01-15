# Trajectory Divergence Analysis

**Purpose:** Detect when neural activity diverges between GO and STOP trials to understand the timing of inhibitory control signals in MSN neurons.

**Based on:** Pani et al. (2022) methodology for spike density function computation and trajectory analysis.

---

## Conceptual Overview

### The Question
In a countermanding stop-signal task, subjects must:
- **GO trials:** Execute a saccade when they see the go cue
- **STOP trials:** Inhibit the saccade when a stop signal appears after the go cue

**Key insight:** On STOP trials, the brain initially prepares the movement (like GO trials), but then must cancel it. The neural activity should start the same and then *diverge* when the stop signal is processed.

### The Approach
1. Compare neural activity between GO and STOP trials over time
2. Measure when and how much the trajectories diverge
3. The divergence point indicates when the stop signal affects the neuron

```
Time →
         GO cue                    Stop signal
            ↓                          ↓
GO trial:   ████████████████████████████████  (movement)
STOP trial: ████████████████░░░░░░░░░░░░░░░░  (inhibition)
                            ↑
                     Divergence point
```

---

## Analysis Pipeline

### Stage 1: Data Preparation

```
Raw Data → Filter GO trials → Validate trial counts → Ready for analysis
```

**GO Trial Filtering:**
- Remove GO trials where saccade occurs before `1.5 × mean_SSD`
- Rationale: These fast GO trials couldn't have been stopped anyway, so comparing them to STOP trials is unfair

### Stage 2: Spike Density Function (SDF)

Convert discrete spikes to continuous firing rate using the **Pani et al. kernel**:

```
K(t) = [1 - exp(-t/τg)] × exp(-t/τd)
```

- `τg = 1 ms` (growth time constant)
- `τd = 20 ms` (decay time constant)
- **Causal:** A spike only affects the SDF at its time and later (no backward influence)

```python
# Pseudocode
spike_train = bin_spikes(spike_times, dt=1ms)
kernel = pani_kernel(tau_g=1, tau_d=20)
sdf = convolve(spike_train, kernel)  # Causal convolution
```

### Stage 3: Rolling Window Distance

Compare two SDFs using **root mean squared error (RMSE)** in sliding windows:

```python
for each window position:
    distance[t] = sqrt(mean((sdf1[t:t+window] - sdf2[t:t+window])²))
```

- Window size: 10 ms
- Captures instantaneous difference between conditions

### Stage 4: Random Sampling

For each neuron:
1. **GO-GO pairs:** Sample 100 random pairs of GO trials → baseline variability
2. **GO-STOP pairs:** Sample 100 random pairs (one GO, one STOP) → divergence signal

```
GO-GO distance:   Measures inherent trial-to-trial variability
GO-STOP distance: Measures actual difference between conditions
Divergence:       |GO-STOP| - |GO-GO| = signal above baseline
```

### Stage 5: Population Analysis

1. Analyze each neuron independently (parallel processing)
2. Find each neuron's peak divergence time
3. Aggregate across population:
   - Population mean divergence curve
   - Distribution of peak times
   - Statistical tests

### Stage 6: Aligned PSTH Database

Create PSTHs aligned to each neuron's peak divergence time:
- Allows comparison of activity at the "moment of divergence"
- Normalizes for different peak times across neurons

---

## Code Architecture

### Files

```
trajectory_divergence/
├── divergence_analyzer.py    # Core analysis class
├── divergence_analysis.ipynb # Analysis notebook
└── README.md                 # This file
```

### Class: `TrajectoryDivergenceAnalyzer`

```python
analyzer = TrajectoryDivergenceAnalyzer(data, config={
    'epoch': [-50, 250],      # Analysis window (ms relative to go_cue)
    'window_size_ms': 10,     # Rolling window for distance
    'n_samples': 100,         # Trial pairs per neuron
    'direction': 0,           # 0=right, 180=left
    'ssd_number': 2.0,        # Which SSD level to analyze
    'n_jobs': -1,             # Parallel cores (-1 = all)
})

# Run analysis
analyzer.analyze_population()

# Access results
analyzer.results['divergence_population_mean']  # Population average
analyzer.results['peak_divergence_time']        # Per-neuron peaks
```

### Key Methods

| Method | Purpose |
|--------|---------|
| `pani_sdf_kernel()` | Create the Pani et al. causal kernel |
| `spikes_to_sdf()` | Convert spike times to SDF |
| `rolling_window_distance()` | Compute RMSE between two SDFs |
| `sample_trial_pairs_distances()` | Random sampling of trial pairs |
| `analyze_single_neuron()` | Full analysis for one neuron |
| `analyze_population()` | Parallel analysis across all neurons |
| `run_paired_ttest()` | Statistical comparison GO vs STOP |
| `create_aligned_psth_database()` | PSTHs aligned to peak divergence |

### Parallel Processing Optimization

**Problem:** Original code passed entire DataFrame to each worker process.

**Solution:** Pre-chunk data in main process, pass only relevant rows to workers.

```python
# Before (slow): Each worker filters 700K rows
results = Parallel(n_jobs=-1)(
    delayed(self.analyze_single_neuron)(cell_id, i)  # Has access to full self.data
    for cell_id in cell_ids
)

# After (fast): Workers receive ~500 rows each
cell_chunks = {cid: data[data['cell_ID'] == cid] for cid in cell_ids}
results = Parallel(n_jobs=-1)(
    delayed(_analyze_single_neuron_worker)(cell_chunks[cid], cid, config, i)
    for cell_id in cell_ids
)
```

---

## Interpretation Guide

### What the Results Mean

**Population Divergence Curve:**
- X-axis: Time relative to GO cue
- Y-axis: Distance between GO and STOP conditions (above baseline)
- Peak indicates when stop signal maximally affects the population

**Peak Divergence Times Histogram:**
- Shows distribution of when individual neurons diverge
- Spread indicates heterogeneity in timing across the population
- Mean/median indicate typical divergence latency

**Aligned PSTH:**
- All neurons aligned to their individual peak divergence time
- Shows what the activity looks like at the "moment of decision"
- GO vs STOP separation should be maximal at t=0

### Statistical Tests

**Paired t-test (GO vs STOP):**
- Compares mean firing rate in a window around peak divergence
- Significant difference confirms the divergence is real
- Cohen's d indicates effect size

---

## Usage Examples

### Basic Analysis

```python
from divergence_analyzer import TrajectoryDivergenceAnalyzer, DEFAULT_CONFIG
from cell_analysis import Cell
import pandas as pd

# Load data
data = pd.read_pickle('msn_fiona_cell_trial_data.pkl')

# Single session for testing
data = data[data['trial_session'] == 'fi211110a']

# Create analyzer
analyzer = TrajectoryDivergenceAnalyzer(data, config={
    'direction': 0,
    'ssd_number': 2.0,
})

# Run analysis
analyzer.analyze_population()

# View results
print(analyzer.summary())
```

### Accessing Results

```python
# Population-level
t = analyzer.results['t_centers']                    # Time axis
div = analyzer.results['divergence_population_mean'] # Mean divergence
peak_t = analyzer.results['population_peak_divergence_time']

# Per-neuron
for i, cell_id in enumerate(analyzer.results['cell_ids']):
    peak = analyzer.results['peak_divergence_time'][i]
    print(f"Cell {cell_id}: peak at {peak:.1f} ms")
```

### Create PSTH Database

```python
psth_db = analyzer.create_aligned_psth_database(
    cell_class=Cell,
    trial_types=['GO', 'STOP', 'CONT'],
)

# Access individual neuron
cell_psth = psth_db[psth_db['cell_id'] == some_cell_id]
```

---

## Parameters Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tau_g` | 1.0 | SDF kernel growth constant (ms) |
| `tau_d` | 20.0 | SDF kernel decay constant (ms) |
| `kernel_duration` | 20.0 | SDF kernel length (ms) |
| `dt` | 1.0 | Time resolution (ms) |
| `epoch` | [-50, 250] | Analysis window relative to go_cue |
| `window_size_ms` | 10 | Rolling window for distance |
| `n_samples` | 100 | Trial pairs per condition |
| `direction` | 0 | Saccade direction (0=right, 180=left) |
| `ssd_number` | 2.0 | SSD level (1.0-4.0) |
| `min_trials` | 5 | Minimum trials required per condition |
| `n_jobs` | -1 | Parallel workers (-1 = all cores) |
| `random_state` | 42 | Seed for reproducibility |

---

## Notes and Caveats

### Methodological Choices

1. **Why RMSE over Euclidean distance?**
   - RMSE normalizes by window size, making it comparable across different window sizes
   - Original code used RMSE: `sqrt(mean((x-y)²))` not `sqrt(sum((x-y)²))`

2. **Why absolute divergence?**
   - We use `|GO-STOP - GO-GO|` to capture magnitude regardless of sign
   - Some neurons may increase, others decrease activity on STOP trials

3. **Why random sampling?**
   - Avoids bias from trial order
   - Provides confidence intervals through multiple samples
   - Computationally tractable for large datasets

### Limitations

1. **Single direction analysis:** Each run analyzes one direction only
2. **Fixed SSD:** Analyzes one SSD level at a time
3. **Assumes stationarity:** Doesn't account for within-session drift

### Future Extensions

- [ ] Multi-direction analysis in single run
- [ ] SSD comparison (early vs late stop signals)
- [ ] Sliding window peak detection
- [ ] Bootstrap confidence intervals
- [ ] Cluster-based permutation tests

---

## Version History

- **January 2026:** Initial implementation with parallel processing optimization
- Refactored from exploratory notebook `single_cell_divergence_analysis.ipynb`

---

## References

- Pani, P., et al. (2022). Methods for spike density function computation and analysis.
- Original exploration: `single_cell_divergence_analysis.ipynb`

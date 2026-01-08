# Neural Trajectory Divergence Analysis - Session Summary

**Date**: January 2026
**Notebook**: `exploratory_neural_divergence_analysis.ipynb`
**Objective**: Detect when STOP trial neural trajectories diverge from GO baseline

---

## What We Accomplished

### 1. Code Organization & Refactoring
- **Imported MultiSessionPCA** at the top with other classes for consistency
- **Refactored filter functions**: Created `filter_trials()` as base function, `get_trial_numbers()` calls it
- **Created reusable heatmap function**: `plot_neuron_trial_heatmap()` to visualize neuron-trial recording patterns
- **Added tuning analysis functions**: `test_cell_tuning()` and `compute_session_tuning_stats()` using ANOVA

### 2. Session Selection & Analysis
- **Found optimal session**: `fi211110a` with 84 neurons (vs original fi211115a with 53)
- **Optimal direction**: Left (180°) with 18 STOP SSD2 trials
- **Recording density**: 62.04% - measures how consistently neurons are recorded across trials

### 3. Neural Tuning Integration
Implemented ANOVA-based tuning tests from `multi_session_tuning_analysis.ipynb`:
- **Task modulation**: Tests if neurons change activity from baseline to GO period
- **Direction tuning**: Tests if neurons distinguish left vs right movements
- **Analysis windows**:
  - Baseline: [-500, 0] ms relative to go_cue
  - Task: [0, 500] ms relative to go_cue
  - Direction: [0, 500] ms relative to go_cue

### 4. Quality Score Metric
Created composite metric to identify best sessions:
```
quality_score = expected_density × pct_tuned
where:
  expected_density = recording_density × n_neurons
  pct_tuned = percentage of neurons showing task/direction tuning
```

**Why this matters**: A million unrelated neurons with 100% recording density is useless. We need neurons that are **task-relevant** (tuned) AND **consistently recorded**.

---

## Current Best Configuration

Based on quality score analysis:

| Metric | Value |
|--------|-------|
| Session | fi211110a |
| Direction | Left (180°) |
| Neurons | 84 |
| STOP SSD2 Trials | 18 |
| Recording Density | 62.04% |
| Expected Density | 52.11 |
| Tuned Neurons | TBD (run cell-23) |
| Quality Score | TBD (run cell-23) |

---

## How Things Should Work

### 1. Session Selection Priority
When choosing a session for trajectory divergence analysis, prioritize:
1. **Quality Score** (expected_density × pct_tuned) - HIGHEST PRIORITY
2. Expected density (n_neurons × recording_density)
3. Number of neurons
4. Number of trials

**Never** select based on neuron count alone - untuned neurons add noise.

### 2. Code Organization Principles
- **No code duplication**: Use functions, not copy-paste
- **Modular design**: Separate filtering, computing, and plotting
- **Consistent imports**: All classes imported/reloaded at the top
- **Reusable functions**: Parameters for flexibility, not hard-coded values

### 3. Analysis Workflow
```
1. Load data → Session object
2. Filter trials (STOP SSD2, specific direction)
3. Check recording density (heatmap visualization)
4. Test neural tuning (ANOVA)
5. Compute quality score
6. Proceed with trajectory analysis if quality is sufficient
```

### 4. Key Functions Reference

#### Trial Filtering
```python
filter_trials(session_data, trial_type, direction, ssd_number, success_only)
# Returns: Filtered DataFrame

get_trial_numbers(session_data, trial_type, direction, ssd_number, success_only)
# Returns: Sorted list of unique trial numbers
```

#### Visualization
```python
plot_neuron_trial_heatmap(filtered_data, session_id, trial_type, direction,
                          ssd_number, dir_map, figsize, show)
# Returns: (fig, ax, recording_matrix, summary_stats)
```

#### Recording Density
```python
compute_session_recording_density(session_df, trial_type, direction,
                                  ssd_number, success_only)
# Returns: dict with n_neurons, n_trials, recording_density, expected_density
```

#### Neural Tuning
```python
test_cell_tuning(cell, alpha)
# Returns: dict with task_modulated, direction_tuned, p-values

compute_session_tuning_stats(session, alpha)
# Returns: dict with n_tuned, pct_task_modulated, pct_direction_tuned, etc.
```

---

## Next Steps

### Immediate
1. **Run cell-23** to get tuning results and final quality scores
2. **Update session parameters** (cell-5) based on best quality score
3. **Generate heatmap** for the optimal session/direction combination

### Analysis Pipeline
1. Extract neural trajectories for GO and STOP trials
2. Project onto low-dimensional space (PCA or similar)
3. Implement divergence detection using:
   - Statistical tests (e.g., Poisson-based)
   - Distance metrics between trajectories
   - Sliding window analysis
4. Visualize divergence time course

### Future Improvements
- Consider multiple sessions if quality scores are similar
- Test different SSD levels (not just SSD2)
- Add bootstrap analysis for robust divergence estimates
- Compare results with/without tuning filter

---

## References

- **ANOVA methodology**: `docs/ANOVA/TUNING_ANALYSIS_README.md`
- **MultiSessionPCA**: `population_analysis/multi_session_pca.py`
- **Tuning analysis**: `population_analysis/multi_session_tuning_analysis.ipynb`
- **Task reference**: `docs/TASK_REFERENCE.md`
- **Data structures**: `docs/DATA_REFERENCE.md`

---

## Important Parameters

```python
# Session selection
session_id = 'fi211110a'  # Best session with 84 cells
DIRECTION = 180  # Left direction (more STOP trials)
SSD_NUM = 2  # Signal delay level

# Analysis windows
EPOK = [-300, 300]  # Overall epoch for trajectory analysis
BIN_SIZE = 50  # Bin width in ms
BIN_STEP = 10  # Sliding window step in ms

# Tuning windows (ANOVA)
TUNING_WINDOWS = {
    'baseline': [-500, 0],
    'task': [0, 500],
    'direction': [0, 500]
}

# Statistical thresholds
alpha = 0.05  # Significance level for tuning tests
FR_MIN = 1.0  # Minimum firing rate (Hz)
```

---

**Status**: ✓ Setup complete, ready for trajectory divergence analysis
**Last Updated**: January 2026

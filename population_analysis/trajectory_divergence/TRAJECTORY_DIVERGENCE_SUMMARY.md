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
- **Found optimal session**: `fi211025a` with 63 neurons (previously considered fi211110a with 84)
- **Optimal direction**: Left (180°) with 11 STOP SSD2 trials
- **Recording density**: 77.92% - measures how consistently neurons are recorded across trials
- **Quality score**: 38.96 (best among all sessions) - combines recording density with neural tuning percentage

### 3. Neural Tuning Integration
Implemented ANOVA-based tuning tests from `multi_session_tuning_analysis.ipynb`:
- **Task modulation**: Tests if neurons change activity from baseline to GO period
- **Direction tuning**: Tests if neurons distinguish left vs right movements
- **STOP sensitivity (original)**: Compares GO [100-300ms] vs STOP [0-200ms from stop_cue]
- **STOP sensitivity (matched)**: Compares GO [0-mean_SSD] vs STOP [0-mean_SSD from stop_cue]
- **Analysis windows**:
  - Baseline: [-200, 0] ms relative to go_cue
  - Task: [0, 500] ms relative to go_cue
  - Direction: [0, 500] ms relative to go_cue
  - STOP original: GO [100, 300] from go_cue vs STOP [0, 200] from stop_cue
  - STOP matched: GO [0, mean_SSD] from go_cue vs STOP [0, mean_SSD] from stop_cue

**Results for fi211025a (Left, SSD2)**:
- Task-modulated: 35 neurons (55.6%)
- Direction-tuned: 23 neurons (36.5%)
- STOP-sensitive (original): 10 neurons (15.9%)
- STOP-sensitive (matched): 14 neurons (22.2%)
- **Total tuned**: 50 neurons (79.4%)

### 4. Quality Score Metric
Created composite metric to identify best sessions:
```
quality_score = expected_density × pct_tuned
where:
  expected_density = recording_density × n_neurons
  pct_tuned = percentage of neurons showing ANY tuning (task/direction/STOP)
```

**Why this matters**: A million unrelated neurons with 100% recording density is useless. We need neurons that are **task-relevant** (tuned) AND **consistently recorded**.

### 5. STOP-Sensitive Neuron Visualization
Added comprehensive visualization strategy for neurons that respond to STOP signals:
- Uses `Cell.plot_psth_by_type_direction()` method
- Shows all 4 conditions on same graph: GO left, GO right, STOP left, STOP right
- Epoch: [-100, 400]ms from go_cue to cover tuning test windows
- Bin size: 1ms for fine temporal resolution
- Gaussian smoothing with 25ms kernel

### 6. Heatmap Ordering Enhancement
Updated `plot_neuron_trial_heatmap()` to reveal dense recording blocks:
- Trials ordered by neuron count (most → least)
- Neurons ordered by trial count (most → least)
- Returns ordered lists: `ordered_trial_numbers`, `ordered_cell_ids`
- Makes it easy to identify which neuron-trial combinations provide best coverage

---

## Key Insights from This Session

### Why fi211025a Beats fi211110a (Despite Fewer Neurons)
Although fi211110a has 84 neurons vs fi211025a's 63, **fi211025a is superior** because:

1. **Higher Recording Density**: 77.92% vs 62.04%
   - More consistent recordings across neurons and trials
   - Less missing data to interpolate or exclude

2. **Higher Tuning Percentage**: 79.4% vs 63.1%
   - More task-relevant neurons
   - Less noise from unresponsive neurons

3. **Quality Score Wins**: 38.96 vs 32.88
   - Better combination of density and relevance
   - Quality over quantity for trajectory analysis

**Lesson**: More neurons ≠ better session. Task-relevant neurons with consistent recording are what matter.

### STOP Sensitivity Testing - Major Methodological Advance
Two complementary tests detect different aspects of STOP processing:

1. **Original Test** (GO [100-300ms] vs STOP [0-200ms from stop_cue]):
   - Captures behavioral state differences
   - Compares movement preparation vs inhibitory response
   - Found 10 neurons (15.9%) in fi211025a

2. **Matched Test** (GO [0-mean_SSD] vs STOP [0-mean_SSD from respective cues]):
   - Time-matched comparison
   - Isolates STOP signal processing
   - Found 14 neurons (22.2%) in fi211025a

**Why Both?** The original test is sensitive to behavioral differences. The matched test isolates the neural signature of the STOP signal itself. Together they provide comprehensive characterization.

---

## Current Best Configuration

Based on quality score analysis:

| Metric | Value |
|--------|-------|
| Session | fi211025a |
| Direction | Left (180°) |
| Neurons | 63 |
| STOP SSD2 Trials | 11 |
| Recording Density | 77.92% |
| Expected Density | 49.09 |
| Tuned Neurons | 50 (79.4%) |
| Task-Modulated | 35 (55.6%) |
| Direction-Tuned | 23 (36.5%) |
| STOP-Sensitive (Original) | 10 (15.9%) |
| STOP-Sensitive (Matched) | 14 (22.2%) |
| **Quality Score** | **38.96** |

**Second Best**: fi211110a-Left (Quality: 32.88, Neurons: 84, Trials: 18, Density: 62.04%, Tuned: 63.1%)

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
# Returns: (fig, ax, recording_matrix, summary_stats, ordered_trial_numbers, ordered_cell_ids)
# Note: Matrix and lists are ordered by coverage (most to least) to reveal dense recording blocks
```

#### Recording Density
```python
compute_session_recording_density(session_df, trial_type, direction,
                                  ssd_number, success_only)
# Returns: dict with n_neurons, n_trials, recording_density, expected_density
```

#### Neural Tuning
```python
test_cell_tuning(cell, alpha, ssd_number)
# Returns: dict with:
#   - task_modulated, direction_tuned (bool)
#   - stop_sensitive_original, stop_sensitive_matched (bool)
#   - task_p, direction_p, stop_original_p, stop_matched_p (float)
#   - is_tuned (bool, True if ANY tuning type is significant)

compute_session_tuning_stats(session, alpha, ssd_number)
# Returns: dict with:
#   - n_tuned, pct_tuned
#   - n_task_modulated, pct_task_modulated
#   - n_direction_tuned, pct_direction_tuned
#   - n_stop_sensitive_original, pct_stop_sensitive_original
#   - n_stop_sensitive_matched, pct_stop_sensitive_matched
```

---

## Next Steps

### Immediate ✓ COMPLETED
1. ✓ Run cell-26 to get tuning results and final quality scores
2. ✓ Identified best session: fi211025a-Left with quality score 38.96
3. ✓ Generated heatmap for the optimal session/direction combination
4. ✓ Implemented STOP sensitivity testing (2 methods)
5. ✓ Created STOP-sensitive neuron visualizations

### Ready for Trajectory Analysis
Session fi211025a is now fully characterized and ready for divergence detection

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
session_id = 'fi211025a'  # Best session with 63 cells, quality score 38.96
DIRECTION = 180  # Left direction (180°)
SSD_NUM = 2  # Signal delay level

# Analysis windows
EPOK = [-300, 300]  # Overall epoch for trajectory analysis
BIN_SIZE = 50  # Bin width in ms
BIN_STEP = 10  # Sliding window step in ms

# Tuning windows (ANOVA)
TUNING_WINDOWS = {
    'baseline': [-200, 0],  # Updated from [-500, 0]
    'task': [0, 500],
    'direction': [0, 500],
    'stop_vs_go_original': {
        'go': [100, 300],    # GO: 100-300ms post go-cue
        'stop': [0, 200]     # STOP: 0-200ms post stop-cue
    },
    'stop_vs_go_matched': {
        'go': [0, 'mean_SSD'],    # GO: 0-mean_SSD from go-cue (~108ms for fi211025a)
        'stop': [0, 'mean_SSD']   # STOP: 0-mean_SSD from stop-cue
    }
}

# Visualization parameters
PSTH_BIN_SIZE = 1  # Fine temporal resolution for PSTH plots
SMOOTH_KERNEL = 25  # Gaussian smoothing kernel size (ms)

# Statistical thresholds
alpha = 0.05  # Significance level for tuning tests
FR_MIN = 1.0  # Minimum firing rate (Hz)
```

---

**Status**: ✓ Exploratory analysis complete - Best session identified (fi211025a-Left, QS: 38.96)
**Last Updated**: January 11, 2026

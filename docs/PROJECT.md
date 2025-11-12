# Project Overview

This project analyzes neuronal recordings from the caudate nucleus during a countermanding stop-signal task (CSST), replicating and extending the methodology from **Pani et al. (2022)**: "Neuronal Activity in the Primate Caudate Nucleus Relates to Visual Salience and Action Selection During Stop-Signal Task".

## Current Scope

- **Cell Type**: Currently analyzing Medium Spiny Neurons (MSN), but architecture is designed to generalize to other cell types
- **Subject**: Fiona (Female Macaque, ~7-8 kg) - primary focus
- **Screen Configuration**: `screen_rotation = 0` (standard horizontal configuration). For Yasmin, in some trials `screen_rotation ≠ 0`, coordinate transformations will be needed.

## Database Statistics (Fiona - Grade <= 8 cells only)

- **Recording period**: Multiple sessions over several months
- **Total cell-trial combinations**: 844,694 rows
- **Recording sessions**: 59 sessions (3 sessions excluded due to data quality issues)
- **Quality threshold**: Grade <= 8 (scale 5-11, based on waveform quality and isolation)

**Excluded sessions** (Fiona):
- `fi210628`, `fi210629`, `fi210704` - excluded due to data quality issues

**Note**: The 'msn_{monkey}_cell_trial_data.pkl' database contains cell-trial combinations, not unique trials. Each row represents one cell's activity during one trial, so multiple cells recorded simultaneously in the same trial create multiple rows.

## Trial Distribution & Performance

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

## Data Quality Control

**Saccade Amplitude Filtering**:
- Trial failures for STOP trials are validated by saccade amplitude
- Method: `update_stop_trial_failures()` recalculates failures based on whether saccade amplitude exceeds threshold
- This ensures that "failed" STOP trials actually had executed saccades, not just noise

**Session Exclusions**:
- Some sessions excluded due to data quality issues (e.g., recording artifacts, insufficient trials)
- Fiona excluded sessions: `fi210628`, `fi210629`, `fi210704`

---

# Future Development & Scope

## Planned Expansions

### 1. Yasmin Data Integration

- **Status**: Data collected but not yet analyzed
- **Challenge**: Yasmin has `screen_rotation ≠ 0` (rotated display configuration)
- **Blocker**: Need to develop coordinate transformation methods to align with Fiona's standard orientation
- **Impact**: Will require updates to:
  - Direction encoding (currently 0° and 180°)
  - Saccade direction calculations
  - Target position mappings

### 2. Cell Type Generalization

- **Current**: MSN (Medium Spiny Neurons) only
- **Future**: Extend to other cell types recorded simultaneously
- **Architecture**: Code is already designed to be cell-type agnostic
- **Required Changes**:
  - Cell type filtering options in analysis classes
  - Cell type-specific visualization palettes
  - Comparative analyses across cell types

### 3. Multi-Subject Comparisons

- Once Yasmin data is integrated:
  - Cross-subject population comparisons
  - Individual difference analyses
  - Subject as factor in statistical models

## Known Limitations

1. **Current Scope**: Fiona only, MSN cells only, screen_rotation=0 only
2. **Normalization**: Global normalization may not be ideal for all analyses
3. **Peak-based Sorting**: Assumes unimodal responses
4. **Smoothing**: Gaussian smoothing parameters fixed (could be adaptive)
5. **Grid Layouts**: May be too small for many cells (>200)
6. **Session Exclusions**: Manual exclusion list (could be automated with quality metrics)

## Planned Enhancements

1. **Statistical Testing**: Add significance tests between conditions
2. **Clustering**: Group cells by response profiles
3. **Latency Analysis**: Measure response onset times
4. **Cross-correlation**: Population synchrony measures
5. **Dimensionality Reduction**: PCA/t-SNE on population responses
6. **Export**: Save plots and data in publication-ready formats

---

**Related Documentation**:
- [Stop-Signal Task Reference](TASK_REFERENCE.md)
- [Data Structures](DATA_REFERENCE.md)
- [API Reference](API_REFERENCE.md)

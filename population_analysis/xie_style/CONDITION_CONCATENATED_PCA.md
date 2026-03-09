# Condition-Concatenated PCA Analysis

## Overview

This analysis creates a PCA representation where each neuron is characterized by concatenating its responses across six task conditions (3 trial types × 2 directions) into a single feature vector.

## Methodology

### Data Matrix Construction

Each neuron (row) is represented by a concatenated vector of PSTHs from six conditions:

1. **GO Left trials** (180°): Aligned to go_cue, epoch [50, 300] ms (250 bins)
2. **GO Right trials** (0°): Aligned to go_cue, epoch [50, 300] ms (250 bins)
3. **STOP Left trials** (180°): Aligned to stop_cue, epoch [-50, 200] ms (250 bins)
4. **STOP Right trials** (0°): Aligned to stop_cue, epoch [-50, 200] ms (250 bins)
5. **CONT Left trials** (180°): Aligned to stop_cue, epoch [-50, 200] ms (250 bins)
6. **CONT Right trials** (0°): Aligned to stop_cue, epoch [-50, 200] ms (250 bins)

**Total features per neuron**: 1500 (6 conditions × 250 bins)

### Key Parameters

- **Bin size**: 1 ms
- **Smoothing**: 25 ms Gaussian kernel
- **Trial selection**: Successful trials only
- **SSD selection**: None (uses all SSDs) - applies only to STOP/CONT trials
- **Normalization**: Mean-centering per neuron (subtract mean) applied to the concatenated vector
- **PCA method**: Configurable (PCA or TruncatedSVD)
- **Components**: 5 (visualizing top 3)
- **Minimum cells per session**: 15
- **Excluded sessions**: fi210628, fi210629, fi210704

### Analysis Pipeline

```
1. Load data → Filter to successful trials → Filter sessions with ≥15 cells
2. For each neuron:
   - Extract GO Left PSTH (50-300 ms from go_cue, dir=180°, ssd_number=None)
   - Extract GO Right PSTH (50-300 ms from go_cue, dir=0°, ssd_number=None)
   - Extract STOP Left PSTH (-50-200 ms from stop_cue, dir=180°, ssd_number=config)
   - Extract STOP Right PSTH (-50-200 ms from stop_cue, dir=0°, ssd_number=config)
   - Extract CONT Left PSTH (-50-200 ms from stop_cue, dir=180°, ssd_number=config)
   - Extract CONT Right PSTH (-50-200 ms from stop_cue, dir=0°, ssd_number=config)
   - Concatenate: [GO_L | GO_R | STOP_L | STOP_R | CONT_L | CONT_R]
   - Mean-center the concatenated vector (subtract mean)
3. Remove neurons missing data in any of the 6 conditions
4. Fit PCA on normalized data matrix (transposed: features × neurons)
5. Extract condition trajectories by splitting PC components
   - Each PC component is 1500-dimensional
   - Split into 6 portions: GO_L(250) | GO_R(250) | STOP_L(250) | STOP_R(250) | CONT_L(250) | CONT_R(250)
6. Visualize trajectories in 3D and 2D
```

**Important Implementation Details**:
- **SSD filtering**: GO trials don't have SSDs, so `ssd_number=None` for GO, but `ssd_number` from config for STOP/CONT
- **JSON serialization**: Config contains class objects (e.g., PCA), handled by `make_json_serializable()` helper
- **Typical results**: ~1212 neurons with all 6 conditions (from 1213 total)

## Key Differences from Other Approaches

| Aspect | Xie-Style Regression | Multi-Session PCA | This Approach |
|--------|---------------------|-------------------|---------------|
| **Input** | Single-trial spike counts | Trial-averaged PSTHs | Trial-averaged PSTHs |
| **Method** | Sparse regression → PCA on β | Direct PCA on PSTHs | PCA on concatenated PSTHs |
| **Time structure** | Single window | Full time-series | Multiple windows (per condition) |
| **Normalization** | None (Lasso handles scale) | Optional | Mean-centering per neuron |
| **Feature space** | Task regressors | Time × Conditions | Time × Conditions (concatenated) |
| **Interpretability** | Feature importance | Temporal dynamics | Condition-specific dynamics |

## Rationale

This approach allows the PCA to discover components that capture:
- **Within-condition dynamics**: How activity evolves during GO, STOP, or CONT
- **Between-condition structure**: Relationships between neural responses across conditions
- **Neuron-specific profiles**: Each neuron contributes based on its full response pattern

By mean-centering each neuron's concatenated vector:
- Each neuron's response is normalized relative to its own average across all conditions
- PCA focuses on deviations from mean activity rather than absolute rates
- Preserves relative magnitude differences between neurons while removing baseline offsets

## Outputs

### Saved Data

Location: `../../data/condition_concatenated_pca/`

- `X_raw.npy`: Raw PSTH data matrix (n_cells × 1500)
- `X_normalized.npy`: Mean-centered data matrix (n_cells × 1500)
- `pca_components.npy`: PC loading vectors (n_components × 1500)
- `explained_variance_ratio.npy`: Variance explained by each PC
- `go_left_trajectory.npy`: GO Left trajectory in PC space (n_components × 250)
- `go_right_trajectory.npy`: GO Right trajectory in PC space (n_components × 250)
- `stop_left_trajectory.npy`: STOP Left trajectory in PC space (n_components × 250)
- `stop_right_trajectory.npy`: STOP Right trajectory in PC space (n_components × 250)
- `cont_left_trajectory.npy`: CONT Left trajectory in PC space (n_components × 250)
- `cont_right_trajectory.npy`: CONT Right trajectory in PC space (n_components × 250)
- `cell_ids.npy`: Cell identifiers for neurons included in analysis
- `metadata.json`: Analysis parameters, summary statistics, and condition indices

**Note**: Default `n_components = 5`, configurable in the notebook

### Typical Results

**Dataset statistics** (with default parameters):
- Total neurons processed: ~1213
- Neurons with all 6 conditions: ~1212
- Neurons excluded (missing conditions): ~1
- Final data matrix shape: (1212, 1500)

**Trial counts per neuron** (successful trials, for neurons with all conditions):
- GO_L: mean ~150, min ~6, max ~264
- GO_R: mean ~150, min ~13, max ~259
- STOP_L: mean ~28, min ~1, max ~64
- STOP_R: mean ~36, min ~1, max ~73
- CONT_L: mean ~58, min ~3, max ~106
- CONT_R: mean ~54, min ~2, max ~103

**Variance explained** (5 components):
- PC1: ~27%
- PC2: ~25%
- PC3: ~10%
- PC4: ~5%
- PC5: ~5%
- Total: ~73%

### Visualizations

1. **3D trajectory plot**: Neural trajectories in PC1-PC2-PC3 space
2. **2D projection grid**: All pairwise PC projections
3. **PC time series**: Evolution of each PC over time for each condition

## Usage

```bash
cd population_analysis/xie_style
conda activate $PWD/../../.conda
jupyter notebook condition_concatenated_pca.ipynb
```

## Interpretation Guide

### PC Loadings

Each PC is a 1500-dimensional vector with six segments:
- Bins 0-249: Contribution from GO Left epoch
- Bins 250-499: Contribution from GO Right epoch
- Bins 500-749: Contribution from STOP Left epoch
- Bins 750-999: Contribution from STOP Right epoch
- Bins 1000-1249: Contribution from CONT Left epoch
- Bins 1250-1499: Contribution from CONT Right epoch

Inspect `pca_components.npy` to see which time points in which conditions drive each PC.

### Trajectories

The trajectories represent how each PC varies over time within each condition:
- **GO Left/Right trajectories**: PC loadings for respective GO bins (aligned to go_cue, 50-300ms)
- **STOP Left/Right trajectories**: PC loadings for respective STOP bins (aligned to stop_cue, -50-200ms)
- **CONT Left/Right trajectories**: PC loadings for respective CONT bins (aligned to stop_cue, -50-200ms)

Each trajectory shows the "temporal signature" of a PC within that condition - i.e., which time points in that condition contribute most to that PC.

**Note**: These are PC component loadings (in feature space), not population state projections. They show the pattern learned by each PC, split by condition and direction.
**Alignment**: GO conditions aligned to go_cue [50-300ms], STOP/CONT conditions aligned to stop_cue [-50-200ms]

## Common Issues & Solutions

### Issue 1: Zero neurons with all 6 conditions

**Symptom**: Output shows "Neurons with all 6 conditions: 0"

**Cause**: The `ssd_number` parameter was being passed to GO trials, but GO trials don't have stop signal delays. This filtered out all GO trials.

**Solution**: Modified `extract_neuron_psths_all_conditions()` to only pass `ssd_number` for STOP and CONT trials:
```python
# Only pass ssd_number for STOP and CONT trials (GO trials don't have SSDs)
ssd_param = config['ssd_number'] if trial_type in ['STOP', 'CONT'] else None
```

### Issue 2: JSON serialization error when saving metadata

**Symptom**: `TypeError: Object of type ABCMeta is not JSON serializable`

**Cause**: The `config` dictionary contains the `PCA_function` parameter which is a class object (not a string).

**Solution**: Added `make_json_serializable()` helper function that converts class objects to their name strings:
```python
def make_json_serializable(obj):
    """Convert non-serializable objects to serializable ones."""
    if isinstance(obj, type):  # Handle class objects like PCA
        return obj.__name__
    # ... handle other types ...
```

### Questions to Ask

1. **Direction selectivity**: Do left vs. right trajectories separate in PC space? Are neurons direction-selective?
2. **Trial type separation**: How distinct are GO, STOP, and CONT trajectories within each direction?
3. **Interaction effects**: Do trial types show different patterns for left vs. right directions?
4. **Dynamics**: Do trajectories show similar temporal evolution patterns across directions?
5. **Initial states**: Are starting points clustered or separated by condition and direction?
6. **Final states**: Do trajectories converge or diverge over time?
7. **PC structure**: Do top PCs capture:
   - Direction differences (left vs. right)?
   - Trial type differences (GO vs. STOP vs. CONT)?
   - Interactions between direction and trial type?

## Next Steps

### Potential Extensions

1. **Simplified analysis**: Combine directions to reduce to 3 conditions (GO, STOP, CONT) if direction selectivity is weak
2. **Session stratification**: Analyze subsets of sessions or compare across time
3. **SSD analysis**: Include STOP trials from different SSDs as separate conditions
4. **Time alignment**: Align all conditions to a common reference point (e.g., saccade onset)
5. **Cross-validation**: Train/test splits to assess generalization
6. **Decoding**: Use PC projections to predict trial type, direction, or outcome
7. **Direction-specific PCA**: Run separate PCAs for left and right trials to see if different PCs emerge

### Comparisons

Compare results with:
- Standard multi-session PCA (docs/PCA_GUIDE.md)
- Xie-style regression subspaces (xie_style_regression.ipynb)

Assess whether different methods reveal consistent population structure.

---

**Created**: December 2025
**Author**: Barak
**AI Assistant**: Claude (Anthropic)

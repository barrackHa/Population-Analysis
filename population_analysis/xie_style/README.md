# Xie-Style Compositional Regression for Stop-Signal Tasks

This directory contains an implementation of the compositional regression approach from Xie et al. (2022), adapted for countermanding stop-signal tasks.

## Overview

The analysis treats each trial as a **two-slot sequence**:
- **Slot 1**: GO direction (L/R)
- **Slot 2**: Second cue type (NONE for GO, STOP success/fail by SSD, CONT by SSD)

This creates a sparse "two-hot" design matrix where each trial activates exactly 2 regressors.

## Key Features

### 1. Compositional Design Matrix
- 2 direction features (GO_L, GO_R)
- 13 second-cue features:
  - NONE (for GO trials)
  - STOPsucc_SSD1-4 (successful stop trials by SSD)
  - STOPfail_SSD1-4 (failed stop trials by SSD)
  - CONT_SSD1-4 (continue trials by SSD)
- Total: 15 features with exactly 2 active per trial

### 2. Stable Regression via Half-Splits
- Lasso regression with automatic α selection (LassoCV)
- 100 repeated random half-splits for stability
- Coefficients averaged across all splits
- Per-cell fitting with ~37% sparsity
- **Parallel processing** using `joblib` (6 workers on 8-core machine)
  - Jupyter-friendly parallelization (works seamlessly in notebooks)
  - ~6x faster than sequential fitting
  - Maintains reproducibility with cell-specific random seeds

### 3. Population Vectors from β Coefficients
Each fitted coefficient column becomes a population vector across neurons:
```
v(feature) = [β₁(feature), ..., βₙ(feature)]ᵀ
```

### 4. Subspace Definition via PCA
Groups related population vectors and performs PCA to define low-dimensional subspaces:
- **STOP Success subspace**: PCA on {STOPsucc_SSD1-4} vectors
- **STOP Fail subspace**: PCA on {STOPfail_SSD1-4} vectors
- **CONT subspace**: PCA on {CONT_SSD1-4} vectors

### 5. Subspace Geometry Analysis
Computes **principal angles** between subspace pairs to quantify:
- **0°** = subspaces aligned (overlapping)
- **90°** = subspaces orthogonal (independent)

## Files

### Main Analysis
- **`xie_style_regression.ipynb`**: Complete interactive notebook with all steps
- **`xie_style_regression_executed.ipynb`**: Pre-executed notebook with outputs
- **`xie_style_regression_report.html`**: HTML report with visualizations
- **`test_xie_regression.py`**: Standalone test script

### Reference
- **`plan.md`**: Detailed implementation plan following Xie et al. methodology
- **`README.md`**: This file

## Usage

### Interactive Analysis
```bash
cd population_analysis/xie_style
conda activate $PWD/../../.conda
jupyter notebook xie_style_regression.ipynb
```

### Command-Line Testing
```bash
cd population_analysis/xie_style
../../.conda/bin/python test_xie_regression.py
```

## Results

Analysis on session `fi211110a` (84 cells, 943 trials):

### Regression Statistics
- **Features**: 15 (2 direction + 13 cue types)
- **Sparsity**: 37.4% (most coefficients driven to zero by Lasso)
- **Stability**: 100 half-splits with CV-selected regularization

### Subspace Variance Explained (PC1+PC2)
- **STOP Success**: 88.8% (highly structured response)
- **STOP Fail**: 85.0% (distinct from success)
- **CONT**: 97.2% (very low-dimensional)

### Principal Angles Between Subspaces
```
STOP Success ↔ STOP Fail:  85.9° (nearly orthogonal)
STOP Success ↔ CONT:       90.0° (perfectly orthogonal)
STOP Fail    ↔ CONT:       88.0° (nearly orthogonal)
```

**Interpretation**: The three subspaces are highly independent, suggesting distinct neural populations or coding strategies for successful stopping, failed stopping, and continued movement.

## Saved Outputs

Results are saved to `../../data/xie_style_results/`:

- **`fi211110a_betas.npy`**: Fitted coefficients (84 cells × 15 features)
- **`fi211110a_betas_std.npy`**: Coefficient stability (std across splits)
- **`fi211110a_subspace_*.npy`**: Subspace basis matrices (84 cells × 2 PCs)
- **`fi211110a_angles.csv`**: Principal angles between subspaces
- **`fi211110a_metadata.json`**: Analysis parameters and session info

## Integration with Existing Codebase

The implementation leverages your existing infrastructure:

```python
from cell_analysis import Cell
from session_class import Session

# Load session
session_data = cell_df[cell_df['trial_session'] == 'fi211110a']
session = Session(session_data)
session.drop_cells_with_missing_trial_type_or_dir_data()

# Extract spike counts using Cell methods
for cell_id in session.cell_ids:
    cell = session.get_cell(cell_id)
    # Use cell.data and cell.filter_trials() for spike extraction
```

## Key Differences from Standard PCA

| Aspect | Standard Multi-Session PCA | Xie-Style Regression |
|--------|---------------------------|---------------------|
| **Input** | Trial-averaged PSTHs | Single-trial spike counts |
| **Method** | Direct PCA on activity | Sparse regression → PCA on β vectors |
| **Time** | Full time-series | Single window (0-150ms post-cue) |
| **Sparsity** | Dense (all dimensions) | Sparse (Lasso penalty) |
| **Interpretability** | Temporal dynamics | Feature importance per condition |

## Next Steps

### 1. Multi-Session Analysis
Extend to multiple sessions:
- Fit regression per session independently
- Concatenate β coefficients across sessions
- Define cross-session subspaces

### 2. Time-Resolved Analysis
Create a rolling window version:
- Fit regression at multiple time points
- Track how subspaces evolve over time
- Measure trajectory angles within subspaces

### 3. Cross-Validation
Implement train/test splitting:
- Fit on train half-splits
- Test generalization on held-out trials
- Quantify prediction accuracy

### 4. Compare with Standard PCA
- Run both analyses side-by-side
- Compare subspace geometries
- Assess which method better captures task structure

## References

**Xie et al. (2022)**: "Geometry of sequence working memory in macaque prefrontal cortex."
*Science* 375(6581), 632-639.

Key methodological parallels:
- Two-hot compositional coding (their rank × item)
- Lasso regression with half-split stability
- Population vectors from β coefficients
- Within-group PCA to define subspaces
- Principal angles for geometry analysis

## Citation

If you use this analysis in your work:

```
Population Analysis Toolkit - Xie-Style Regression Module
Developed by: Barak & Claude (Anthropic)
Date: December 2025
Based on: Xie et al. (2022) Science
```

---

**Author**: Barak
**AI Assistant**: Claude (Anthropic)
**Last Updated**: December 17, 2025

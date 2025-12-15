# Cell Tuning Analysis - Implementation Summary

## Overview

This implementation provides comprehensive statistical testing to determine if cells are "tuned" to:
1. **Task modulation**: Activity changes from baseline to task period
2. **Direction tuning**: Distinguishes between left (180°) and right (0°) movements
3. **Trial type tuning**: Distinguishes between GO, STOP, and CONT trials
4. **Signal sensitivity**: Distinguishes between STOP and CONT signals

## Files

- **`cell_tuning_analysis.ipynb`**: Full interactive notebook with visualizations
- **`test_tuning_analysis.py`**: Standalone test script (verified working)

## Key Features

### 1. Statistical Tests
- **ANOVA** (F-test): Main test for comparing groups
- **Bonferroni correction**: Controls for multiple comparisons (α/5)
- **Permutation tests**: Non-parametric validation (in notebook)
- **Post-hoc t-tests**: Pairwise comparisons for trial types

### 2. Effect Size Measures
- **Cohen's d**: Standardized difference between groups (0.2=small, 0.5=medium, 0.8=large)
- **Eta-squared (η²)**: Proportion of variance explained by grouping variable
- **Direction Selectivity Index (DSI)**: (Left - Right) / (Left + Right), range [-1, 1]
- **Modulation Index (MI)**: (Task - Baseline) / (Task + Baseline), range [-1, 1]

### 3. Analysis Windows

| Test | Alignment | Window | Rationale |
|------|-----------|--------|-----------|
| Task Modulation | go_cue | Baseline: [-500, 0]<br>Task: [0, 500] | Compare pre-cue vs movement period |
| Direction Tuning | go_cue | [0, 500] | Movement execution period |
| Trial Type Tuning | go_cue | [0, 200] | Early movement initiation |
| Signal Sensitivity | stop_cue | [0, 200] | Signal processing period |

## Test Results (Session fi211110a, 85 cells)

```
Total cells analyzed: 85

Tuning properties:
  - Task modulated: 40 (47.1%)
  - Direction tuned: 10 (11.8%)
  - Trial type tuned: 0 (0.0%)
  - Signal sensitive: 4 (4.7%)
```

### Interpretation

- **47% task modulated**: Nearly half of cells show significant activity changes during task execution
- **12% direction tuned**: Modest proportion encode movement direction
- **0% trial type tuned**: With strict Bonferroni correction (p < 0.01), no cells pass this threshold
  - This may be too conservative; consider using p < 0.05 or FDR correction
- **5% signal sensitive**: Small subset distinguishes STOP vs CONT signals

## Advantages Over Previous ANOVA Notebooks

### cell_ANOVA.ipynb & cell_ANOVA_v2.ipynb Issues:
1. No multiple comparison correction
2. No effect size measures
3. No systematic classification scheme
4. Limited to ANOVA only

### This Implementation Adds:
1. ✓ **Multiple comparison correction** (Bonferroni)
2. ✓ **Effect sizes** (Cohen's d, η², DSI, MI)
3. ✓ **Permutation tests** (robustness check)
4. ✓ **Comprehensive classification** (tuning profiles)
5. ✓ **Population-level statistics**
6. ✓ **Visualization tools**
7. ✓ **Systematic testing** (verified on 85 cells)

## Usage

### Option 1: Jupyter Notebook (Full Analysis)
```bash
cd population_analysis
jupyter notebook cell_tuning_analysis.ipynb
```

### Option 2: Python Script (Quick Test)
```bash
.conda/bin/python test_tuning_analysis.py
```

### Option 3: Use as Module
```python
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd() / 'population_analysis'))

from session_class import Session
from cell_analysis import Cell

# Load data
cell_df = pd.read_pickle('data/unified_cell_trial_data/msn_fiona_cell_trial_data.pkl')
session_data = cell_df[cell_df['trial_session'] == 'fi211110a']
session = Session(session_data)

# Analyze a cell
cell = session.get_cell(session.cell_ids[0])
results = analyze_cell_tuning(cell, verbose=True)

# Access results
print(results['summary']['tuning_profile'])
print(f"Direction tuned: {results['summary']['is_direction_tuned']}")
print(f"DSI: {results['direction_tuning'].get('direction_selectivity_index', 'N/A')}")
```

## Customization

### Adjust Significance Threshold
```python
# More lenient (use uncorrected p-values)
results = analyze_cell_tuning(cell, alpha=0.05, verbose=True)

# More conservative
results = analyze_cell_tuning(cell, alpha=0.01, verbose=True)
```

### Modify Analysis Windows
Edit the `analyze_cell_tuning()` function:
```python
# Example: Extend movement period for direction tuning
go_left = get_firing_rates(cell, 'go_cue', [0, 700], trial_type='GO', direction=180)
go_right = get_firing_rates(cell, 'go_cue', [0, 700], trial_type='GO', direction=0)
```

### Add New Tests
```python
# Example: Add saccade-aligned analysis
saccade_left = get_firing_rates(cell, 'first_relevant_saccade', [-100, 200],
                                 trial_type='GO', direction=180)
saccade_right = get_firing_rates(cell, 'first_relevant_saccade', [-100, 200],
                                  trial_type='GO', direction=0)
```

## Next Steps

### Recommended Analyses:
1. **Multi-session analysis**: Run on all sessions, compare tuning proportions
2. **Temporal dynamics**: Test tuning in sliding time windows
3. **Cell type comparison**: Compare D1 vs D2 MSNs (if labeled)
4. **Correlation with behavior**: Relate tuning strength to RT, accuracy
5. **PCA integration**: Use tuning classification to interpret PC trajectories

### Potential Improvements:
1. **FDR correction**: Less conservative than Bonferroni
2. **Bootstrap confidence intervals**: More robust than parametric tests
3. **ROC analysis**: Alternative to selectivity indices
4. **Mixed-effects models**: Account for session/animal variability
5. **Time-resolved tuning**: When does selectivity emerge?

## References

- Cohen's d: Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences
- Selectivity indices: Britten et al. (1992), J. Neurosci. "The analysis of visual motion"
- Multiple comparisons: Bonferroni (1936), "Teoria statistica delle classi e calcolo delle probabilità"

## Contact

For questions or issues, refer to the main project documentation or modify the code as needed.

---
**Last Updated**: December 2025
**Tested On**: Session fi211110a (85 cells)
**Status**: ✓ Verified working

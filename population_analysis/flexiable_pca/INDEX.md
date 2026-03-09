# FlexiblePCA - Quick Index

**A modular PCA system for flexible neural population analysis.**

Fit PCA on specific trial conditions/epochs, then project other conditions onto those PCs.

---

## 🚀 Quick Start (30 seconds)

```bash
cd /Users/barak/Projects/population_analysis/population_analysis/xie_style
../../.conda/bin/python flexible_pca_quickstart.py
```

**What it does:**
- Fits PCA on GO trials
- Projects STOP/CONT trials onto GO-defined PC space
- Generates 4 publication-quality figures
- Saves to `data/flexible_pca_results/`

---

## 📚 Documentation

| File | Purpose | When to Use |
|------|---------|-------------|
| **FLEXIBLE_PCA_SUMMARY.md** | Complete overview | First time setup |
| **README_FLEXIBLE_PCA.md** | Detailed guide & API reference | Learning the system |
| **flexible_pca_demo.ipynb** | Interactive examples | Hands-on exploration |
| **INDEX.md** | This file - quick navigation | Finding what you need |

---

## 💻 Code Files

| File | Purpose | Lines |
|------|---------|-------|
| **flexible_pca.py** | Main analysis module | ~500 |
| **flexible_pca_plots.py** | Visualization utilities | ~400 |
| **flexible_pca_quickstart.py** | Quick demo script | ~130 |

---

## 🎯 Common Tasks

### I want to...

**Run a quick demo**
```bash
../../.conda/bin/python flexible_pca_quickstart.py
```

**See interactive examples**
```bash
jupyter notebook flexible_pca_demo.ipynb
```

**Use in my own script**
```python
from flexible_pca import FlexiblePCA, TrialSpec
# See README_FLEXIBLE_PCA.md for examples
```

**Understand the concepts**
- Read: FLEXIBLE_PCA_SUMMARY.md → "Key Advantages" section
- Read: README_FLEXIBLE_PCA.md → "Core Concepts" section

**Learn the API**
- Read: README_FLEXIBLE_PCA.md → "Key Methods" section
- Check: Docstrings in flexible_pca.py

**Make custom plots**
```python
from flexible_pca_plots import plot_3d_trajectories, plot_pc_timeseries
# See flexible_pca_plots.py for all plotting functions
```

**Compare with original approach**
- Original: `condition_concatenated_pca.ipynb` (fixed 6 conditions)
- New: `flexible_pca.py` (any conditions, reusable projections)
- Details: FLEXIBLE_PCA_SUMMARY.md → "Comparison" section

---

## 📖 Reading Order

### First Time Users
1. **FLEXIBLE_PCA_SUMMARY.md** - Get the big picture (15 min)
2. Run `flexible_pca_quickstart.py` - See it in action (2 min)
3. **flexible_pca_demo.ipynb** - Try examples (30 min)
4. **README_FLEXIBLE_PCA.md** - Deep dive when needed (reference)

### Quick Reference Users
1. **INDEX.md** - This file (find what you need)
2. Jump to relevant section in README_FLEXIBLE_PCA.md

---

## 🔧 Key Classes & Functions

### Main Classes
- `FlexiblePCA` - The main analysis class
- `TrialSpec` - Specify trial conditions

### Key Methods
- `fpca.fit(trial_specs)` - Fit PCA on specified conditions
- `fpca.project(trial_specs)` - Project conditions onto fitted PCs
- `fpca.get_fit_trajectories()` - Get trajectories from fit data
- `fpca.get_time_axis(spec)` - Get time axis for plotting

### Plotting Functions
- `plot_3d_trajectories()` - 3D PC space
- `plot_2d_projections()` - All pairwise PC projections
- `plot_pc_timeseries()` - PC evolution over time
- `plot_variance_explained()` - Scree & cumulative variance
- `create_comparison_figure()` - Complete figure set

### Helper Functions
- `create_standard_specs()` - Standard 6-condition setup

---

## 📊 Example Use Cases

| Goal | Fit On | Project | Why |
|------|--------|---------|-----|
| Compare trial types | GO trials | STOP/CONT | See how signals affect GO space |
| Temporal evolution | Early epoch | Later epochs | Track dynamics in fixed space |
| Alignment effects | go_cue aligned | stop_cue aligned | Compare reference frames |
| Direction selectivity | Right trials | Left trials | Test shared vs. separate PCs |
| Standard analysis | All 6 conditions | - | Recreate original analysis |

See README_FLEXIBLE_PCA.md for code examples.

---

## 🏗️ Architecture

```
Cell Database (pandas DataFrame)
        ↓
  TrialSpec (define conditions)
        ↓
  FlexiblePCA.fit(specs)
        ↓
  PCA Model (fitted on specified conditions)
        ↓
  FlexiblePCA.project(different_specs)
        ↓
  Trajectories (n_components, n_time_bins)
        ↓
  Visualization (flexible_pca_plots)
```

---

## 🎨 Output Examples

After running `flexible_pca_quickstart.py`, check:
```
data/flexible_pca_results/
├── quickstart_3d_trajectory.png    # 3D visualization
├── quickstart_2d_projections.png   # All PC pairs
├── quickstart_timeseries.png       # PC vs time
└── quickstart_variance.png         # Variance explained
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Import error | Check you're using `../../.conda/bin/python` |
| "Must call fit()" | Call `fpca.fit(specs)` before `fpca.project()` |
| Low variance | Increase `n_components` or check data quality |
| Missing neurons | Check trial counts in verbose output |

Full troubleshooting: README_FLEXIBLE_PCA.md → "Troubleshooting"

---

## 📞 Getting Help

1. **Quick answer**: Search this INDEX.md
2. **How-to**: Check README_FLEXIBLE_PCA.md
3. **Examples**: Run flexible_pca_demo.ipynb
4. **Concepts**: Read FLEXIBLE_PCA_SUMMARY.md
5. **Code details**: Read docstrings in flexible_pca.py

---

## 🔄 Workflow Recommendation

```
1. Read FLEXIBLE_PCA_SUMMARY.md (overview)
   ↓
2. Run flexible_pca_quickstart.py (see it work)
   ↓
3. Check output figures (understand results)
   ↓
4. Open flexible_pca_demo.ipynb (try examples)
   ↓
5. Use README_FLEXIBLE_PCA.md as reference (when needed)
   ↓
6. Write your own analysis scripts (research!)
```

---

## 📦 What's Included

- ✅ Full-featured PCA analysis class
- ✅ Flexible trial specification system
- ✅ Projection onto fitted PCs (key feature!)
- ✅ Ready-to-use plotting functions
- ✅ Quick-start demo script
- ✅ Interactive Jupyter notebook
- ✅ Comprehensive documentation
- ✅ Code examples for common use cases
- ✅ Type hints and docstrings
- ✅ Error handling and validation

**Total: ~2500 lines of code and documentation**

---

## 🆚 vs. Original Notebook

| Feature | Original | FlexiblePCA |
|---------|----------|-------------|
| Conditions | 6 fixed | Any |
| Projection | ❌ | ✅ |
| Reusability | Low | High |
| Organization | Cells | Classes |
| Docs | Comments | Full docs |

**Bottom line**: Original is great for standard analysis. FlexiblePCA extends it for hypothesis testing.

---

## 🎯 Key Innovation

**The projection feature** is what makes this system powerful:

```python
# Fit on one condition
fpca.fit([TrialSpec('GO', ...)])

# Project many different conditions onto same PCs
proj1 = fpca.project([TrialSpec('STOP', ...)])
proj2 = fpca.project([TrialSpec('CONT', ...)])
proj3 = fpca.project([TrialSpec('GO', epoch=[later], ...)])

# All using the SAME PC space defined by fit!
```

This lets you:
- Compare conditions in same coordinate system
- Track temporal evolution in fixed space
- Test hypotheses about neural representations

---

**Built with ❤️ by Claude & Barak | December 2024**

---

## 🚦 Status Indicators

- 🟢 **Ready to use**: flexible_pca.py, flexible_pca_plots.py
- 🟢 **Tested**: flexible_pca_quickstart.py works end-to-end
- 🟢 **Documented**: Complete docs in README_FLEXIBLE_PCA.md
- 🟡 **Examples**: Demo notebook needs execution
- 🔵 **Optional**: Extensions suggested in SUMMARY.md

---

**Quick Links:**
- [Complete Overview](FLEXIBLE_PCA_SUMMARY.md)
- [Detailed Guide](README_FLEXIBLE_PCA.md)
- [Interactive Demo](flexible_pca_demo.ipynb)
- [Original Approach](condition_concatenated_pca.ipynb)

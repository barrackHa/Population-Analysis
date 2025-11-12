# Coding and Visualization Standards

## Coding Guidelines

### Data Processing

#### 1. Always Check for Missing Data

```python
# Example: SSD numbers are NaN for GO trials
if trial_type in ['STOP', 'CONT']:
    ssd_numbers = sorted(data['ssd_number'].dropna().unique())
```

#### 2. Spike Time Handling

```python
# Spikes are stored as numpy arrays
spikes = np.array(row['neural_data'], dtype=float)

# Always filter by epoch after alignment
aligned_spikes = spikes - alignment_time
spikes_in_epoch = aligned_spikes[(aligned_spikes >= epok[0]) &
                                  (aligned_spikes <= epok[1])]
```

#### 3. Trial Filtering Best Practices

```python
# Chain filters explicitly
filtered = data.copy()
if trial_type is not None:
    filtered = filtered[filtered['type'] == trial_type]
if direction is not None:
    filtered = filtered[filtered['dir'] == direction]
if success_only:
    filtered = filtered[filtered['trial_failed'] == False]
```

---

### Performance

#### 1. Avoid Repeated Alignment

```python
# Check before aligning
col_name = f'spikes_aligned_to_{alignment_point}'
if col_name not in self.data.columns:
    self.align_spikes_to_event(alignment_point)
```

#### 2. Reuse Computed Data

```python
# Generate once
data_left = self.get_population_data_trial_types(direction=180, ...)
data_right = self.get_population_data_trial_types(direction=0, ...)

# Use multiple times
plot1 = self.plot_trial_type_comparison(data_left, data_right)
# Can analyze data_left['go']['psth_matrix'] separately
```

#### 3. SSD Analysis Separation

```python
# Instead of 8×2 grid (heavy):
plot_all_ssd()  # 16 heatmaps

# Use 4×2 grids (lighter):
plot_trial_type_by_ssd(trial_type='STOP')  # 8 heatmaps
plot_trial_type_by_ssd(trial_type='CONT')  # 8 heatmaps
```

---

## Visualization Standards

### Default Parameters

```python
# Temporal
EPOCH = [-200, 700]  # ms
BIN_SIZE = 10  # ms
ALIGNMENT = 'go_cue' or 'stop_cue'

# Processing
SMOOTH = True  # Gaussian smoothing
SIGMA = bin_size  # Same as bin size
TRUNCATE = 2  # ±2σ window
NORMALIZE = True  # Global normalization
SUCCESS_ONLY = True  # Exclude failed trials

# Visual
COLORMAP = 'Plasma'  # Yellow (high) to purple (low)
COLORBAR = True
TOOLS = ['hover']
SHOW_GRID = True
```

---

### Consistent Color Schemes

```python
# Use defined palettes
SSD_COLORS = {1: '#e377c2', 2: '#8c564b', 3: '#bcbd22', 4: '#17becf'}
TYPE_COLORS = {'GO': '#2ca02c', 'STOP': '#d62728', 'CONT': '#9467bd'}
DIRECTION_COLORS = {0: '#1f77b4', 180: '#ff7f0e'}
```

---

### Plasma Colormap Interpretation

- **Yellow/Bright**: High firing rate
- **Orange**: Moderate-high firing rate
- **Pink/Purple**: Moderate firing rate
- **Dark purple/Black**: Low/no firing rate

With global normalization:
- **1.0 (yellow)**: Highest firing rate across all cells
- **0.0 (dark purple)**: Lowest firing rate or no activity

---

### Holoviews Configuration

#### Image Kdims Order

```python
# CORRECT: Time on X-axis, Neurons on Y-axis
img = hv.Image(
    psth_matrix,
    kdims=['Time', 'Neurons'],  # Order matters!
    vdims='Firing Rate',
    bounds=(epok[0], 0, epok[1], n_cells)
)
```

#### Bounds vs Transpose

```python
# Use bounds parameter, NOT transpose
# bounds=(x_min, y_min, x_max, y_max)
bounds=(epok[0], 0, epok[1], n_cells)

# Set invert_yaxis=False (default behavior we want)
.opts(invert_yaxis=False)
```

#### Plot Sizing for Grids

```python
# Single plots
width=800, height=600

# 2-column grids
width=400, height=300

# 3-column grids
width=350, height=600

# 4-column grids (SSD analysis)
width=300, height=250
```

#### Layout Management

```python
# Single plot
plot

# Horizontal layout (+ operator)
plot1 + plot2

# Vertical layout (.cols(1))
(plot1 + plot2 + plot3).cols(1)

# Grid layout
(plot1 + plot2 + plot3 + plot4).cols(2)  # 2×2 grid

# NdOverlay for multiple curves
overlay = hv.NdOverlay({
    'curve1': hv.Curve(data1),
    'curve2': hv.Curve(data2)
})
```

---

### Grid Layout Guidelines

#### 3×2 Grid (Trial Type Comparison)

```
Row 1: GO-Left    | GO-Right
Row 2: STOP-Left  | STOP-Right
Row 3: CONT-Left  | CONT-Right

All use same cell ordering (from GO-left peaks)
```

#### 4×2 Grid (SSD Comparison)

```
Row 1: SSD1-Left  | SSD1-Right
Row 2: SSD2-Left  | SSD2-Right
Row 3: SSD3-Left  | SSD3-Right
Row 4: SSD4-Left  | SSD4-Right

All use same cell ordering (from GO-left peaks)
```

---

## Important Implementation Notes

### 1. Normalization Philosophy

**Old approach** (per-cell):
```python
firing_rate = (firing_rate - firing_rate.min()) /
              (firing_rate.max() - firing_rate.min())
```
- Made all cells appear equally strong
- Lost relative magnitude information

**New approach** (global):
```python
psth_matrix = np.array(psth_list)
global_max = psth_matrix.max()
psth_matrix = psth_matrix / global_max
```
- Preserves relative firing rate differences
- Shows which cells are dominant contributors
- Critical for population analysis

---

### 2. Cell Ordering Consistency

**Key principle**: Use ONE reference condition for ordering

```python
# Example: Use GO-left for all plots
data_go = get_population_data_single_condition(
    trial_type='GO', direction=180, sort_by_peak=True
)
cell_order = data_go['cell_ids']

# Then reorder all other conditions to match
for other_condition in [stop_left, stop_right, cont_left, cont_right]:
    reorder_to_match(cell_order)
```

---

### 3. Alignment Point Selection

**GO trials**: Always use `go_cue`
- No secondary signal present
- `stop_cue` is NaN

**STOP trials**: Choose based on question
- `go_cue`: Compare initiation phase across conditions
- `stop_cue`: Focus on inhibition process (stop signal appears)
- Typical epoch with go_cue: [-200, 1000] to see stop signal effect
- Typical epoch with stop_cue: [-200, 700] to focus on inhibition

**CONT trials**: Choose based on question
- `go_cue`: Compare initiation phase across conditions
- `stop_cue`: Focus on continue signal processing (**green cue, not stop**)
- Note: `stop_cue` contains **continue signal time** for CONT trials
- Same epoch recommendations as STOP trials

---

### 4. SSD Effects on Analysis

```python
# GO trials: No SSD
trial_type='GO' → ssd_number=None (always)

# STOP/CONT: Must handle SSD
if trial_type in ['STOP', 'CONT']:
    if ssd_number is None:
        # Combine all SSDs
    else:
        # Filter specific SSD
        data = data[data['ssd_number'] == ssd_number]
```

---

## Git Workflow

```bash
# Current branch
git branch  # pca

# Commit pattern
git add session_class.py session_analysis.ipynb
git commit -m "Add feature: description"

# Check status frequently
git status
```

---

## References

1. **Pani et al. (2022)**: Original methodology paper
2. **HoloViews Documentation**: http://holoviews.org/
3. **Bokeh Documentation**: https://docs.bokeh.org/
4. **SciPy ndimage**: Gaussian filtering documentation

---

**Last Updated**: November 2025
**Project Lead**: Barak
**AI Assistant**: Claude (Anthropic)
**Repository**: Population-Analysis (pca branch)

---

**Related Documentation**:
- [API Reference](API_REFERENCE.md)
- [Workflows & Usage Patterns](WORKFLOWS.md)
- [PCA Analysis Guide](PCA_GUIDE.md)

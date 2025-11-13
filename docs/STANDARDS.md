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

## Clean, Modular Code Principles

### 1. Class Ownership and Responsibility

**Principle**: Classes should own their validation and state-checking logic.

**Bad Example** (validation outside the class):
```python
# Validation logic scattered in calling code
def validate_session(session_data):
    n_cells = session_data['cell_ID'].nunique()
    if n_cells < 10:
        return False

    types = set(session_data['type'].unique())
    if not {'GO', 'STOP'}.issubset(types):
        return False

    # More validation...
    return True

# Caller needs to know validation details
session = Session(session_data)
if validate_session(session_data):
    # Use session
```

**Good Example** (validation methods in the class):
```python
class Session:
    def has_trial_types(self, required_types):
        """Check if session has all required trial types."""
        return set(required_types).issubset(set(self.trial_types))

    def validate_min_trials_per_condition(self, trial_types, directions, min_trials):
        """Validate minimum trials per condition."""
        # Implementation...
        return is_valid, reason

# Caller uses clean interface
session = Session(session_data)
if session.has_trial_types(['GO', 'STOP']):
    # Use session
```

**Benefits**:
- ✓ Session class owns its validation logic
- ✓ Validation methods are reusable
- ✓ Easier to test individually
- ✓ Clear separation of concerns

---

### 2. Validate After Creation, Not Before

**Principle**: Create objects first, then validate their state using the object's methods.

**Bad Example** (validate raw data before creating object):
```python
# Check 1: Raw data validation
if session_data['cell_ID'].nunique() < min_cells:
    return False

# Check 2: More raw data validation
if not set(required_types).issubset(set(session_data['type'].unique())):
    return False

# Check 3: Create object (too late!)
session = Session(session_data)
session.drop_incomplete_cells()

# Check 4: Now check after dropping
if session.n_cells < min_cells:
    return False
```

**Good Example** (create object, then validate):
```python
# Step 1: Create Session
session = Session(session_data)

# Step 2: Prepare data
session.drop_incomplete_cells()

# Step 3: Validate using Session methods
if session.n_cells < min_cells:
    return False

if not session.has_trial_types(required_types):
    return False

if not session.validate_min_trials_per_condition(...):
    return False
```

**Benefits**:
- ✓ Simpler workflow: create → prepare → validate
- ✓ Validation uses actual object state (not raw data)
- ✓ Avoids duplicate checks
- ✓ Clearer logic flow

---

### 3. Keep Methods Short and Focused

**Principle**: Each method should do ONE thing. Methods longer than ~30 lines should be split.

**Bad Example** (long, multi-purpose method):
```python
def process_and_validate_session(session_data, config):
    # 50+ lines doing everything:
    # - Creating session
    # - Dropping cells
    # - Checking cell count
    # - Checking trial types
    # - Checking directions
    # - Checking trial counts
    # - All inline with complex conditionals
    pass
```

**Good Example** (short, focused methods):
```python
def _validate_single_session(self, session_data, session_id):
    """Validate session. Each check is a focused method."""
    session = self._create_session(session_data)
    session.drop_incomplete_cells()

    if not self._has_sufficient_cells(session):
        return False, "Insufficient cells"

    if not session.has_trial_types(self.required_types):
        return False, "Missing trial types"

    # Each check: one line, clear purpose
    return True, "Valid"
```

**Benefits**:
- ✓ Easy to read and understand
- ✓ Easy to test individual checks
- ✓ Easy to modify or extend
- ✓ Self-documenting code

---

### 4. Methods Should Return Useful Values

**Principle**: Design methods to return information that enables decision-making.

**Good patterns**:

```python
# Boolean check
if session.has_trial_types(['GO', 'STOP']):
    # Simple yes/no decision
    pass

# Count for threshold checking
n_trials = session.get_trial_count_for_condition('GO', 0)
if n_trials >= min_threshold:
    pass

# Validation with reason (tuple return)
is_valid, reason = session.validate_min_trials_per_condition(...)
if not is_valid:
    print(f"Validation failed: {reason}")
```

**Return patterns**:
- `bool` - Simple yes/no checks
- `int/float` - Counts, measurements
- `tuple` - `(is_valid, reason)` for detailed validation
- `list/dict` - Collections of results

---

### 5. Make Code Testable

**Principle**: Design methods so they can be tested independently.

**Testable design**:
```python
# Each method can be tested in isolation
def test_has_trial_types_all_present():
    session = create_test_session()
    assert session.has_trial_types(['GO', 'STOP', 'CONT'])

def test_has_trial_types_missing():
    session = create_test_session()
    assert not session.has_trial_types(['GO', 'INVALID'])

def test_get_trial_count_for_condition():
    session = create_test_session()
    count = session.get_trial_count_for_condition('GO', 0)
    assert count > 0
```

**Benefits**:
- ✓ Test one thing at a time
- ✓ Catch bugs early
- ✓ Refactor with confidence
- ✓ Document expected behavior

---

### 6. DRY Principle (Don't Repeat Yourself)

**Bad Example** (repeated logic):
```python
# In validator class
for trial_type in trial_types:
    for direction in directions:
        condition_data = session.data[
            (session.data['type'] == trial_type) &
            (session.data['dir'] == direction) &
            (session.data['trial_failed'] == False)
        ]
        n_trials = len(condition_data['trial_number'].unique())
        # Use n_trials...

# Somewhere else in the same codebase
# DUPLICATE: Same logic repeated
condition_data = session.data[
    (session.data['type'] == trial_type) &
    (session.data['dir'] == direction) &
    (session.data['trial_failed'] == False)
]
n_trials = len(condition_data['trial_number'].unique())
```

**Good Example** (reusable method):
```python
# In Session class - one implementation
def get_trial_count_for_condition(self, trial_type, direction, success_only=True):
    """Get trial count for a specific condition."""
    condition_data = self.data[
        (self.data['type'] == trial_type) &
        (self.data['dir'] == direction)
    ]
    if success_only:
        condition_data = condition_data[condition_data['trial_failed'] == False]
    return len(condition_data['trial_number'].unique())

# Reuse everywhere
n_trials = session.get_trial_count_for_condition('GO', 0)
```

---

### 7. Use Clear, Descriptive Names

**Naming conventions**:

```python
# Methods: verb_noun or action_description
def has_trial_types(...)          # Check/query methods
def get_trial_count(...)           # Getter methods
def validate_min_trials(...)       # Validation methods
def drop_incomplete_cells(...)     # Action methods

# Variables: descriptive nouns
n_cells_after_drop                 # Not just 'n' or 'count'
required_trial_types               # Not just 'types'
session_validation_results         # Not just 'results'

# Boolean variables/returns: is_, has_, should_
is_valid
has_sufficient_data
should_exclude_session
```

---

### 8. Refactoring Checklist

When refactoring code, ask:

- [ ] Can this validation be a method of the class being validated?
- [ ] Is this method doing more than one thing?
- [ ] Can this method be tested independently?
- [ ] Is this logic duplicated elsewhere?
- [ ] Are the names clear and descriptive?
- [ ] Does the method return useful information?
- [ ] Is the workflow logical (create → prepare → validate)?
- [ ] Would this be easy for someone else to understand?

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

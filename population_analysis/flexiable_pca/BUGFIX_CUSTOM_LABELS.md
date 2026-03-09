# Bug Fix: Custom Labels in FlexiblePCA

## Issue

When using custom labels in `TrialSpec` (e.g., `label='GO_R_early'`), the `get_fit_trajectories()` method would raise a `KeyError` because it was using auto-generated labels instead of the custom ones.

### Error Example
```python
# This would fail:
spec = TrialSpec('GO', direction=0, epoch=[-50, 100], alignment='go_cue', label='GO_R_early')
fpca.fit([spec])
trajectories = fpca.get_fit_trajectories()
# KeyError: 'GO_R_early' (it was looking for 'GO_R' instead)
```

## Root Cause

The code was constructing labels as `f"{spec.trial_type}_{dir_str}"` in multiple places:
- `_extract_neuron_concat_psths()` (line 212)
- `fit()` print statements (line 259, 307)
- `fit()` condition_indices (line 345)
- `project()` (line 408)

This ignored any custom labels provided in the `TrialSpec`.

## Solution

Added a new method `get_label_for_direction(direction: int)` to the `TrialSpec` class:

```python
def get_label_for_direction(self, direction: int) -> str:
    """
    Get the label for a specific direction.

    If this TrialSpec has a single direction and a custom label, use it.
    Otherwise, construct label from trial_type and direction.
    """
    if isinstance(self.direction, int) and self.direction == direction:
        return self.label  # Use custom label
    else:
        # Multiple directions or mismatch - construct label
        dir_str = self._dir_to_str(direction)
        return f"{self.trial_type}_{dir_str}"
```

Then replaced all instances of manual label construction with calls to this method.

## Changes Made

### File: `flexible_pca.py`

1. **Added method** (lines 89-103):
   - `TrialSpec.get_label_for_direction(direction)`

2. **Updated `_extract_neuron_concat_psths()`** (line 227):
   ```python
   # Old:
   cond_label = f"{spec.trial_type}_{dir_str}"

   # New:
   cond_label = spec.get_label_for_direction(direction)
   ```

3. **Updated `fit()` method** (lines 273, 321, 358):
   - Print statements now use `spec.get_label_for_direction(direction)`
   - condition_indices now use `spec.get_label_for_direction(direction)`

4. **Updated `project()` method** (lines 408, 421):
   - Print statements now use `spec.get_label_for_direction(direction)`
   - Projection loop now uses `spec.get_label_for_direction(direction)`

## Testing

Created `test_custom_labels.py` to verify the fix:

```bash
../../.conda/bin/python test_custom_labels.py
```

**All tests pass:**
- ✓ Fit with custom labels
- ✓ Project with custom labels
- ✓ get_fit_trajectories() returns correct labels
- ✓ Mix of custom and auto-generated labels

## Usage Examples

### Example 1: Custom labels for temporal comparison
```python
# Fit on early epoch with custom labels
early_specs = [
    TrialSpec('GO', direction=0, epoch=[-50, 100], alignment='go_cue', label='GO_R_early'),
    TrialSpec('GO', direction=180, epoch=[-50, 100], alignment='go_cue', label='GO_L_early'),
]
fpca.fit(early_specs)

# Project late epoch with custom labels
late_specs = [
    TrialSpec('GO', direction=0, epoch=[100, 300], alignment='go_cue', label='GO_R_late'),
    TrialSpec('GO', direction=180, epoch=[100, 300], alignment='go_cue', label='GO_L_late'),
]
projections = fpca.project(late_specs)

# Get trajectories - custom labels are preserved
fit_traj = fpca.get_fit_trajectories()
# Returns: {'GO_R_early': ..., 'GO_L_early': ...}

# Projections also have custom labels
# Returns: {'GO_R_late': ..., 'GO_L_late': ...}
```

### Example 2: Mix of custom and auto labels
```python
specs = [
    TrialSpec('GO', direction=0, epoch=[-50, 300], alignment='go_cue'),  # Auto: 'GO_R'
    TrialSpec('STOP', direction=0, epoch=[-50, 300], alignment='go_cue',
              ssd_number=2, label='STOP_custom'),  # Custom label
]

fpca.fit(specs)
trajectories = fpca.get_fit_trajectories()
# Returns: {'GO_R': ..., 'STOP_custom': ...}
```

### Example 3: Multiple directions (auto labels)
```python
# When direction=None, always uses auto-generated labels
spec = TrialSpec('GO', direction=None, epoch=[-50, 300], alignment='go_cue')
# Creates labels: 'GO_R' and 'GO_L' (even if custom label provided)
```

## Backward Compatibility

✅ **Fully backward compatible**
- Code without custom labels works exactly as before
- Auto-generated labels follow the same pattern: `{trial_type}_{R/L}`
- Existing code doesn't need any changes

## Demo Notebook

The demo notebook (`flexible_pca_demo.ipynb`) Example 2 now works correctly:

```python
# This now works without KeyError:
early_specs = [
    TrialSpec('GO', direction=0, epoch=[-50, 100], alignment='go_cue', label='GO_R_early'),
    TrialSpec('GO', direction=180, epoch=[-50, 100], alignment='go_cue', label='GO_L_early'),
]
fpca.fit(early_specs)
early_traj = fpca.get_fit_trajectories()

# Access with custom labels:
ax.plot(early_traj['GO_R_early'][0, :], ...)  # ✓ Works!
```

## Implementation Notes

The `get_label_for_direction()` method handles three cases:

1. **Single direction with custom label**: Returns the custom label
   ```python
   spec = TrialSpec('GO', direction=0, label='custom')
   spec.get_label_for_direction(0)  # Returns: 'custom'
   ```

2. **Single direction, no custom label**: Constructs standard label
   ```python
   spec = TrialSpec('GO', direction=0)  # label auto-set to 'GO_R'
   spec.get_label_for_direction(0)  # Returns: 'GO_R'
   ```

3. **Multiple directions**: Always constructs labels per direction
   ```python
   spec = TrialSpec('GO', direction=None, label='custom')
   spec.get_label_for_direction(0)    # Returns: 'GO_R' (not 'custom')
   spec.get_label_for_direction(180)  # Returns: 'GO_L' (not 'custom')
   ```

## Files Modified

- `flexible_pca.py` - Added method and updated 5 locations

## Files Created

- `test_custom_labels.py` - Comprehensive test suite
- `BUGFIX_CUSTOM_LABELS.md` - This documentation

## Status

✅ **Bug fixed and tested**
✅ **All tests passing**
✅ **Backward compatible**
✅ **Demo notebook works**

---

**Date**: December 2024
**Fix**: Custom label support in FlexiblePCA
**Tested**: ✓ Passed all tests

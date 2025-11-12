# Data Structures Reference

## Main DataFrame (`cell_df`)

Unified cell-trial database stored as pickle file: `msn_fiona_cell_trial_data.pkl`

### Key Columns

```python
# Cell identification
'cell_ID'           # Unique cell identifier (int)
'cell_type'         # Cell type classification
'trial_session'     # Session identifier (e.g., 'fi211110a')

# Trial classification
'type'              # 'GO', 'STOP', 'CONT'
'dir'               # Direction: 0 (right) or 180 (left)
'ssd_number'        # SSD level: 1.0-4.0 (NaN for GO trials)
'trial_failed'      # Boolean: Success depends on trial type
                    #   GO: False=saccade made, True=no saccade
                    #   STOP: False=successfully inhibited, True=failed to inhibit
                    #   CONT: False=saccade made, True=inappropriately inhibited

# Temporal events (timestamps in ms)
'go_cue'            # Go signal time
'stop_cue'          # Stop signal (STOP) OR Continue signal (CONT) time (NaN for GO trials)
'first_relevant_saccade'  # Array [start, end]: saccade start time (movement onset) and end time (target fixation)
'reaction_time'     # RT from go_cue to first_relevant_saccade[0]

# Neural data
'neural_data'       # Array of spike times (absolute timestamps)

# Trial metadata
'trial_number'      # Trial number within session
'trial_length'      # Total trial duration
'screen_rotation'   # Screen orientation (0 for Fiona, varies for Yasmin)
'saccades'          # All saccade events
'blinks'            # Blink events
'grade'             # Cell quality score (1-10, ≥8 recommended)
```

---

## Important Notes

### screen_rotation
Currently only analyzing `screen_rotation = 0` (Fiona). Yasmin has rotated display requiring coordinate transformations.

### grade
Quality metric for cell isolation. Threshold of <= 8 filters out poorly isolated units.

### trial_failed Interpretation
The meaning of `trial_failed` depends on the trial type:
- **GO trials**: `False` = saccade made (success), `True` = no saccade (failure)
- **STOP trials**: `False` = successfully inhibited (success), `True` = failed to inhibit (failure)
- **CONT trials**: `False` = saccade made (success), `True` = inappropriately inhibited (failure)

### stop_cue Column
Contains different information based on trial type:
- **STOP trials**: Time of red stop signal
- **CONT trials**: Time of green continue signal
- **GO trials**: `NaN` (no secondary signal)

---

## Data Quality and Filtering

### Trial Failure Validation
- STOP trial failures are validated by saccade amplitude
- Ensures that "failed to stop" means an actual saccade was executed
- Method: `update_stop_trial_failures()` in behavioral analysis
- Prevents false failures from noise or small eye movements

### Session Quality
Not all recording sessions are suitable for analysis. Some sessions excluded due to:
- Recording artifacts
- Insufficient trial counts
- Equipment issues
- Behavioral anomalies

Always check for session exclusion lists before analysis.

### Grade Filtering
Cell quality rated 1-10 based on:
- Spike waveform consistency
- Signal-to-noise ratio
- Unit isolation quality

**Standard threshold**: Grade ≥ 8

Lower grades may have contamination from other units.

---

## Data Dictionary Structure

Methods that return population data use this standard format:

```python
{
    'bin_centers': np.array,      # Time points
    'psth_matrix': np.array,      # (n_cells × n_bins)
    'cell_ids': list,             # Ordered cell IDs
    'params': {                   # Metadata
        'epok': [-200, 700],
        'bin_size': 10,
        'alignment_point': 'go_cue',
        'trial_type': 'STOP',
        'direction': 0,
        'ssd_number': None,
        'success_only': True,
        'smooth': True,
        'normalize': True
    }
}
```

---

## Common Epochs

- **Full trial**: [-500, 1500]
- **Standard**: [-200, 700]
- **Pre-stimulus**: [-200, 0]
- **Post-stimulus**: [0, 700]
- **PCA epoch**: [-50, 150] (from before go_cue to after movement onset - captures fixation end, target processing, reaction time, and movement initiation)

---

**Related Documentation**:
- [Stop-Signal Task Reference](TASK_REFERENCE.md)
- [API Reference](API_REFERENCE.md)
- [Workflows & Usage Patterns](WORKFLOWS.md)

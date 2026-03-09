# Stop-Signal Task Reference

## Task Description

A countermanding stop-signal task (CSST) where the subject performs visually-guided saccades with three trial types:

**Basic Task Flow:**
1. **Fixation**: Subject fixates on a central point
2. **Go Cue**: Peripheral target appears, signaling the subject to make a saccade
3. **Signal Cue** (STOP/CONT trials only): Secondary visual cue appears after a variable delay
   - **Stop Signal**: Red cue instructs subject to cancel the saccade
   - **Continue Signal**: Green cue instructs subject to proceed with the saccade (control condition)
4. **Response**: Subject either executes or inhibits the saccade depending on the trial type

---

## Trial Types

### 1. GO Trials

- **Description**: No secondary signal presented
- **Expected behavior**: Execute saccade to target
- **Purpose**: Baseline condition measuring normal saccadic response
- **Success criterion**: Saccade executed within time window
- **Data markers**:
  - `type = 'GO'`
  - `stop_cue = NaN` (no signal)
  - `ssd_number = NaN`
- **Alignment**: `go_cue`

### 2. STOP Trials (Successful Inhibition)

- **Description**: **Red stop signal** presented after variable delay
- **Expected behavior**: Cancel/inhibit the planned saccade
- **Outcome**: `trial_failed = False` (successfully inhibited)
- **Purpose**: Measure inhibitory control capacity
- **Success criterion**: No saccade made after stop signal
- **Data markers**:
  - `type = 'STOP'`
  - `stop_cue = timestamp` (time of red stop signal)
  - `ssd_number = 1.0-4.0` (stop signal delay level)
  - `first_relevant_saccade = NaN` (no saccade)
- **Alignment**: `go_cue` or `stop_cue`

### 3. CONT (Continue) Trials (Control Condition)

- **Description**: **Green continue signal** presented after variable delay
- **Expected behavior**: Proceed with the saccade despite the visual cue
- **Outcome**: `trial_failed = False` if saccade executed, `True` if inhibited
- **Purpose**: Control condition to match visual stimulation of STOP trials without inhibitory demand
- **Success criterion**: Saccade executed after continue signal
- **Data markers**:
  - `type = 'CONT'`
  - `stop_cue = timestamp` (time of **green continue signal**, NOT a stop signal)
  - `ssd_number = 1.0-4.0` (continue signal delay level)
  - `first_relevant_saccade = timestamp` (saccade executed)
- **Alignment**: `go_cue` or `stop_cue`

**IMPORTANT**: The `stop_cue` column contains:
- Time of **stop signal** (red) for STOP trials
- Time of **continue signal** (green) for CONT trials
- `NaN` for GO trials

This naming convention can be confusing but reflects the experimental design where both signals appear at matched delays (CSD/SSD).

---

## Directional Targets

- **0°**: Right direction (rightward saccade)
- **180°**: Left direction (leftward saccade)
- Targets appear at equal eccentricity on opposite sides of the screen

---

## Signal Delay Levels

- **CSD (Continue Signal Delay)**: Time between go cue and continue signal
- **SSD (Stop Signal Delay)**: Time between go cue and stop signal
- **Levels**: 4 discrete values encoded as `ssd_number = 1.0, 2.0, 3.0, 4.0`
- **Effect**:
  - Shorter delays → easier inhibition (for STOP) / faster continue response (for CONT)
  - Longer delays → harder inhibition (for STOP) / approaching natural GO RT (for CONT)
- **Adaptive staircase**: Delays adjusted during session to maintain ~50% stop success rate

---

## Key Temporal Events

1. **Fixation period** (t < 0): Subject fixates at center waiting for target to appear
2. **go_cue** (t = 0): Visual target appears, signals saccade initiation (all trials)
3. **stop_cue**: Secondary signal appears (STOP/CONT trials only)
   - Red stop signal for STOP trials
   - Green continue signal for CONT trials
4. **first_relevant_saccade**: Array [start, end] of saccade times (if executed)
   - `first_relevant_saccade[0]`: Saccade start time (movement onset)
   - `first_relevant_saccade[1]`: Saccade end time (fixation on target)
5. **reaction_time**: Interval from `go_cue` to `first_relevant_saccade[0]`
   - Typical RT: 100-150 ms

---

## Experimental Rationale

The CONT trials serve as a critical control condition:
- Match the visual stimulation of STOP trials (a secondary cue appears)
- Control for visual distraction effects
- Isolate inhibitory processes specific to stopping vs. general cue processing
- Enable comparison: successful STOP vs. successful CONT (both inhibit vs. both execute)

---

**Related Documentation**:
- [Project Overview](PROJECT.md)
- [Data Structures](DATA_REFERENCE.md)
- [Workflows & Usage Patterns](WORKFLOWS.md)

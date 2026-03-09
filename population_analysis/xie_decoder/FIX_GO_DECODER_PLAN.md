# FIX vs GO Mode Neural Decoder - Implementation Plan

## Overview

Build a linear neural decoder to classify whether the neural population is in **FIX mode** (fixating, pre-go-cue) or **GO mode** (saccade preparation/execution, post-go-cue). The decoder will be trained exclusively on GO trials and later applied to STOP trials to investigate whether successful stopping involves maintaining a FIX-like state, transitioning to GO, or something inconclusive.

**Reference**: Xie et al. (2022) "Geometry of sequence working memory in macaque prefrontal cortex" - Figure 3A decoder architecture.

---

## 1. Data Selection & Filtering

### 1.1 Trial Selection Criteria

**Training & Test Data: GO Trials Only (Current Phase)**
- `type == 'GO'`
- `trial_failed == False` (successful trials)
- `reaction_time > 1.3 * mean_ssd2` where `reaction_time = first_relevant_saccade[0] - go_cue`
  - This ensures sufficient post-go-cue neural activity before movement onset
  - Guarantees we can extract the GO_late epoch [+1×mean_ssd, +2×mean_ssd] before saccade

**Future Phases (Deferred)**
- STOP trials (`type == 'STOP'`, `trial_failed == False`, `ssd_number == 2`)
- CONT trials (`type == 'CONT'`, `trial_failed == False`, `ssd_number == 2`)
- These will be addressed after validating decoder performance on GO trials

### 1.2 Direction-Specific Decoders

Build **separate decoders** for each direction:
- **Direction 0** (rightward saccades): `dir == 0`
- **Direction 180** (leftward saccades): `dir == 180`

This accounts for direction-selective neural tuning.

### 1.3 Computing mean_ssd2

Compute separately **per direction** using STOP trials (CONT trials could also be used):

```python
# For a given direction (0 or 180) and SSD level (2)
DIRECTION = 0  # or 180
SSD_NUM = 2

stop_trials = session.data[
    (session.data['type'] == 'STOP') &
    (session.data['trial_failed'] == False) &
    (session.data['dir'] == DIRECTION) &
    (session.data['ssd_number'] == SSD_NUM)
]
mean_ssd = stop_trials['ssd_len'].mean()
```

**Note**:
- GO trials have `ssd_len = NaN`, so we compute from STOP (or CONT) trials only
- `ssd_len` = stop_cue - go_cue (time in ms between go cue and signal)
- Compute separately for each direction to account for potential differences

---

## 2. Epoch Definition

All epochs are **aligned to go_cue** (t=0).

| Epoch | Time Range (ms) | Label | Description |
|-------|-----------------|-------|-------------|
| FIX_early | [-2×mean_ssd, -1×mean_ssd] | FIX | Early fixation period |
| FIX_late | [-1×mean_ssd, 0] | FIX | Late fixation, just before go cue |
| GO_early | [0, +1×mean_ssd] | GO | Early post-go-cue period |
| GO_late | [+1×mean_ssd, +2×mean_ssd] | GO | Late post-go-cue period |

**Configurable Parameters** (for later experimentation):
```python
epoch_multipliers = [
    (-2, -1, 'FIX'),  # (start_mult, end_mult, label)
    (-1, 0, 'FIX'),
    (0, 1, 'GO'),
    (1, 2, 'GO'),
]
```

---

## 3. Feature Extraction

### 3.1 Option A: Mean Firing Rate per Epoch (Recommended Start)

For each neuron, compute the **mean firing rate** within each epoch:

```python
def extract_epoch_firing_rate(spike_times, epoch_start, epoch_end):
    """Count spikes in epoch and convert to firing rate (Hz)"""
    spikes_in_epoch = spike_times[(spike_times >= epoch_start) & (spike_times < epoch_end)]
    duration_sec = (epoch_end - epoch_start) / 1000.0
    return len(spikes_in_epoch) / duration_sec
```

**Feature vector per trial-epoch**: `[FR_neuron1, FR_neuron2, ..., FR_neuronN]`

### 3.2 Option B: Time-Binned Activity (For Temporal Dynamics)

Divide each epoch into smaller bins (e.g., 10ms or 25ms bins) to preserve temporal dynamics:

```python
bin_size = 25  # ms
bins_per_epoch = int(mean_ssd2 / bin_size)
# Feature vector: [bin1_n1, bin1_n2, ..., bin1_nN, bin2_n1, ...]
```

**Trade-off**: More features but captures temporal structure.

### 3.3 Normalization

**Z-score normalization** per neuron (across all training trials):
```python
# For each neuron, compute mean and std across training data
neuron_mean = training_data[:, neuron_idx].mean()
neuron_std = training_data[:, neuron_idx].std()
normalized = (data - neuron_mean) / neuron_std
```

---

## 4. Decoder Architectures

### 4.1 Decoder A: Logistic Regression (Simple, Interpretable)

Binary classifier with L2 regularization:

```python
from sklearn.linear_model import LogisticRegression

decoder_LR = LogisticRegression(
    penalty='l2',
    C=1.0,  # Inverse regularization strength (tune via CV)
    solver='lbfgs',
    max_iter=1000,
    class_weight='balanced'  # Handle class imbalance
)
```

**Output**: `P(GO | neural_activity)` via sigmoid

**Interpretation**: Weight vector shows each neuron's contribution to GO vs FIX discrimination.

### 4.2 Decoder B: Xie-Style Linear Decoder (2D Hidden Layer)

Following Figure 3A architecture:

```python
import torch
import torch.nn as nn

class XieDecoder(nn.Module):
    def __init__(self, n_neurons, hidden_dim=2, n_classes=2):
        super().__init__()
        # Linear projection to hidden state
        self.W = nn.Linear(n_neurons, hidden_dim, bias=True)
        # Target vectors (learnable or fixed)
        self.M = nn.Linear(hidden_dim, n_classes, bias=False)

    def forward(self, x):
        h = self.W(x)  # Project to 2D hidden space
        logits = self.M(h)  # Compare with target vectors
        return logits, h  # Return both for visualization
```

**Training**:
- Loss: CrossEntropyLoss
- Optimizer: Adam (lr=1e-3)
- Regularization: L2 on W (weight_decay)

**Advantage**: 2D hidden space allows visualization of neural state trajectories.

---

## 5. Train/Test Split

### 5.1 Split Strategy

**Per-direction, per-session split** (or pooled across sessions):

```python
from sklearn.model_selection import train_test_split

# For each direction separately
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,  # Maintain FIX/GO ratio
    random_state=42
)
```

### 5.2 Data Organization

Each sample is a **(trial, epoch)** pair:
- Multiple epochs per trial (4 epochs × N trials)
- Labels: 0 = FIX, 1 = GO

---

## 6. Training Procedure

### 6.1 Logistic Regression

```python
# Simple fit
decoder_LR.fit(X_train, y_train)

# With cross-validation for C parameter
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
```

### 6.2 Xie Decoder

```python
# Training loop
optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

for epoch in range(n_epochs):
    optimizer.zero_grad()
    logits, h = decoder(X_batch)
    loss = criterion(logits, y_batch)
    loss.backward()
    optimizer.step()
```

---

## 7. Evaluation Metrics

### 7.1 Classification Performance

- **Accuracy**: Overall correct rate
- **Balanced Accuracy**: (Sensitivity + Specificity) / 2
- **AUC-ROC**: Area under ROC curve
- **Confusion Matrix**: FIX→FIX, FIX→GO, GO→FIX, GO→GO

### 7.2 Per-Epoch Performance

Evaluate separately for each epoch to see if decoder generalizes:
- Train on all epochs, test on each epoch separately
- Cross-temporal generalization matrix (train time × test time)

### 7.3 Confidence Calibration

Check if decoder probabilities are well-calibrated:
```python
from sklearn.calibration import calibration_curve
prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
```

---

## 8. Application to STOP Trials (FUTURE PHASE - Deferred)

**Note**: This section describes future analysis after decoder validation on GO trials is complete.

### 8.1 Decoding STOP Trial Epochs

Apply trained decoder to STOP trials at matched time points:

```python
# For STOP trials with ssd_number == 2
stop_epochs = [
    (-2*mean_ssd, -1*mean_ssd),  # Pre-go-cue (should be FIX)
    (-1*mean_ssd, 0),             # Just before go cue (should be FIX)
    (0, stop_cue_time),           # Go cue to stop cue (??? interesting!)
    (stop_cue_time, stop_cue_time + mean_ssd),  # Post-stop-cue (??? key period!)
]
```

### 8.2 Classification Rules

| P(GO) | Classification |
|-------|----------------|
| < 0.30 | **FIX** (confident) |
| 0.30 - 0.70 | **Inconclusive** |
| > 0.70 | **GO** (confident) |

### 8.3 Key Questions to Answer

1. **Pre-stop-cue period (go_cue to stop_cue)**:
   - Does neural state start transitioning toward GO?
   - Is the transition partial or similar to GO trials?

2. **Post-stop-cue period**:
   - Does neural state revert to FIX?
   - Does it stay in an intermediate state?
   - Is there a unique STOP state (neither FIX nor GO)?

3. **Comparison with CONT trials**:
   - CONT trials should show GO-like activity post-signal
   - Useful validation of decoder

---

## 9. Visualization

### 9.1 Decoder Performance

- ROC curves (per decoder type, per direction)
- Confusion matrices
- Learning curves (for Xie decoder)

### 9.2 Neural State Space (Xie Decoder)

- 2D scatter plot of hidden states colored by FIX/GO label
- Trajectories through time (for STOP trials)
- Separation boundary visualization

### 9.3 Time-Resolved Decoding

- P(GO) over time for GO vs STOP trials
- Aligned to go_cue and stop_cue
- Error bands across trials

### 9.4 Weight Interpretation (Logistic Regression)

- Neuron weights ranked by contribution
- Relationship to cell properties (baseline FR, direction selectivity)

---

## 10. Implementation Checklist

### Phase 1: Data Preparation
- [ ] Load cell_df and compute mean_ssd2
- [ ] Filter GO trials (successful, RT > 1.3×mean_ssd2)
- [ ] Split by direction (0° and 180°)
- [ ] Extract epochs and labels
- [ ] Z-score normalize features
- [ ] Train/test split (stratified)

### Phase 2: Decoder Training
- [ ] Implement Logistic Regression decoder
- [ ] Implement Xie-style decoder (PyTorch)
- [ ] Train both decoders (per direction)
- [ ] Evaluate on test set
- [ ] Save trained models

### Phase 3: Visualization & Interpretation (Current Focus)
- [ ] ROC curves and confusion matrices
- [ ] Neural state space plots (Xie decoder)
- [ ] Time-resolved P(GO) traces
- [ ] Weight interpretation

### Phase 4: STOP Trial Analysis (Future - Deferred)
- [ ] Extract STOP trial epochs (SSD2)
- [ ] Apply trained decoders
- [ ] Classify as FIX/GO/Inconclusive
- [ ] Compare pre- vs post-stop-cue
- [ ] Statistical analysis

### Phase 5: Extensions (Future)
- [ ] Other SSD levels (1, 3, 4)
- [ ] Failed STOP trials comparison
- [ ] Session-wise analysis
- [ ] Different epoch definitions

---

## 11. Code Organization

```
population_analysis/xie_decoder/
├── FIX_GO_DECODER_PLAN.md      # This document
├── xie_decoder.ipynb            # Main analysis notebook
├── decoder_utils.py             # Helper functions
│   ├── extract_epoch_features()
│   ├── compute_mean_ssd()
│   ├── filter_trials()
│   └── normalize_features()
├── models.py                    # Decoder classes
│   ├── XieDecoder (PyTorch)
│   └── Wrapper for sklearn LogisticRegression
└── visualization.py             # Plotting functions
    ├── plot_roc_curves()
    ├── plot_neural_state_space()
    └── plot_time_resolved_decoding()
```

---

## 12. Expected Outcomes

### If STOP = Maintained FIX State
- STOP trials post-stop-cue: P(GO) < 0.30
- Suggests successful stopping = preventing GO state transition

### If STOP = Partial GO Transition
- STOP trials post-go-cue: P(GO) intermediate (0.30-0.70)
- Suggests stopping catches movement preparation mid-transition

### If STOP = Unique State
- STOP trials systematically inconclusive
- Suggests stopping engages distinct neural mechanism
- Would need separate STOP-specific decoder

---

## References

- Xie et al. (2022). Geometry of sequence working memory in macaque prefrontal cortex. *Science*, 375, 632-639.
- Pani et al. (2022). [Reference for CSST task methodology]

---

*Plan created: January 2026*
*Project: Population Analysis - FIX/GO Neural Decoder*

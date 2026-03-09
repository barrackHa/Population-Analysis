# FIX vs GO Neural Decoder Analysis - Summary Report

**Session**: fi211025a
**Date**: January 2026
**Analysis Type**: Xie-style linear decoder for FIX/GO state classification

---

## Executive Summary

We implemented a neural decoder to distinguish between FIX (fixation) and GO (movement preparation) states in caudate neurons during a countermanding stop-signal task. While the decoder successfully discriminates between these states in GO trials (74-77% accuracy, 83-86% AUC-ROC), **it fails to reveal expected differences between successful STOP and GO trials**, suggesting the neural mechanisms of stopping may not be captured by this FIX/GO axis.

---

## What We Did: Implementation Overview

### Phase 1: Data Preparation
**Objective**: Prepare neural data for decoder training

**Method**:
- **Session**: fi211025a (63 neurons, 685 trials)
- **Training data**: GO trials only (successful, RT > 1.3×mean_ssd)
  - Right direction: 64 trials
  - Left direction: 63 trials
- **Epochs** (aligned to go_cue, relative to mean_ssd = 108ms):
  - `FIX_early`: [-216, -108] ms
  - `FIX_late`: [-108, 0] ms
  - `GO_early`: [0, +108] ms
  - `GO_late`: [+108, +216] ms
- **Features**: Mean firing rate per neuron per epoch
- **Normalization**: Z-score per neuron (using training statistics)
- **Split**: 80% train, 20% test (stratified by FIX/GO label)

**Output**: Normalized feature matrices for decoder training

---

### Phase 2: Decoder Training
**Objective**: Train two decoder architectures to classify FIX vs GO states

**Models**:
1. **Logistic Regression**
   - L2 regularization (C=1.0 via grid search CV)
   - Balanced class weights
   - Linear, interpretable weights

2. **Xie Decoder** (PyTorch)
   - 2D hidden layer for visualization
   - Architecture: neurons → 2D → 2 classes
   - 500 epochs, batch_size=64, lr=1e-3, weight_decay=0.01

**Performance (Test Set)**:

| Direction | Model | Accuracy | Balanced Acc | AUC-ROC |
|-----------|-------|----------|--------------|---------|
| Right | Logistic Regression | 76.56% | 76.56% | 0.8535 |
| Right | Xie Decoder | 74.22% | 74.22% | 0.8433 |
| Left | Logistic Regression | 74.60% | 74.60% | 0.8592 |
| Left | Xie Decoder | 73.02% | 73.02% | 0.8340 |

**Conclusion**: Both decoders successfully learned to discriminate FIX from GO states with good performance.

---

### Phase 3: Visualization & Interpretation
**Objective**: Understand what the decoders learned

**Key Findings**:

#### 1. 2D State Space Separation (Xie Decoder)
- **Right direction**:
  - FIX centroid: (-0.608, -0.456)
  - GO centroid: (0.894, 0.802)
  - Separation: 1.96 Euclidean units
- **Left direction**:
  - FIX centroid: (0.642, 0.638)
  - GO centroid: (-0.814, -0.694)
  - Separation: 1.97 Euclidean units

✓ Clear separation confirms the decoder captures meaningful FIX/GO structure

#### 2. Top Discriminative Neurons (Logistic Regression)
- **Right direction**: Cell 1599 (weight=+0.35, GO-preferring)
- **Left direction**: Cell 1572 (weight=-0.50, FIX-preferring)

#### 3. Per-Epoch Accuracy ⚠️ **CRITICAL FINDING**

| Epoch | Right Acc | Left Acc | Interpretation |
|-------|-----------|----------|----------------|
| FIX_early | 79.3% | 78.6% | ✓ Clear FIX state |
| FIX_late | 80.0% | 74.3% | ✓ Clear FIX state |
| **GO_early** | **44.8%** | **46.7%** | ⚠️ **Near chance!** |
| GO_late | 97.1% | 97.0% | ✓ Clear GO state |

**Key Observation**: The GO_early epoch (0 to +108ms post-go-cue) shows **near-chance accuracy**, indicating the neural population doesn't instantly transition to a clear GO state. The GO state only becomes well-defined in GO_late (+108 to +216ms).

**Implication**: There's a **transitional period** (~108ms) after the go cue where the neural state is ambiguous. This is exactly when stop signals arrive (mean_ssd = 108ms).

---

### Phase 4: Application to STOP Trials
**Objective**: Apply trained decoders to STOP trials to understand stopping mechanisms

**Trials Analyzed**:
- STOP trials: 13 (Right), 11 (Left) - all successful, SSD2
- CONT trials: 11 (Right), 24 (Left) - for validation

**Complete P(GO) Trajectory for STOP Trials**:

| Epoch | Time (rel. to go_cue) | Right P(GO) | Left P(GO) | State |
|-------|-----------------------|-------------|------------|-------|
| FIX_early | -162ms | 0.32 ± 0.12 | 0.35 ± 0.16 | ✓ FIX-like |
| FIX_late | -54ms | 0.37 ± 0.10 | 0.35 ± 0.16 | ✓ FIX-like |
| pre_signal | +54ms | 0.55 ± 0.17 | 0.54 ± 0.14 | ⚠️ Ambiguous |
| **post_signal** | **+162ms** | **0.79 ± 0.13** | **0.76 ± 0.12** | **❌ GO-like!** |

**Comparison with CONT trials (post-signal)**:
- CONT: P(GO) = 0.81 (Right), 0.75 (Left)
- STOP: P(GO) = 0.79 (Right), 0.76 (Left)

**Statistical Test**: Pre-signal vs Post-signal for STOP trials
- Right: t=-4.28, p=0.001 (significant increase)
- Left: t=-3.75, p=0.004 (significant increase)

---

## The Problem: Why These Results Are Concerning

### Expected vs. Observed

**What we expected**:
1. STOP trials would maintain FIX-like state (P(GO) < 0.3)
2. OR show intermediate state (0.3 < P(GO) < 0.7)
3. Clear difference from CONT trials post-signal

**What we observed**:
1. ❌ STOP trials show **GO-like state** (P(GO) > 0.7) post-stop-cue
2. ❌ **No meaningful difference** between STOP and CONT post-signal
3. ❌ Both show high P(GO) despite opposite behaviors (stopping vs. moving)

### Why This Is Problematic

1. **Behavioral dissociation**:
   - STOP trials: No movement (successful inhibition)
   - CONT trials: Saccade execution
   - **Yet neural states appear identical in decoder space**

2. **Missing the stopping signal**:
   - We trained the decoder to find FIX vs GO differences
   - But successful stopping doesn't involve maintaining FIX state
   - **The critical difference between STOP and GO must lie in dimensions orthogonal to the FIX/GO axis**

3. **Cannot identify divergence point**:
   - Original goal: Find when STOP and GO trajectories diverge
   - Current finding: They don't diverge in FIX/GO space
   - **We're looking in the wrong subspace**

---

## What We've Learned

### 1. The FIX/GO Decoder Works Well for GO Trials
- Successfully distinguishes fixation from movement preparation
- Clear temporal progression: FIX → Ambiguous → GO
- ~100ms transition period aligns with reaction time distributions

### 2. Stopping ≠ Maintaining Fixation
- **Critically important negative result**
- Stops don't work by "staying in FIX mode"
- Challenges simple "inhibit GO state" models

### 3. The Neural State Space is Multidimensional
- FIX/GO is one axis (captures ~74-77% of GO trial variance)
- Successful stopping must involve changes in **other dimensions**:
  - Possibly orthogonal inhibitory signals
  - Different population dynamics not captured by mean firing rates
  - Timing-dependent state trajectories beyond simple FIX/GO dichotomy

### 4. Caudate May Not Be the "Stopping" Region
Two possible interpretations:
a) **Stopping happens downstream**: Caudate encodes "intention" or "readiness", but actual motor inhibition occurs in motor cortex, brainstem, or spinal cord
b) **We need different features**: The stopping signal exists in caudate but requires different analysis (e.g., temporal dynamics, population geometry, spike timing)

---

## Critical Limitations

### 1. Training Set Bias
- **Decoder trained only on GO trials**
- Optimized to find FIX vs GO differences
- By definition, cannot detect dimensions that vary between STOP and GO

### 2. Epoch-Based Analysis
- Averages firing rates over 108ms windows
- Loses temporal dynamics within epochs
- Stopping might involve **precise timing** not captured by mean rates

### 3. Linear Decoder
- Assumes FIX and GO are linearly separable
- Stopping might involve **nonlinear** population dynamics
- The 2D Xie decoder constrains the solution space

### 4. Mean Firing Rate Features
- Ignores spike timing, synchrony, oscillations
- Stopping might involve **temporal codes** not in rate

### 5. Single Session, Small Sample
- Session fi211025a: Only 13 (Right) and 11 (Left) STOP trials
- Limited statistical power
- Cannot assess cross-session generalization

---

## Why We Can't Find When Differences Appear

The fundamental issue: **We're constrained to the FIX/GO manifold**

1. **Decoder definition**: Trained to maximize FIX vs GO separation
   - Finds the direction in neural space with maximum variance between fixation and movement preparation
   - This direction may be **orthogonal** to the STOP vs GO difference

2. **Projection problem**:
   - We project high-dimensional neural activity onto a 1D axis (P(GO))
   - If STOP vs GO differences are in orthogonal dimensions, projection collapses them
   - Like looking at 3D objects from an angle that makes them indistinguishable

3. **Temporal resolution**:
   - 108ms epochs are coarse
   - Critical STOP-specific signals might occur at finer timescales (10-20ms)
   - Averaging obscures rapid dynamics

4. **The "when" question requires knowing "what"**:
   - To find when STOP/GO diverge, we need to know what signal to track
   - We assumed the signal was FIX/GO probability
   - **But that assumption appears to be wrong**

---

## Next Steps: Alternative Approaches

### Immediate Priorities

#### 1. **Unconstrained Population Analysis**
Instead of forcing data through FIX/GO decoder:
- **Full PCA on STOP, GO, and CONT trials combined**
- Find the dimensions that best separate STOP from GO
- Don't assume it's the same as FIX vs GO

```python
# Pseudo-code
all_trials = [STOP_trials, GO_trials, CONT_trials]
pca = PCA(n_components=10)
pca.fit(all_trials)
# Analyze: Which PCs separate STOP from GO?
```

#### 2. **Time-Resolved Analysis**
Replace epoch-based with sliding windows:
- 10-20ms bins, stepped by 5ms
- Track P(GO) continuously from -200ms to +300ms
- Allows finer temporal resolution to catch rapid changes

#### 3. **Multivariate Pattern Analysis (MVPA)**
- Train decoder to distinguish **STOP from GO** (not FIX from GO)
- Use both as training data
- Find what distinguishes successful stopping from movement

#### 4. **Failed STOP Trials**
Critical comparison:
- Successful STOP vs Failed STOP trials
- Both receive stop signal, different outcomes
- Difference must contain stopping mechanism

### Methodological Alternatives

#### 5. **Temporal Dynamics**
Beyond mean rates:
- **Spike timing precision**: Do STOP trials show tighter temporal locking?
- **Oscillatory activity**: Theta/beta power differences?
- **Population coupling**: Cross-neuron correlations or synchrony?

#### 6. **Single-Trial Geometry**
- **Neural trajectories** in state space
- **Trajectory curvature** or **velocity** differences
- **Dimensionality** (does STOP constrain trajectories to lower dimensions?)

#### 7. **Different Brain Regions**
- Caudate may primarily encode preparation/intention
- **Stopping signal might originate elsewhere**:
  - Superior colliculus (eye movement control)
  - Frontal eye fields (saccade command)
  - Pre-supplementary motor area (inhibitory control)

### Statistical Considerations

#### 8. **Cross-Session Analysis**
- Replicate in multiple sessions
- Pool data for higher statistical power
- Assess generalization across animals

#### 9. **Parametric Manipulation**
- **SSD levels**: Do longer SSDs show higher P(GO)?
- **Failed stops**: Do they have even higher P(GO)?
- **Trial difficulty**: Correlate P(GO) with RT or saccade metrics

---

## Scientific Interpretation

### What This Tells Us About Stopping

This is actually a **scientifically important negative result**:

#### The "Active Inhibition" Model is Too Simple
- Stopping ≠ reverting to fixation
- Stopping ≠ preventing GO state activation
- The neural population enters a GO-like state even during successful stops

#### Implications for Stopping Mechanisms

**Scenario A: Downstream Inhibition**
- Caudate neurons encode "action readiness" or "intention"
- The stop signal is implemented in downstream motor structures
- Caudate shows GO-like activity because the "plan" is still there, just blocked at execution

**Scenario B: Orthogonal Inhibitory Signals**
- Stopping involves an independent inhibitory dimension
- This dimension is orthogonal to FIX/GO axis
- Current decoder simply doesn't "see" this dimension

**Scenario C: Timing-Dependent Commitment**
- GO-like state early in preparation is not yet "committed"
- Stop signal prevents transition from "prepared" to "committed"
- Commitment might involve timing or synchrony, not mean rates

#### Implications for Caudate Function
- May be more about action selection/intention than motor execution
- Movement inhibition may be implemented elsewhere
- Caudate contributes to "what" and "whether", not "how"

---

## Recommendations

### For This Dataset

1. **Urgent**: Run unconstrained PCA on STOP, GO, and CONT combined
2. **Priority**: Analyze failed STOP trials as control condition
3. **Important**: Time-resolved analysis (sliding windows, not epochs)
4. **Consider**: Include frontal eye field or superior colliculus recordings if available

### For Future Experiments

1. **Record from motor output stages** (FEF, SC, motor cortex) simultaneously
2. **Parametric SSD manipulation** with more trials per condition
3. **Include explicit fixation-only trials** for better FIX state definition
4. **Measure muscle activity (EMG)** to detect covert preparation

### For Analysis Methods

1. **Don't constrain to predefined axes** - let data reveal relevant dimensions
2. **Use multiple feature types** - rates, timing, synchrony, oscillations
3. **Single-trial analysis** - avoid averaging away critical variance
4. **Cross-validate between sessions** - ensure generalization

---

## Conclusion

We successfully built and validated a FIX/GO neural decoder that works well for distinguishing fixation from movement preparation in GO trials (74-77% accuracy). However, when applied to STOP trials, **we discovered that successful stopping does not work by maintaining a FIX-like neural state**. Instead, the neural population in successful STOP trials is virtually indistinguishable from CONT trials in the FIX/GO subspace (both show P(GO) > 0.75).

**This is a critical negative finding**: The mechanism of successful stopping in caudate neurons is **not captured by the FIX vs GO dimension**. The difference between stopping and going must lie in:
- Orthogonal neural dimensions not captured by this decoder
- Temporal dynamics beyond mean firing rates
- Downstream motor structures rather than caudate
- Or some combination of the above

**We cannot currently identify when STOP and GO trajectories diverge** because we are constrained to a subspace where they appear similar. To address this, we need unconstrained dimensionality reduction (PCA on combined STOP/GO data) and finer temporal resolution to capture rapid dynamics that may differentiate successful stopping from movement execution.

The FIX/GO decoder is a valuable tool for understanding movement preparation dynamics, but it is **not the right tool** for understanding stopping mechanisms. We need analysis approaches that don't presuppose the relevant dimensions.

---

## Files Generated

### Notebooks
- `xie_decoder_phase1.ipynb` - Data preparation
- `xie_decoder_phase2.ipynb` - Decoder training
- `xie_decoder_phase3.ipynb` - Visualization & interpretation
- `xie_decoder_phase4.ipynb` - STOP trial analysis

### Data
- `data/decoder_data/decoder_data_fi211025a_ssd2.pkl` - Prepared features
- `data/decoder_models/lr_decoder_fi211025a_ssd2_*.joblib` - Trained LR models
- `data/decoder_models/xie_decoder_fi211025a_ssd2_*.pt` - Trained Xie models
- `data/decoder_models/results_summary_fi211025a_ssd2.pkl` - Performance metrics

### Documentation
- `FIX_GO_DECODER_PLAN.md` - Original implementation plan
- `DECODER_ANALYSIS_SUMMARY.md` - This document

---

**Analysis Date**: January 2026
**Analyst**: Claude & Barak
**Status**: Complete - Negative result, alternative approaches needed

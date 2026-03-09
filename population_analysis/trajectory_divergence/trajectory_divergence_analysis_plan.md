# N-dimentional neurla trajectory divergence analysis

## Day1A plan (session-wise, Go-aligned, 10 ms bins, no smoothing)

### Data subsets (SSD2 focus)

Do everything **within each session** and **within each direction**.

For a given session:

* **Train/calibrate baseline on GO trials**:

  * Go-left baseline uses all **GO + Left** trials (e.g., 260).
  * Go-right baseline uses all **GO + Right** trials (e.g., 259).
* **Score STOP SSD2 trials** separately for each direction:

  * Score **STOP + Left + SSD2** against the Go-left baseline.
  * Score **STOP + Right + SSD2** against the Go-right baseline.

(We ignore CONT for now. Later we will add an additional “should look Go-like” sanity check.)

### Time axis and binning

* Analysis window: **(-50 ms, +300 ms)** relative to **go cue**.
* Bin size: **10 ms**, no smoothing.
* Convert each trial’s 1 kHz spike train to counts:
  [
  y_{n,t} \in {0,1,2,\dots}, \quad t=1,\dots,T\ (T=35\ \text{bins})
  ]

### Neuron inclusion (drop low firing rate)

For each session and direction baseline (GoL or GoR), compute each neuron’s mean firing rate across **GO trials** in the window:
[
\bar r_n = \frac{\sum_{trials}\sum_{t} y_{n,t}}{#trials \cdot T \cdot \Delta}
]
Drop neurons with (\bar r_n < r_{\min}) (configurable; start with e.g. **1 Hz**). Keep the same neuron set for scoring STOP in that session/direction.

---

## Model + scoring (Poisson mean trajectory, no PCA)

### 1) Mean Go trajectory (per neuron, per time bin)

From GO baseline trials (train split), estimate:
[
\hat\mu_{n,t} = \mathbb{E}[y_{n,t}\mid \text{GO, dir}]
]
No smoothing. Add a small floor to avoid zeros:
[
\hat\mu_{n,t} \leftarrow \max(\hat\mu_{n,t}, \epsilon)
]
with (\epsilon) small (e.g. (10^{-6})).

### 2) Per-bin negative log-likelihood (“surprise”) under Go baseline

For a single trial:
[
s(t) = -\sum_{n=1}^{N} \log \text{Poisson}\big(y_{n,t}; \hat\mu_{n,t}\big)
]
Compute using stable log-factorial via `gammaln(y+1)`.

### 3) Cumulative surprise trace

[
S(t) = \sum_{\tau=1}^{t} s(\tau)
]
This is your **trace**. Also store (s(t)) itself.

---

## Divergence time: “first time it stops looking like GO” (calibrated, family-wise)

You want an honest “unlikely GO” call over many time points, so we calibrate the threshold from held-out GO trials.

### Split GO trials into:

* **train** (fit (\hat\mu_{n,t}))
* **calibration** (estimate what GO looks like under the fitted baseline)

Split the GO data: 70/30, stratified by direction (direction handled separately anyway).

### Compute calibration statistics on held-out GO

For each held-out GO trial (k), compute (S_k(t)).

Compute the timewise mean and std across calibration GO trials:
[
m(t)=\mathbb{E}_k[S_k(t)], \quad \sigma(t)=\mathrm{Std}_k[S_k(t)]
]

Define a standardized trace:
[
z_k(t)=\frac{S_k(t)-m(t)}{\sigma(t)+\delta}
]
((\delta) small for stability, e.g. (10^{-9}).)

For each calibration GO trial, take the **max over time**:
[
z^{\max}_k = \max_t z_k(t)
]

Pick a family-wise false-alarm rate (\alpha) (default e.g. **0.001** or **0.0013** if you want the “3 SD tail” spirit). Set:
[
z_\text{thr} = \text{Quantile}_{1-\alpha}\left({z^{\max}_k}\right)
]

### Divergence time for a STOP trial

Compute its (S(t)) and
[
z(t)=\frac{S(t)-m(t)}{\sigma(t)+\delta}
]
Then define:
[
\tau = \min{t: z(t) > z_\text{thr}}
]
If it never crosses, set (\tau = \text{None}).

Report (\tau) both:

* in **bin index**
* in **milliseconds relative to go cue** (bin-center time)
  Optionally also report relative to the trial’s stop cue time:
  [
  \tau_{\text{rel-stop}} = \tau_{\text{ms}} - SSD_{\text{ms}}
  ]

---

## Outputs description

Per session × direction:

* Fitted (\hat\mu_{n,t})
* Calibration (m(t), \sigma(t), z_\text{thr})
* For each STOP SSD2 trial:

  * divergence time (\tau) (ms)
  * trace (S(t))
  * trace (z(t))

Across sessions:

* Distribution of divergence times for STOP SSD2 (Left and Right separately; and pooled if desired)
* Optional: compare divergence distributions by stop success vs stop fail (if labels exist)

---

## Agent prompt 


You are writing Python code to implement “Day1A” divergence-time analysis for neural spiking trials in a stop-signal task.

Goal:
- For each session and direction separately, fit a GO baseline model (Poisson mean trajectory) from GO trials.
- Score STOP trials (SSD2 only) against the corresponding GO baseline and compute:
  (1) divergence time tau
  (2) cumulative surprise trace S(t)
  (3) standardized trace z(t)
- Aggregate divergence times across sessions into a distribution.

Constraints / choices:
A) Bin spikes into 10 ms bins with NO smoothing.
B) Analyze time window (-50 ms, +300 ms) relative to GO cue.
C) Focus on SSD2 STOP trials (SSD2 approx 108 ms but use per-trial SSD_ms if provided).
D) No z-scoring and no baseline subtraction.
E) Drop neurons below a minimum mean firing rate threshold r_min (Hz), computed on GO trials within the analysis window, per session+direction.
F) Output tau, traces, and divergence-time distributions.

Definitions:
- For a session with N neurons, represent each trial as counts y[n, t] in 10 ms bins over T=35 bins spanning -50..300 ms.
- Fit GO baseline per direction:
  mu_hat[n,t] = mean over GO trials (train split) of y[n,t], with mu_hat floored by eps to avoid zeros.
- For any trial, per-bin negative log likelihood under Poisson(mu_hat):
  s(t) = -sum_n log PoissonPMF(y[n,t]; mu_hat[n,t])
  Use stable log-factorial with scipy.special.gammaln(y+1).
- Cumulative surprise:
  S(t) = cumsum_t s(t)

Calibration for family-wise threshold:
- Split GO trials of the given direction into train/calibration (default 70/30, random seed configurable).
- Use train to fit mu_hat.
- For each calibration GO trial k, compute S_k(t) using mu_hat.
- Compute m(t)=mean_k S_k(t), sigma(t)=std_k S_k(t).
- z_k(t)=(S_k(t)-m(t))/(sigma(t)+delta)
- z_max_k = max_t z_k(t)
- Choose alpha (default 0.001). Set z_thr = quantile_{1-alpha} of {z_max_k}.
- For each STOP SSD2 trial, compute S(t) and z(t) using m(t), sigma(t), then divergence time:
  tau = first t with z(t) > z_thr. If none, tau=None.
- Convert tau to ms relative to go cue using bin centers. Also compute tau_rel_stop = tau_ms - SSD_ms if SSD_ms exists.

Data interface (make minimal assumptions but provide adapters):
- Expect input as a list of sessions.
- Each session provides:
  - spikes: array shape (n_trials, n_neurons, n_time_ms) at 1 kHz aligned to GO cue (t=0 at go).
    OR already-binned counts; support both via a flag.
  - trial metadata arrays length n_trials:
    - trial_type in {"GO","STOP","CONT"}
    - direction in {"Left","Right"} (or 0/1)
    - SSD_id in {1,2,3,4} or SSD_ms (STOP/CONT only)
    - optional stop_success boolean (STOP only)
- Implement a binning function that extracts the [-50,300] ms window and bins into 10 ms counts.

Implementation requirements:
- Write clean, modular code:
  - bin_trials(...)
  - select_trials(...)
  - drop_low_fr_neurons(...)
  - fit_go_mean(mu_hat)
  - compute_surprise_and_cumsum(y, mu_hat)
  - calibrate_threshold(go_calib_trials, mu_hat, alpha) -> m(t), sigma(t), z_thr
  - score_stop_trials(stop_trials, mu_hat, m, sigma, z_thr) -> per-trial results
  - aggregate_across_sessions(...)
- Provide plotting utilities:
  - plot_single_trial_trace(time_ms, S, z, z_thr, tau_ms)
  - plot_divergence_distribution(tau_ms_list) (histogram/ECDF)
- Save outputs as a structured dict per session and a combined summary (e.g. JSON/pickle + CSV table of trial-level results).

Deliverables:
- A single runnable Python module (or small package) with an example “main()” that:
  - loads/accepts sessions data (mocked if necessary)
  - runs analysis for SSD2
  - prints summary stats and generates plots
  - writes outputs to disk

Be careful about:
- eps floor for mu_hat to avoid log(0)
- handling neurons dropped (apply same neuron mask to STOP scoring)
- reproducible random splits (seed)
- missing SSD for GO trials (ignore)
- time axis correctness (bin centers)


General instructions: 
- Use functionality imlimented in the Cell and Session classes instead of rewrting it.
- When needed use inheritance or outside functions to increase capabilities - leave the existing classes as is.
- Keep the code short and clean. Use functions and parapeters. 
- This is a POC - please keep to the plan and to necessary things only so we can see if it works. 
- Keep everything in a single ipynb file in the trajectory_divergence sub-folder.
- Don't do everything at once - keep me in the loop. Let's use a short feedbak loop. After each functionality block stop and let me see. I'll give you feedback if I have any anwe'll continue from there. 
- Ask if you need clarifications.
- Use session fi211115a to begin with.
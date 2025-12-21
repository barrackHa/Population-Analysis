* **Event slot 1 (GO-plan):** direction (L/R)
* **Event slot 2 (second cue):** NONE (GO), STOP at SSD1–4, or CONT at SSD1–4 (and optionally STOP outcome: success/fail)

This mirrors their “rank × item” factorization, where each trial turns on exactly one regressor per slot (their three-hot sequence representation and late-delay scalar response are explicit in their Methods). 

## 1) What to regress, and when

Xie regress a **single scalar per trial**: the neuron’s **average response in a fixed epoch**, then fit coefficients with Lasso and repeated half-splits for stability. 

For stop-signal, you want the epoch to be *comparable across GO/STOP/CONT* and not trivially confounded by “time since GO”:

* Pick an epoch **locked to the second cue time** (STOP/CONT) because that is when “cancel vs continue” information arrives.
* For GO trials (no second cue), create a **pseudo second-cue time**: sample an SSD from the empirical SSD distribution and set `t2 = t_go + sampled_SSD`. Then compute the same cue-locked neural epoch on GO trials.

A typical choice is something like mean firing rate in ([t2 + 0,\mathrm{ms},\ t2 + 150,\mathrm{ms}]) (or up to SSRT if you want to emphasize cancellation), but the exact window is your design decision.

## 2) Design matrix: a “two-hot” compositional code

Let each trial activate exactly:

* one **GO-direction** regressor
* one **second-cue** regressor

### Minimal model (no STOP outcome)

* Slot 1: `GO_L`, `GO_R`  (2 regressors)
* Slot 2: `NONE` (GO trials), `STOP_SSD1..4` (4), `CONT_SSD1..4` (4)  → 9 regressors

Total: 11 + intercept.

### Outcome-aware model (recommended if you have enough trials)

Split STOP trials into success vs fail:

* Slot 2 becomes: `NONE` (1) + `STOPsucc_SSD1..4` (4) + `STOPfail_SSD1..4` (4) + `CONT_SSD1..4` (4) → 13 regressors

Total: 15 + intercept.

This is directly analogous to Xie’s “one regressor per rank-item combination,” fit with Lasso and repeated half-splits. 

## 3) What this model approximates

For neuron (i), trial (k):
[
y_i(k)\approx \beta_i^{\text{GO}}(\text{dir}_k) ;+; \beta_i^{\text{2nd}}(\text{type}_k,\text{SSD}_k,\text{outcome}_k);+;\epsilon_i(k).
]

So (\beta^{\text{2nd}}(\text{STOP},\text{SSD3},\text{succ})) is not “the mean firing rate” in that condition; it is the **marginal component** the model assigns to that slot, just like in Xie.

## 4) From betas to geometry (the PCA step)

After fitting all neurons, build **population vectors** for each regressor:
[
v(\text{feature}) = [\beta_1(\text{feature}),\dots,\beta_N(\text{feature})]^\top.
]

Then do what Xie do: **group vectors by a factor and PCA within the group** (they divide 18 vectors into 3 rank groups and PCA each). 

For stop-signal you might do:

* PCA on `{v(STOPsucc_SSD1..4)}` → “successful stopping subspace”
* PCA on `{v(CONT_SSD1..4)}` → “continue subspace”
* PCA on `{v(STOPfail_SSD1..4)}` → “failed-stop (go-dominated) subspace”

Then quantify relations via **principal angles** / variance overlap (same spirit as their rank-subspace orthogonality analysis).

## 5) Pseudocode outline

```python
# trials: list of dicts with fields:
#   dir ∈ {L,R}
#   type ∈ {GO, STOP, CONT}
#   SSD ∈ {1,2,3,4}  (only if type in {STOP, CONT})
#   outcome ∈ {succ, fail} (only if type==STOP)
#   t_go, t_stop_or_cont (if present), spikes[i][t] for neuron i

# -------------------------
# 1) define trial-wise scalar responses y_i(k)
# -------------------------
def second_cue_time(trial, SSD_sampler):
    if trial["type"] == "GO":
        ssd = SSD_sampler()                # sample from empirical SSD distribution
        return trial["t_go"] + ssd
    else:
        return trial["t_stop_or_cont"]     # actual second cue time

def response_in_window(spikes_i, t2, win=(0, 0.150)):
    return mean_rate(spikes_i, t2 + win[0], t2 + win[1])

# -------------------------
# 2) build two-hot design X
# -------------------------
# Slot 1 columns: GO_L, GO_R
# Slot 2 columns: NONE,
#                 STOPsucc_SSD1..4, STOPfail_SSD1..4, CONT_SSD1..4
P = 2 + 1 + 4 + 4 + 4   # = 15
X = zeros((n_trials, P))

for k, tr in enumerate(trials):
    # slot 1
    X[k, col("GO_"+tr["dir"])] = 1

    # slot 2
    if tr["type"] == "GO":
        X[k, col("NONE")] = 1
    elif tr["type"] == "CONT":
        X[k, col(f"CONT_SSD{tr['SSD']}")] = 1
    elif tr["type"] == "STOP":
        X[k, col(f"STOP{tr['outcome']}_SSD{tr['SSD']}")] = 1

# -------------------------
# 3) fit per-neuron Lasso with repeated half-splits (Xie-style)
# -------------------------
betas = zeros((N, P))

for i in range(N):
    y = zeros(n_trials)
    for k, tr in enumerate(trials):
        t2 = second_cue_time(tr, SSD_sampler)
        y[k] = response_in_window(spikes[i], t2, win=(0, 0.150))

    beta_samples = []
    for rep in range(100):
        idx = permute(n_trials)
        A, B = idx[:n_trials//2], idx[n_trials//2:]

        alphaA = choose_alpha_ml_or_cv(X[A], y[A])
        alphaB = choose_alpha_ml_or_cv(X[B], y[B])

        beta_samples.append(lasso_fit(X[A], y[A], alphaA).coef_)
        beta_samples.append(lasso_fit(X[B], y[B], alphaB).coef_)

    betas[i, :] = mean(beta_samples, axis=0)

# -------------------------
# 4) population vectors + PCA per group
# -------------------------
def popvec(feature):         # length N
    return betas[:, col(feature)]

STOPsucc = stack([popvec(f"STOPsucc_SSD{s}") for s in [1,2,3,4]], axis=0)  # 4×N
CONT     = stack([popvec(f"CONT_SSD{s}")     for s in [1,2,3,4]], axis=0)  # 4×N
STOPfail = stack([popvec(f"STOPfail_SSD{s}") for s in [1,2,3,4]], axis=0)  # 4×N

basis_STOPsucc = PCA(n_components=2).fit(demean(STOPsucc)).components_.T   # N×2
basis_CONT     = PCA(n_components=2).fit(demean(CONT)).components_.T
basis_STOPfail = PCA(n_components=2).fit(demean(STOPfail)).components_.T

# compare subspaces: principal angles(basis_STOPsucc, basis_CONT), etc.
```

With spikes (1 kHz bins) you can keep the *exact* logic of Xie et al.’s pipeline - **(i) regress trial-wise activity on a sparse, compositional design matrix; (ii) treat fitted (\beta)’s as population vectors; (iii) run PCA within groups to define subspaces**. In their case the “groups” were ranks (6 locations each), and PCA within each rank produced rank subspaces. 

Below is a concrete stop-signal analogue for your GO/STOP/CONT × L/R × SSD1–4 dataset.

---

## 1) Make stop-signal trials into a 2-slot “sequence”

Think of each trial as two slots (“ranks”):

* **Slot 1 (GO instruction):** direction (\in{\mathrm{L},\mathrm{R}})
* **Slot 2 (post-GO cue):** one of

  * **NONE** (for GO trials)
  * **STOP_SSD1..4** (and optionally split into STOP_succ vs STOP_fail)
  * **CONT_SSD1..4**

This is the direct analogue of Xie’s “one-hot vectors as task variables” and “linear combination of task variables” idea. 

### Key practical issue: GO trials don’t have a real SSD

If you want slot-2-aligned analyses (recommended), you need a *matched* time on GO trials. Two standard choices:

1. **Pseudo-SSD assignment**: for each GO trial, randomly assign SSD1–4 according to the empirical SSD distribution, and define a pseudo cue time (t_2=t_\mathrm{GO}+SSD).
2. **Stratified GO controls**: split GO trials into four “GO@SSD1..4” groups by sampling without replacement to match trial counts per SSD.

Either way lets you compare neural activity at equal “time since GO” across GO/STOP/CONT.

---

## 2) Choose the dependent variable (y_i(k)) from spikes

Xie regress a *single scalar per trial* (their mean activity in a fixed epoch) and fit (\beta)’s per neuron. 

For striatal spikes, do the same but use **spike counts** (or mean firing rate) in a cue-locked window:

* Align each trial to slot 2 time (t_2) (real STOP/CONT cue; pseudo for GO).
* Define (y_i(k)=) spike count in ([t_2 + a,\ t_2 + b]).

Practical bin/window choices for striatum:

* Use **20–50 ms bins** (or a 100–200 ms window) so counts aren’t almost all zeros.
* Consider **baseline subtraction** (e.g., subtract mean rate in ([t_2-300,t_2-100]) ms) if you want coefficients to reflect modulation rather than absolute rate.

You can keep linear regression like Xie, or switch to a Poisson GLM; the geometry step works either way as long as (\beta)’s are comparable across conditions.

---

## 3) Design matrix (X): “two-hot” compositional coding

Example with outcome split:

* Slot 1 columns: `GO_L`, `GO_R`  (2)
* Slot 2 columns:
  `NONE` (1)
  `STOPsucc_SSD1..4` (4)
  `STOPfail_SSD1..4` (4)
  `CONT_SSD1..4` (4)

Total = 15 predictors (+ intercept). Each trial has exactly **two 1’s**: one in slot 1 and one in slot 2.

This mirrors Xie’s approach: define task-variable one-hots; fit a linear model with Lasso and repeated half splits. 

---

## 4) “Rank subspaces” become “cue subspaces”

After fitting all neurons you will have population vectors like
[
v(\text{STOPsucc_SSD3}) = [\beta_1,\dots,\beta_N]^\top.
]

Now do the Xie step: **group vectors and PCA within-group** to get a low-dim subspace that captures variance across SSD levels *within that cue type* (analogous to “within-rank PCA across items”). 

Natural groupings:

* STOP-success group: {STOPsucc_SSD1..4} → “stopping subspace”
* CONT group: {CONT_SSD1..4} → “continue/control subspace”
* STOP-fail group: {STOPfail_SSD1..4} → “failed-stop / go-execution subspace”
* GO group (optional): {GO@SSD1..4} if you constructed it → “pure go-timing subspace”

Then compare subspaces with **principal angles** (exactly what they do for rank subspaces). 

---

## 5) Pseudocode: Xie-style regression for striatal spikes

```python
# trials: list with fields
#   dir ∈ {L,R}
#   type ∈ {GO, STOP, CONT}
#   SSD ∈ {1,2,3,4} (STOP/CONT)
#   outcome ∈ {succ, fail} (STOP)
#   t_go, t_cue2 (STOP/CONT), spikes[i][t] in 1 ms bins

# ----- choose cue2 time (pseudo for GO) -----
def cue2_time(tr, sample_SSD):
    if tr["type"] == "GO":
        ssd = sample_SSD()                # draw from empirical SSD distribution
        return tr["t_go"] + ssd
    return tr["t_cue2"]

# ----- response definition from spikes -----
def spike_count(spikes_1ms, t0_ms, t1_ms):
    return spikes_1ms[t0_ms:t1_ms].sum()

win = (0, 150)  # ms after cue2; adjust based on your question

# ----- build design matrix X (two-hot) -----
cols = make_columns([
  "GO_L","GO_R",
  "NONE",
  "STOPsucc_SSD1","STOPsucc_SSD2","STOPsucc_SSD3","STOPsucc_SSD4",
  "STOPfail_SSD1","STOPfail_SSD2","STOPfail_SSD3","STOPfail_SSD4",
  "CONT_SSD1","CONT_SSD2","CONT_SSD3","CONT_SSD4"
])
P = len(cols)
X = zeros((n_trials, P))

for k,tr in enumerate(trials):
    X[k, cols[f"GO_{tr['dir']}"]] = 1

    if tr["type"] == "GO":
        X[k, cols["NONE"]] = 1
    elif tr["type"] == "CONT":
        X[k, cols[f"CONT_SSD{tr['SSD']}"]] = 1
    elif tr["type"] == "STOP":
        X[k, cols[f"STOP{tr['outcome']}_SSD{tr['SSD']}"]] = 1

# ----- fit per neuron, with half-split stability like Xie -----
betas = zeros((N, P))

for i in range(N):
    y = zeros(n_trials)
    for k,tr in enumerate(trials):
        t2 = cue2_time(tr, sample_SSD)
        y[k] = spike_count(spikes[i], t2+win[0], t2+win[1])

    beta_samples = []
    for rep in range(100):
        idx = permute(n_trials)
        A, B = idx[:n_trials//2], idx[n_trials//2:]

        # Xie: Lasso + choose amplitude by maximum likelihood, repeat splits. :contentReference[oaicite:6]{index=6}
        alphaA = select_alpha_ml_or_cv(X[A], y[A])
        alphaB = select_alpha_ml_or_cv(X[B], y[B])

        beta_samples += [lasso_fit(X[A], y[A], alphaA).coef_,
                         lasso_fit(X[B], y[B], alphaB).coef_]

    betas[i,:] = mean(beta_samples, axis=0)

# ----- population vectors + PCA within cue groups -----
def popvec(feature):  # length N
    return betas[:, cols[feature]]

STOPsucc = stack([popvec(f"STOPsucc_SSD{s}") for s in [1,2,3,4]], axis=0)  # 4×N
CONT     = stack([popvec(f"CONT_SSD{s}")     for s in [1,2,3,4]], axis=0)
STOPfail = stack([popvec(f"STOPfail_SSD{s}") for s in [1,2,3,4]], axis=0)

basis_STOPsucc = PCA(2).fit(demean(STOPsucc)).components_.T  # N×2
basis_CONT     = PCA(2).fit(demean(CONT)).components_.T
basis_STOPfail = PCA(2).fit(demean(STOPfail)).components_.T

# compare subspaces via principal angles (as in Xie). :contentReference[oaicite:7]{index=7}
```

---

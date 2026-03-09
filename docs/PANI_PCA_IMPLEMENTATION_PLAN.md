### PCA Methodology in Pani et al. (2022)

The core idea is to decompose the neural state space into two orthogonal subspaces that are functionally meaningful for movement generation and inhibition.

1.  **Data Preparation**:
    *   They use Spike Density Functions (SDFs), created by convolving spike trains with a function mimicking a postsynaptic potential. This is similar to your smoothed PSTHs.
    *   Trials are grouped by condition and outcome (e.g., correct-stop, wrong-stop, no-stop). No-stop and wrong-stop trials are further grouped by reaction time (deciles or tertiles).
    *   They create an average population activity matrix for each group.

2.  **Initial PCA**:
    *   A large matrix is formed by concatenating the average population activities for all groups: `N (neurons) x Ctg (conditions x time bins x groups)`.
    *   This matrix is normalized (mean subtraction and division by SD across all conditions).
    *   A standard PCA is performed on this large matrix to get the first few principal components (PCs), which define the initial low-dimensional space (typically 3D).

3.  **Identifying Functional Subspaces (The Core of the Method)**:
    *   **Holding Plane**: They first identify a "holding plane". This plane is defined as the plane that best fits the trajectories of *correct-stop* trials, using Singular Value Decomposition (SVD). This plane represents the subspace where activity is confined during movement inhibition.
    *   **Holding-and-Planning Axis (HPA)**: Within the holding plane, they define the HPA. This axis is the line that best fits the initial part (first 300ms after Go cue) of *all* trial trajectories projected onto the holding plane. This axis captures the dynamics of motor planning.
    *   **Planning-and-Execution Axis (PEA)**: The PEA is defined as the axis *orthogonal* to the holding plane. It captures the activity that leads to movement execution, i.e., the "escape" from the holding plane.

4.  **Analysis**:
    *   Neural trajectories are projected onto the HPA and PEA.
    *   The paper shows that activity along the HPA ramps up for all trials (planning), but only when activity along the PEA crosses a threshold does a movement occur.

### Comparison to Your Current Method

| Feature | Current Method | Pani et al. Method |
| :--- | :--- | :--- |
| **Goal** | General dimensionality reduction. | Identify functional subspaces (Holding vs. Execution). |
| **Input Matrix** | `(n_cells, n_time_bins * n_conditions)` | `(n_cells, n_time_bins * n_conditions * n_groups)` |
| **Core Technique** | Standard PCA on a condition-concatenated matrix. | PCA followed by SVD to find specific axes (HPA, PEA). |
| **Key Output** | Principal Components (PCs) that explain variance. | Functionally-defined axes (HPA and PEA). |
| **Interpretation** | PCs are abstract dimensions of variance. | Axes have clear functional roles: planning vs. execution. |

### Plan for Implementation

I will add this new functionality in a modular way, likely by creating a new class, to keep it distinct from your existing `MultiSessionPCA` analyzer.

Here is my proposed plan:

1.  **Create a New Class**:
    *   I will create a new file `population_analysis/pani_pca.py` containing a class named `PaniEtAlPCA`.
    *   This class will inherit from `MultiSessionPCA` to reuse its data loading, validation, and parallel PSTH extraction capabilities. This follows the DRY principle and leverages your existing, well-documented code.

2.  **Implement the Pani et al. PCA Logic**:
    *   I will add a new method, `fit_pani_pca()`, to the `PaniEtAlPCA` class. This method will perform the core analysis steps described in the paper:
        a.  Group trials by condition and reaction time, then compute average PSTHs for each group.
        b.  Perform an initial PCA on the concatenated data to get the primary PC space (e.g., top 3 PCs).
        c.  Identify the "holding plane" by applying SVD to the correct-stop trial trajectories.
        d.  Calculate the HPA by fitting a line to the initial phase of all trajectories within the holding plane.
        e.  Define the PEA as the axis orthogonal to the holding plane.
        f.  Store the HPA and PEA vectors as attributes of the class.

3.  **Add Projection and Visualization Methods**:
    *   I will add a method `project_on_axes()` to project any given trial trajectory onto the HPA and PEA.
    *   I will create new plotting methods within the `PaniEtAlPCA` class, mirroring the figures in the paper:
        *   `plot_hpa_pea_projections()`: To create plots similar to Fig. 4 in the paper, showing the time course of population activity projected onto the HPA and PEA for different conditions.
        *   `plot_3d_trajectories_with_axes()`: To visualize the neural trajectories in the original PC space, with the HPA and PEA overlaid, similar to Fig. 3.

4.  **Create an Example Notebook**:
    *   To demonstrate the new functionality and ensure it works correctly, I will create a new notebook, `pani_pca_analysis.ipynb`, in the `population_analysis` directory.
    *   This notebook will walk through the process of using the new `PaniEtAlPCA` class, from loading data to generating the final plots.

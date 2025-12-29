"""
Flexible PCA Analysis Module

Allows fitting PCA on specific trial conditions/epochs and projecting
arbitrary conditions/epochs onto the fitted principal components.

Author: Claude & Barak
Date: December 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict, Optional, Tuple, Union
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD, PCA
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path

# Add parent directory to path for cell_analysis import
_module_dir = Path(__file__).parent
_parent_dir = _module_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from cell_analysis import Cell


def _worker_extract_neuron_concat_psths(args):
    """
    Worker function for parallel PSTH extraction during fit.
    Must be at module level for pickling.

    Parameters:
    -----------
    args : tuple
        (cell_id, cell_data_df, trial_specs, bin_size, smooth_ker_size, success_only)
        Note: cell_data_df is pre-filtered for this specific cell

    Returns:
    --------
    tuple : (cell_id, concat_psth, has_all_conditions, trial_counts)
    """
    cell_id, cell_data_df, trial_specs, bin_size, smooth_ker_size, success_only = args

    psths = []
    has_all_conditions = True
    trial_counts = {}

    # Create Cell object (data already filtered for this cell)
    cell = Cell(cell_data_df, verbose=False)

    for spec in trial_specs:
        for direction in spec.get_directions():
            # Get condition label
            cond_label = spec.get_label_for_direction(direction)

            # Extract PSTH
            _, firing_rate, n_trials = cell.calculate_psth(
                epok=spec.epoch,
                bin_size=bin_size,
                alignment_point=spec.alignment,
                trial_type=spec.trial_type,
                direction=direction,
                ssd_number=spec.ssd_number,
                success_only=success_only,
                smooth=True,
                smooth_ker_size=smooth_ker_size,
                delta=False,
                normalize_bins=False
            )

            # Check if we have data
            if firing_rate is None or n_trials == 0:
                has_all_conditions = False
                n_bins = (spec.epoch[1] - spec.epoch[0]) // bin_size
                firing_rate = np.zeros(n_bins)
                n_trials = 0

            psths.append(firing_rate)
            trial_counts[cond_label] = n_trials

    # Concatenate all conditions
    concat_psth = np.concatenate(psths)

    return cell_id, concat_psth, has_all_conditions, trial_counts


def _worker_extract_single_condition_psth(args):
    """
    Worker function for parallel PSTH extraction during project.
    Extracts PSTH for a single cell for a single condition.
    Must be at module level for pickling.

    Parameters:
    -----------
    args : tuple
        (cell_id, cell_data_df, spec, direction, bin_size, smooth_ker_size, success_only)

    Returns:
    --------
    tuple : (cell_id, firing_rate, has_data)
    """
    cell_id, cell_data_df, spec, direction, bin_size, smooth_ker_size, success_only = args

    # Create Cell object (data already filtered for this cell)
    cell = Cell(cell_data_df, verbose=False)

    # Extract PSTH
    _, firing_rate, n_trials = cell.calculate_psth(
        epok=spec.epoch,
        bin_size=bin_size,
        alignment_point=spec.alignment,
        trial_type=spec.trial_type,
        direction=direction,
        ssd_number=spec.ssd_number,
        success_only=success_only,
        smooth=True,
        smooth_ker_size=smooth_ker_size,
        delta=False,
        normalize_bins=False
    )

    # Check if we have data
    if firing_rate is None or n_trials == 0:
        n_bins = (spec.epoch[1] - spec.epoch[0]) // bin_size
        firing_rate = np.zeros(n_bins)
        has_data = False
    else:
        has_data = True

    return cell_id, firing_rate, has_data


class TrialSpec:
    """Specification for a trial condition to extract or project."""

    def __init__(self,
                 trial_type: str,
                 direction: Optional[Union[int, List[int]]] = None,
                 epoch: List[int] = [-50, 300],
                 alignment: str = 'go_cue',
                 ssd_number: Optional[int] = None,
                 label: Optional[str] = None):
        """
        Define a trial condition specification.

        Parameters:
        -----------
        trial_type : str
            'GO', 'STOP', or 'CONT'
        direction : int, list of int, or None
            Direction(s) in degrees (0, 180, or None for both)
            If list, will create separate conditions for each direction
        epoch : list of int
            [start_ms, end_ms] time window
        alignment : str
            Alignment point ('go_cue', 'stop_cue', etc.)
        ssd_number : int or None
            SSD index for STOP/CONT trials (ignored for GO)
        label : str or None
            Custom label for this condition (auto-generated if None)
        """
        self.trial_type = trial_type
        self.direction = direction
        self.epoch = epoch
        self.alignment = alignment
        self.ssd_number = ssd_number if trial_type in ['STOP', 'CONT'] else None

        # Generate label if not provided
        if label is None:
            dir_str = self._format_direction()
            self.label = f"{trial_type}_{dir_str}" if dir_str else trial_type
        else:
            self.label = label

    def _format_direction(self) -> str:
        """Format direction for label."""
        if self.direction is None:
            return "Both"
        elif isinstance(self.direction, list):
            return "_".join([self._dir_to_str(d) for d in self.direction])
        else:
            return self._dir_to_str(self.direction)

    @staticmethod
    def _dir_to_str(direction: int) -> str:
        """Convert direction to string."""
        return 'L' if direction == 180 else 'R'

    def get_directions(self) -> List[int]:
        """Get list of directions to process."""
        if self.direction is None:
            return [0, 180]
        elif isinstance(self.direction, list):
            return self.direction
        else:
            return [self.direction]

    def get_label_for_direction(self, direction: int) -> str:
        """
        Get the label for a specific direction.

        If this TrialSpec has a single direction and a custom label, use it.
        Otherwise, construct label from trial_type and direction.
        """
        # If we have a single direction and it matches the requested direction,
        # use the spec's label (which may be custom)
        if isinstance(self.direction, int) and self.direction == direction:
            return self.label
        else:
            # Multiple directions or mismatch - construct label
            dir_str = self._dir_to_str(direction)
            return f"{self.trial_type}_{dir_str}"

    def __repr__(self):
        return (f"TrialSpec({self.trial_type}, dir={self.direction}, "
                f"epoch={self.epoch}, align={self.alignment}, label={self.label})")


class FlexiblePCA:
    """
    Flexible PCA for neural population analysis.

    Allows fitting PCA on specific conditions and projecting different conditions
    onto the fitted principal components.
    """

    def __init__(self,
                 cell_df: pd.DataFrame,
                 bin_size: int = 1,
                 smooth_ker_size: int = 25,
                 success_only: bool = True,
                 n_components: int = 5,
                 pca_function = TruncatedSVD,
                 random_state: int = 42,
                 verbose: bool = True,
                 n_jobs: int = -1,
                 z_score: bool = False):
        """
        Initialize FlexiblePCA.

        Parameters:
        -----------
        cell_df : DataFrame
            Cell trial database
        bin_size : int
            Bin size for PSTHs (ms)
        smooth_ker_size : int
            Smoothing kernel size (ms)
        success_only : bool
            Use only successful trials
        n_components : int
            Number of principal components
        pca_function : class
            PCA class (PCA or TruncatedSVD)
        random_state : int
            Random seed
        verbose : bool
            Print progress messages
        n_jobs : int
            Number of parallel workers for PSTH extraction
            1 = sequential processing (default)
            -1 = use all available CPU cores
            n > 1 = use n parallel workers
        z_score : bool
            If True, normalize by dividing by std (z-scoring)
            If False, only subtract mean (centering)
        """
        self.cell_df = cell_df
        self.bin_size = bin_size
        self.smooth_ker_size = smooth_ker_size
        self.success_only = success_only
        self.n_components = n_components
        self.pca_function = pca_function
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.z_score = z_score

        # To be set during fit
        self.pca_model = None
        self.fit_specs = None
        self.X_fit_raw = None
        self.X_fit_normalized = None
        self.cell_ids = None
        self.normalization_stats = None
        self.condition_indices = None  # Track which bins belong to which condition

    def _print(self, msg: str):
        """Print if verbose."""
        if self.verbose:
            print(msg)

    def _extract_psth_for_spec(self,
                               cell_id: int,
                               spec: TrialSpec,
                               direction: int) -> Tuple[np.ndarray, int]:
        """
        Extract PSTH for a single condition.

        Returns:
        --------
        psth : ndarray
            Firing rate array
        n_trials : int
            Number of trials used
        """
        cell_data = self.cell_df[self.cell_df['cell_ID'] == cell_id]
        cell = Cell(cell_data, verbose=False)

        _, firing_rate, n_trials = cell.calculate_psth(
            epok=spec.epoch,
            bin_size=self.bin_size,
            alignment_point=spec.alignment,
            trial_type=spec.trial_type,
            direction=direction,
            ssd_number=spec.ssd_number,
            success_only=self.success_only,
            smooth=True,
            smooth_ker_size=self.smooth_ker_size,
            delta=False,
            normalize_bins=False
        )

        return firing_rate, n_trials

    def _extract_neuron_concat_psths(self,
                                     cell_id: int,
                                     specs: List[TrialSpec]) -> Tuple[np.ndarray, bool, Dict]:
        """
        Extract and concatenate PSTHs for all specified conditions for one neuron.

        Returns:
        --------
        concat_psth : ndarray
            Concatenated PSTH across all conditions
        has_all_conditions : bool
            True if neuron has data for all conditions
        trial_counts : dict
            Trial counts for each condition
        """
        psths = []
        has_all_conditions = True
        trial_counts = {}

        for spec in specs:
            for direction in spec.get_directions():
                # Get condition label (respects custom labels)
                cond_label = spec.get_label_for_direction(direction)

                # Extract PSTH
                firing_rate, n_trials = self._extract_psth_for_spec(cell_id, spec, direction)

                # Check if we have data
                if firing_rate is None or n_trials == 0:
                    has_all_conditions = False
                    # Fill with zeros
                    n_bins = (spec.epoch[1] - spec.epoch[0]) // self.bin_size
                    firing_rate = np.zeros(n_bins)
                    n_trials = 0

                psths.append(firing_rate)
                trial_counts[cond_label] = n_trials

        # Concatenate all conditions
        concat_psth = np.concatenate(psths)

        return concat_psth, has_all_conditions, trial_counts

    def fit(self, trial_specs: List[TrialSpec]) -> 'FlexiblePCA':
        """
        Fit PCA on specified trial conditions.

        Parameters:
        -----------
        trial_specs : list of TrialSpec
            Conditions to fit PCA on

        Returns:
        --------
        self : FlexiblePCA
            Returns self for method chaining
        """
        self._print("\n" + "="*80)
        self._print("FITTING PCA")
        self._print("="*80)

        self.fit_specs = trial_specs

        # Print fit conditions
        self._print("\nConditions for fitting:")
        for spec in trial_specs:
            directions = spec.get_directions()
            for direction in directions:
                cond_label = spec.get_label_for_direction(direction)
                self._print(f"  {cond_label}: "
                          f"epoch {spec.epoch} ms, aligned to {spec.alignment}")

        # Get all unique cell IDs
        cell_ids = self.cell_df['cell_ID'].unique()

        # Extract PSTHs for all neurons
        self._print(f"\nExtracting PSTHs for {len(cell_ids)} neurons...")

        neurons_data = []
        neurons_ids = []
        neurons_complete = []
        all_trial_counts = []

        # Choose parallel or sequential processing
        if self.n_jobs == 1:
            # Sequential processing (original implementation)
            for cell_id in tqdm(cell_ids, desc="Processing cells", disable=not self.verbose):
                concat_psth, has_all, trial_counts = self._extract_neuron_concat_psths(
                    cell_id, trial_specs
                )

                neurons_data.append(concat_psth)
                neurons_ids.append(cell_id)
                neurons_complete.append(has_all)
                all_trial_counts.append(trial_counts)
        else:
            # Parallel processing using ProcessPoolExecutor
            import os
            n_workers = os.cpu_count() if self.n_jobs == -1 else self.n_jobs
            self._print(f"  Using {n_workers} parallel workers")

            # Pre-filter data for each cell to reduce pickling overhead
            self._print("  Pre-filtering cell data...")
            worker_args = []
            for cell_id in cell_ids:
                cell_data = self.cell_df[self.cell_df['cell_ID'] == cell_id]
                worker_args.append((
                    cell_id, cell_data, trial_specs, self.bin_size,
                    self.smooth_ker_size, self.success_only
                ))

            # Process in parallel with chunking to reduce task scheduling overhead
            # Chunksize helps balance overhead vs parallelism
            chunksize = max(1, len(cell_ids) // (n_workers * 4))
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                results = list(tqdm(
                    executor.map(_worker_extract_neuron_concat_psths, worker_args, chunksize=chunksize),
                    total=len(cell_ids),
                    desc="Processing cells",
                    disable=not self.verbose
                ))

            # Unpack results
            for cell_id, concat_psth, has_all, trial_counts in results:
                neurons_data.append(concat_psth)
                neurons_ids.append(cell_id)
                neurons_complete.append(has_all)
                all_trial_counts.append(trial_counts)

        # Convert to arrays
        X_raw_all = np.array(neurons_data)
        all_cell_ids = np.array(neurons_ids)
        complete_mask = np.array(neurons_complete)

        # Filter to only neurons with all conditions
        self.X_fit_raw = X_raw_all[complete_mask]
        self.cell_ids = all_cell_ids[complete_mask]
        n_cells = len(self.cell_ids)
        n_features = self.X_fit_raw.shape[1]

        self._print(f"\n✓ PSTH extraction complete")
        self._print(f"  Total neurons processed: {len(all_cell_ids)}")
        self._print(f"  Neurons with all conditions: {n_cells}")
        self._print(f"  Neurons excluded (missing conditions): {(~complete_mask).sum()}")
        self._print(f"  Data matrix shape: {self.X_fit_raw.shape}")

        # Print trial counts
        if n_cells > 0:
            self._print(f"\nTrial counts per condition (neurons with all conditions):")
            # Aggregate trial counts
            for spec in trial_specs:
                for direction in spec.get_directions():
                    cond_label = spec.get_label_for_direction(direction)
                    counts = [tc[cond_label] for tc, complete in zip(all_trial_counts, neurons_complete) if complete]
                    if len(counts) > 0:
                        self._print(f"  {cond_label:15s}: mean={np.mean(counts):.1f}, "
                                  f"min={np.min(counts)}, max={np.max(counts)}")

        # Normalize (z-score or center per neuron)
        norm_type = "z-scoring" if self.z_score else "centering (mean subtraction)"
        self._print(f"\nNormalizing data ({norm_type} per neuron)...")

        # Vectorized normalization - compute stats for all neurons at once
        means = np.mean(self.X_fit_raw, axis=1, keepdims=True)  # Shape: (n_cells, 1)
        stds = np.std(self.X_fit_raw, axis=1, keepdims=True)    # Shape: (n_cells, 1)

        # Store stats (squeeze to 1D for compatibility)
        self.normalization_stats = {
            'mean': means.squeeze(),
            'std': stds.squeeze()
        }

        # Normalize: subtract mean, optionally divide by std
        self.X_fit_normalized = self.X_fit_raw - means

        if self.z_score:
            # Z-score: handle zero std case by setting to zero
            # Avoid division by zero using np.where
            self.X_fit_normalized = np.where(
                stds > 0,
                self.X_fit_normalized / stds,
                0
            )

        self._print(f"✓ Normalization complete")
        self._print(f"  Mean firing rate: {np.mean(means):.2f} ± "
                   f"{np.std(means):.2f} spikes/sec")
        self._print(f"  Normalized data: mean={np.mean(self.X_fit_normalized):.6f}, "
                   f"std={np.std(self.X_fit_normalized):.3f}")

        # Store condition indices (which bins belong to which condition)
        self.condition_indices = {}
        current_idx = 0
        for spec in trial_specs:
            n_bins = (spec.epoch[1] - spec.epoch[0]) // self.bin_size
            for direction in spec.get_directions():
                cond_label = spec.get_label_for_direction(direction)
                self.condition_indices[cond_label] = (current_idx, current_idx + n_bins)
                current_idx += n_bins

        # Fit PCA
        self._print(f"\nFitting {self.pca_function.__name__} with {self.n_components} components...")
        self.pca_model = self.pca_function(n_components=self.n_components,
                                          random_state=self.random_state)

        # Fit on transposed data (features × neurons)
        self.pca_model.fit(self.X_fit_normalized.T)

        self._print(f"\n✓ PCA fitted")
        self._print(f"  Explained variance ratios:")
        for i, var in enumerate(self.pca_model.explained_variance_ratio_):
            self._print(f"    PC{i+1}: {var*100:.2f}%")
        self._print(f"  Total variance explained: "
                   f"{self.pca_model.explained_variance_ratio_.sum()*100:.2f}%")

        self._print("\n" + "="*80)

        return self

    def project(self, trial_specs: List[TrialSpec]) -> Dict[str, np.ndarray]:
        """
        Project specified trial conditions onto fitted PCs.

        Parameters:
        -----------
        trial_specs : list of TrialSpec
            Conditions to project

        Returns:
        --------
        projections : dict
            Dictionary mapping condition labels to PC projections
            Each projection has shape (n_components, n_time_bins)
        """
        if self.pca_model is None:
            raise ValueError("Must call fit() before project()")

        self._print("\n" + "="*80)
        self._print("PROJECTING DATA")
        self._print("="*80)

        # Print projection conditions
        self._print("\nConditions for projection:")
        for spec in trial_specs:
            directions = spec.get_directions()
            for direction in directions:
                cond_label = spec.get_label_for_direction(direction)
                self._print(f"  {cond_label}: "
                          f"epoch {spec.epoch} ms, aligned to {spec.alignment}")

        # Extract PSTHs for projection
        self._print(f"\nExtracting PSTHs for {len(self.cell_ids)} neurons...")

        projections = {}

        # Process each condition separately to get separate trajectories
        for spec in trial_specs:
            for direction in spec.get_directions():
                # Get condition label (respects custom labels)
                cond_label = spec.get_label_for_direction(direction)

                # Extract PSTHs for this specific condition
                if self.n_jobs == 1:
                    # Sequential processing
                    condition_psths = []
                    has_data_mask = []

                    for cell_id in tqdm(self.cell_ids,
                                       desc=f"Processing {cond_label}",
                                       disable=not self.verbose):
                        firing_rate, n_trials = self._extract_psth_for_spec(cell_id, spec, direction)

                        # Check if we have data
                        if firing_rate is None or n_trials == 0:
                            n_bins = (spec.epoch[1] - spec.epoch[0]) // self.bin_size
                            firing_rate = np.zeros(n_bins)
                            has_data_mask.append(False)
                        else:
                            has_data_mask.append(True)

                        condition_psths.append(firing_rate)
                else:
                    # Parallel processing
                    import os
                    n_workers = os.cpu_count() if self.n_jobs == -1 else self.n_jobs

                    # Pre-filter data for each cell
                    worker_args = []
                    for cell_id in self.cell_ids:
                        cell_data = self.cell_df[self.cell_df['cell_ID'] == cell_id]
                        worker_args.append((
                            cell_id, cell_data, spec, direction, self.bin_size,
                            self.smooth_ker_size, self.success_only
                        ))

                    # Process in parallel
                    chunksize = max(1, len(self.cell_ids) // (n_workers * 4))
                    with ProcessPoolExecutor(max_workers=n_workers) as executor:
                        results = list(tqdm(
                            executor.map(_worker_extract_single_condition_psth, worker_args, chunksize=chunksize),
                            total=len(self.cell_ids),
                            desc=f"Processing {cond_label}",
                            disable=not self.verbose
                        ))

                    # Unpack results (maintain order by cell_ids)
                    condition_psths = []
                    has_data_mask = []
                    for cell_id, firing_rate, has_data in results:
                        condition_psths.append(firing_rate)
                        has_data_mask.append(has_data)

                # Convert to array (neurons × time_bins)
                X_condition = np.array(condition_psths)
                has_data_mask = np.array(has_data_mask)

                self._print(f"  {cond_label}: {has_data_mask.sum()}/{len(has_data_mask)} "
                          f"neurons have data")

                # Normalize using same stats as fit data (vectorized)
                means = self.normalization_stats['mean'][:, np.newaxis]  # Shape: (n_cells, 1)
                stds = self.normalization_stats['std'][:, np.newaxis]    # Shape: (n_cells, 1)

                # Subtract mean
                X_condition_normalized = X_condition - means

                # Optionally divide by std (z-score)
                if self.z_score:
                    # Handle zero std case
                    X_condition_normalized = np.where(
                        stds > 0,
                        X_condition_normalized / stds,
                        0
                    )

                # Project onto PCs
                # PCA components: (n_components, n_features_fit)
                # We want to project (n_neurons, n_time_bins) onto the PC space
                # Projection: PC_components @ neurons
                projection = self.pca_model.components_ @ X_condition_normalized
                # Shape: (n_components, n_time_bins)

                projections[cond_label] = projection

                self._print(f"  Projection shape: {projection.shape}")

        self._print("\n✓ Projection complete")
        self._print("="*80)

        return projections

    def get_fit_trajectories(self) -> Dict[str, np.ndarray]:
        """
        Get trajectories for the conditions used to fit PCA.

        Returns:
        --------
        trajectories : dict
            Dictionary mapping condition labels to PC trajectories
            Each trajectory has shape (n_components, n_time_bins)
        """
        if self.pca_model is None:
            raise ValueError("Must call fit() before get_fit_trajectories()")

        # Transform the fit data
        X_pca = self.pca_model.transform(self.X_fit_normalized.T).T
        # Shape: (n_components, n_features)

        trajectories = {}

        # Split into condition-specific trajectories
        for cond_label, (start_idx, end_idx) in self.condition_indices.items():
            trajectories[cond_label] = X_pca[:, start_idx:end_idx]

        return trajectories

    def get_time_axis(self, spec: TrialSpec) -> np.ndarray:
        """
        Get time axis for a trial specification.

        Parameters:
        -----------
        spec : TrialSpec
            Trial specification

        Returns:
        --------
        time_axis : ndarray
            Time values in ms
        """
        n_bins = (spec.epoch[1] - spec.epoch[0]) // self.bin_size
        return np.linspace(spec.epoch[0], spec.epoch[1], n_bins, endpoint=False)

    def compute_mean_ssd(self, ssd_number: int = 2) -> float:
        """
        Compute mean SSD time for STOP trials.

        Parameters:
        -----------
        ssd_number : int
            SSD number to use (default: 2)

        Returns:
        --------
        mean_ssd_ms : float
            Mean SSD time in milliseconds
        """
        stop_trials = self.cell_df[
            (self.cell_df['type'] == 'STOP') &
            (self.cell_df['ssd_number'] == ssd_number) &
            (self.cell_df['trial_failed'] == False)
        ]

        if len(stop_trials) == 0:
            return None

        return stop_trials['ssd_len'].mean()

    def plot_3d_trajectories(self,
                             trajectories_dict: Dict[str, np.ndarray],
                             time_axes_dict: Optional[Dict[str, np.ndarray]] = None,
                             marker_time_ms: Optional[float] = None,
                             marker_label: str = 'Mean SSD',
                             show_ssd_marker: bool = True,
                             ssd_number: int = 2,
                             title: Optional[str] = None,
                             figsize: Tuple[int, int] = (16, 12)):
        """
        Plot 3D trajectories in PC space.

        Parameters:
        -----------
        trajectories_dict : dict
            Dictionary mapping condition labels to trajectories (n_components, n_time_bins)
        time_axes_dict : dict, optional
            Dictionary mapping condition labels to time axes (for markers)
        marker_time_ms : float, optional
            Time in milliseconds to mark with a diamond (overrides show_ssd_marker)
        marker_label : str
            Label for the marker (default: 'Mean SSD')
        show_ssd_marker : bool
            If True, automatically compute and show mean SSD marker (default: True)
        ssd_number : int
            SSD number to use for mean SSD calculation (default: 2)
        title : str, optional
            Plot title
        figsize : tuple
            Figure size (default: (16, 12))

        Returns:
        --------
        fig, ax : matplotlib figure and axis
        """
        if self.pca_model is None:
            raise ValueError("Must call fit() before plotting")

        # Auto-compute mean SSD if requested and marker_time_ms not provided
        if marker_time_ms is None and show_ssd_marker:
            marker_time_ms = self.compute_mean_ssd(ssd_number)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Standard color/linestyle configuration
        plot_config = {
            'GO_R': ('cyan', 'GO Right', '-'),
            'GO_L': ('blue', 'GO Left', '--'),
            'STOP_R': ('orange', 'STOP Right', '-'),
            'STOP_L': ('red', 'STOP Left', '--'),
            'CONT_R': ('lime', 'CONT Right', '-'),
            'CONT_L': ('green', 'CONT Left', '--'),
        }

        # Plot trajectories
        for label, traj in trajectories_dict.items():
            # Get plot configuration or use defaults
            if label in plot_config:
                color, display_label, linestyle = plot_config[label]
            else:
                color, display_label, linestyle = 'gray', label, '-'

            ax.plot(traj[0, :], traj[1, :], traj[2, :],
                   label=display_label, color=color, linewidth=2,
                   alpha=0.7, linestyle=linestyle)

            # Mark start and end points
            ax.scatter(traj[0, 0], traj[1, 0], traj[2, 0],
                      marker='o', s=80, color=color,
                      edgecolors='black', linewidths=1.5, alpha=0.8)
            ax.scatter(traj[0, -1], traj[1, -1], traj[2, -1],
                      marker='s', s=80, color=color,
                      edgecolors='black', linewidths=1.5, alpha=0.8)

            # Add time marker if requested
            if marker_time_ms is not None and time_axes_dict is not None:
                time_axis = time_axes_dict.get(label)
                if time_axis is not None:
                    time_idx = np.argmin(np.abs(time_axis - marker_time_ms))
                    ax.scatter(traj[0, time_idx], traj[1, time_idx], traj[2, time_idx],
                              marker='D', s=100, color=color,
                              edgecolors='black', linewidths=1.5, alpha=0.9, zorder=10)

        # Labels
        var_ratios = self.pca_model.explained_variance_ratio_
        ax.set_xlabel(f'PC1 ({var_ratios[0]*100:.1f}%)', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'PC2 ({var_ratios[1]*100:.1f}%)', fontsize=12, fontweight='bold')
        ax.set_zlabel(f'PC3 ({var_ratios[2]*100:.1f}%)', fontsize=12, fontweight='bold')

        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        elif marker_time_ms is not None:
            ax.set_title(f'Neural Trajectories in PC Space\n'
                        f'Circle=start, Square=end, Diamond={marker_label} ({marker_time_ms:.1f} ms)',
                        fontsize=14, fontweight='bold')
        else:
            ax.set_title('Neural Trajectories in PC Space\nCircle=start, Square=end',
                        fontsize=14, fontweight='bold')

        ax.legend(fontsize=10, loc='upper left')
        plt.tight_layout()

        return fig, ax

    def plot_2d_projections(self,
                           trajectories_dict: Dict[str, np.ndarray],
                           time_axes_dict: Optional[Dict[str, np.ndarray]] = None,
                           marker_time_ms: Optional[float] = None,
                           marker_label: str = 'Mean SSD',
                           show_ssd_marker: bool = True,
                           ssd_number: int = 2,
                           n_pcs: int = 3,
                           title: Optional[str] = None,
                           figsize: Tuple[int, int] = (16, 16)):
        """
        Plot matrix of 2D projections showing all pairwise PC combinations.

        Parameters:
        -----------
        trajectories_dict : dict
            Dictionary mapping condition labels to trajectories (n_components, n_time_bins)
        time_axes_dict : dict, optional
            Dictionary mapping condition labels to time axes (for markers)
        marker_time_ms : float, optional
            Time in milliseconds to mark (overrides show_ssd_marker)
        marker_label : str
            Label for the marker (default: 'Mean SSD')
        show_ssd_marker : bool
            If True, automatically compute and show mean SSD marker (default: True)
        ssd_number : int
            SSD number to use for mean SSD calculation (default: 2)
        n_pcs : int
            Number of PCs to visualize (default: 3)
        title : str, optional
            Overall figure title
        figsize : tuple
            Figure size (default: (16, 16))

        Returns:
        --------
        fig, axes : matplotlib figure and axes array
        """
        if self.pca_model is None:
            raise ValueError("Must call fit() before plotting")

        # Auto-compute mean SSD if requested and marker_time_ms not provided
        if marker_time_ms is None and show_ssd_marker:
            marker_time_ms = self.compute_mean_ssd(ssd_number)

        fig, axes = plt.subplots(n_pcs, n_pcs, figsize=figsize)

        # Standard color/linestyle configuration
        plot_config = {
            'GO_R': ('cyan', 'GO Right', '-'),
            'GO_L': ('blue', 'GO Left', '--'),
            'STOP_R': ('orange', 'STOP Right', '-'),
            'STOP_L': ('red', 'STOP Left', '--'),
            'CONT_R': ('lime', 'CONT Right', '-'),
            'CONT_L': ('green', 'CONT Left', '--'),
        }

        var_ratios = self.pca_model.explained_variance_ratio_

        for i in range(n_pcs):
            for j in range(n_pcs):
                ax = axes[i, j]

                if i == j:
                    # Diagonal: show PC index and variance
                    ax.text(0.5, 0.5, f'PC{i+1}\n{var_ratios[i]*100:.1f}%',
                           ha='center', va='center', fontsize=14, fontweight='bold')
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.axis('off')
                else:
                    # Off-diagonal: plot trajectories
                    for label, traj in trajectories_dict.items():
                        # Get plot config
                        if label in plot_config:
                            color, display_label, linestyle = plot_config[label]
                        else:
                            color, display_label, linestyle = 'gray', label, '-'

                        ax.plot(traj[j, :], traj[i, :],
                               label=display_label, color=color, linewidth=1.2,
                               alpha=0.6, linestyle=linestyle)

                        # Mark start points
                        ax.scatter(traj[j, 0], traj[i, 0],
                                  marker='o', s=30, color=color, edgecolors='black',
                                  linewidths=0.5, zorder=5, alpha=0.7)

                        # Add time marker if requested
                        if marker_time_ms is not None and time_axes_dict is not None:
                            time_axis = time_axes_dict.get(label)
                            if time_axis is not None:
                                time_idx = np.argmin(np.abs(time_axis - marker_time_ms))
                                ax.scatter(traj[j, time_idx], traj[i, time_idx],
                                          marker='D', s=40, color=color,
                                          edgecolors='black', linewidths=0.8,
                                          zorder=10, alpha=0.8)

                    ax.grid(alpha=0.3)

                    # Labels only on edges
                    if i == n_pcs - 1:
                        ax.set_xlabel(f'PC{j+1}', fontsize=10)
                    if j == 0:
                        ax.set_ylabel(f'PC{i+1}', fontsize=10)

                    # Legend only on top-right
                    if i == 0 and j == n_pcs - 1:
                        ax.legend(fontsize=7, loc='upper right', framealpha=0.8)
                        if marker_time_ms is not None:
                            ax.text(0.98, 0.02,
                                   f'◆ = {marker_label} ({marker_time_ms:.1f} ms)',
                                   transform=ax.transAxes, fontsize=7,
                                   ha='right', va='bottom',
                                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)

        plt.tight_layout()
        return fig, axes

    def plot_pc_timeseries(self,
                          trajectories_dict: Dict[str, np.ndarray],
                          time_axes_dict: Dict[str, np.ndarray],
                          marker_time_ms: Optional[float] = None,
                          marker_label: str = 'Mean SSD',
                          show_ssd_marker: bool = True,
                          ssd_number: int = 2,
                          n_pcs: int = 3,
                          title: Optional[str] = None,
                          figsize: Tuple[int, int] = (16, 12)):
        """
        Plot PC time series showing how each PC varies over time.

        Parameters:
        -----------
        trajectories_dict : dict
            Dictionary mapping condition labels to trajectories (n_components, n_time_bins)
        time_axes_dict : dict
            Dictionary mapping condition labels to time axes
        marker_time_ms : float, optional
            Time in milliseconds to mark with vertical line (overrides show_ssd_marker)
        marker_label : str
            Label for the marker (default: 'Mean SSD')
        show_ssd_marker : bool
            If True, automatically compute and show mean SSD marker (default: True)
        ssd_number : int
            SSD number to use for mean SSD calculation (default: 2)
        n_pcs : int
            Number of PCs to visualize (default: 3)
        title : str, optional
            Overall figure title
        figsize : tuple
            Figure size (default: (16, 12))

        Returns:
        --------
        fig, axes : matplotlib figure and axes array
        """
        if self.pca_model is None:
            raise ValueError("Must call fit() before plotting")

        # Auto-compute mean SSD if requested and marker_time_ms not provided
        if marker_time_ms is None and show_ssd_marker:
            marker_time_ms = self.compute_mean_ssd(ssd_number)

        fig, axes = plt.subplots(n_pcs, 1, figsize=figsize)

        # Ensure axes is always an array
        if n_pcs == 1:
            axes = [axes]

        # Standard color/linestyle configuration
        plot_config = {
            'GO_R': ('cyan', 'GO Right', '-'),
            'GO_L': ('blue', 'GO Left', '--'),
            'STOP_R': ('orange', 'STOP Right', '-'),
            'STOP_L': ('red', 'STOP Left', '--'),
            'CONT_R': ('lime', 'CONT Right', '-'),
            'CONT_L': ('green', 'CONT Left', '--'),
        }

        var_ratios = self.pca_model.explained_variance_ratio_

        for i in range(n_pcs):
            ax = axes[i]

            # Plot trajectories
            for label, traj in trajectories_dict.items():
                time_axis = time_axes_dict[label]

                # Get plot config
                if label in plot_config:
                    color, display_label, linestyle = plot_config[label]
                else:
                    color, display_label, linestyle = 'gray', label, '-'

                ax.plot(time_axis, traj[i, :], label=display_label,
                       color=color, linewidth=2, alpha=0.7, linestyle=linestyle)

            # Formatting
            ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
            ax.axvline(0, color='black', linestyle=':', alpha=0.5, linewidth=1.5)

            # Add time marker if requested
            if marker_time_ms is not None:
                ax.axvline(marker_time_ms, color='red', linestyle='--', alpha=0.7,
                          linewidth=2, label=f'{marker_label} ({marker_time_ms:.1f} ms)')

            ax.set_ylabel(f'PC{i+1}\n({var_ratios[i]*100:.1f}%)',
                         fontsize=11, fontweight='bold')
            ax.legend(loc='upper right', fontsize=8, ncol=2, framealpha=0.9)
            ax.grid(alpha=0.3)

            if i == 0 and title:
                ax.set_title(title, fontsize=14, fontweight='bold')
            if i == n_pcs - 1:
                ax.set_xlabel('Time (ms)', fontsize=11)

        plt.tight_layout()
        return fig, axes

    def plot_explained_variance(self,
                                title: Optional[str] = None,
                                figsize: Tuple[int, int] = (14, 5)):
        """
        Plot explained variance analysis (scree plot and cumulative variance).

        Parameters:
        -----------
        title : str, optional
            Overall figure title
        figsize : tuple
            Figure size (default: (14, 5))

        Returns:
        --------
        fig, axes : matplotlib figure and axes array
        """
        if self.pca_model is None:
            raise ValueError("Must call fit() before plotting")

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Get explained variance ratio
        explained_var = self.pca_model.explained_variance_ratio_
        n_components = len(explained_var)

        # Scree plot
        axes[0].bar(range(1, n_components + 1),
                   explained_var * 100,
                   alpha=0.7, color='steelblue')
        axes[0].set_xlabel('Principal Component')
        axes[0].set_ylabel('Explained Variance (%)')
        axes[0].set_title('Scree Plot')
        axes[0].grid(axis='y', alpha=0.3)

        # Cumulative variance
        cumvar = np.cumsum(explained_var) * 100
        axes[1].plot(range(1, n_components + 1), cumvar,
                    marker='o', linewidth=2, markersize=8, color='darkred')
        axes[1].axhline(90, color='gray', linestyle='--', alpha=0.5, label='90%')
        axes[1].set_xlabel('Principal Component')
        axes[1].set_ylabel('Cumulative Explained Variance (%)')
        axes[1].set_title('Cumulative Variance Explained')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')

        plt.tight_layout()

        # Print summary
        if self.verbose:
            print(f"\nExplained variance by component:")
            for i, var in enumerate(explained_var):
                print(f"  PC{i+1}: {var*100:.2f}%")
            print(f"\nTotal variance explained by {n_components} components: {cumvar[-1]:.2f}%")

        return fig, axes


def create_standard_specs(go_epok: List[int] = [-50, 300],
                         stop_epok: List[int] = [-50, 300],
                         cont_epok: List[int] = [-50, 300],
                         go_align: str = 'go_cue',
                         stop_cont_align: str = 'go_cue',
                         ssd_number: int = 2,
                         include_both_dirs: bool = True) -> List[TrialSpec]:
    """
    Create standard set of trial specifications (GO, STOP, CONT × 2 directions).

    Parameters:
    -----------
    go_epok : list of int
        Epoch for GO trials
    stop_epok : list of int
        Epoch for STOP trials
    cont_epok : list of int
        Epoch for CONT trials
    go_align : str
        Alignment for GO trials
    stop_cont_align : str
        Alignment for STOP/CONT trials
    ssd_number : int
        SSD for STOP/CONT trials
    include_both_dirs : bool
        If True, creates specs for both directions (L and R)
        If False, creates single spec for each trial type (both directions combined)

    Returns:
    --------
    specs : list of TrialSpec
        Standard trial specifications
    """
    specs = []

    directions = [0, 180] if include_both_dirs else [None]

    for direction in directions:
        specs.extend([
            TrialSpec('GO', direction=direction, epoch=go_epok, alignment=go_align),
            TrialSpec('STOP', direction=direction, epoch=stop_epok,
                     alignment=stop_cont_align, ssd_number=ssd_number),
            TrialSpec('CONT', direction=direction, epoch=cont_epok,
                     alignment=stop_cont_align, ssd_number=ssd_number),
        ])

    return specs


if __name__ == '__main__':
    print("FlexiblePCA module loaded successfully!")
    print("\nExample usage:")
    print("""
    from flexible_pca import FlexiblePCA, TrialSpec, create_standard_specs

    # Load data
    cell_df = pd.read_pickle('../../data/unified_cell_trial_data/msn_fiona_cell_trial_data.pkl')

    # Create PCA analyzer with parallel processing and z-scoring
    fpca = FlexiblePCA(
        cell_df,
        n_components=5,
        n_jobs=-1,      # Use all CPU cores for parallel processing
        z_score=True    # Z-score normalization (default)
    )

    # Fit on GO trials only
    fit_specs = [
        TrialSpec('GO', direction=0, epoch=[-50, 300], alignment='go_cue'),
        TrialSpec('GO', direction=180, epoch=[-50, 300], alignment='go_cue'),
    ]
    fpca.fit(fit_specs)

    # Project STOP and CONT trials
    proj_specs = [
        TrialSpec('STOP', direction=0, epoch=[-50, 300], alignment='go_cue', ssd_number=2),
        TrialSpec('CONT', direction=0, epoch=[-50, 300], alignment='go_cue', ssd_number=2),
    ]
    projections = fpca.project(proj_specs)
    """)

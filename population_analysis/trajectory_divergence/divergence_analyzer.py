"""
Trajectory Divergence Analysis Module

Implements the Pani et al. (2022) methodology for analyzing neural trajectory
divergence between GO and STOP trials in the countermanding stop-signal task.

Classes:
    TrajectoryDivergenceAnalyzer: Main class for divergence analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from joblib import Parallel, delayed
from tqdm.auto import tqdm


def _analyze_single_neuron_worker(cell_data: pd.DataFrame, cell_id: Any, config: Dict,
                                   random_state_offset: int = 0) -> Dict[str, Any]:
    """
    Standalone worker function for parallel processing.

    Takes pre-filtered cell data to avoid passing entire DataFrame to each worker.
    """
    try:
        random_state = config['random_state'] + random_state_offset
        direction = config['direction']
        ssd_number = config['ssd_number']
        epoch = config['epoch']
        n_samples = config['n_samples']
        window_size_ms = config['window_size_ms']
        dt = config['dt']
        tau_g = config['tau_g']
        tau_d = config['tau_d']
        kernel_duration = config['kernel_duration']

        # Filter trials for GO and STOP
        go_mask = (cell_data['type'] == 'GO') & (cell_data['dir'] == direction) & (cell_data['trial_failed'] == False)
        stop_mask = (cell_data['type'] == 'STOP') & (cell_data['dir'] == direction) & \
                    (cell_data['ssd_number'] == ssd_number) & (cell_data['trial_failed'] == False)

        go_trials = cell_data[go_mask]
        stop_trials = cell_data[stop_mask]

        if len(go_trials) == 0 or len(stop_trials) == 0:
            raise ValueError(f"No GO or STOP trials for cell {cell_id}")

        # Sample GO-GO pairs
        rng = np.random.RandomState(random_state)
        gg_distances = []
        for _ in range(n_samples):
            idx1, idx2 = rng.randint(len(go_trials), size=2)
            t1, t2 = go_trials.iloc[idx1], go_trials.iloc[idx2]
            dist, t_centers = _compute_trial_distance(t1, t2, epoch, window_size_ms, dt, tau_g, tau_d, kernel_duration)
            gg_distances.append(dist)

        # Sample GO-STOP pairs
        gs_distances = []
        for _ in range(n_samples):
            idx1 = rng.randint(len(go_trials))
            idx2 = rng.randint(len(stop_trials))
            t1, t2 = go_trials.iloc[idx1], stop_trials.iloc[idx2]
            dist, _ = _compute_trial_distance(t1, t2, epoch, window_size_ms, dt, tau_g, tau_d, kernel_duration)
            gs_distances.append(dist)

        gg_dist = np.array(gg_distances)
        gs_dist = np.array(gs_distances)

        # Compute statistics
        gg_mean = np.mean(gg_dist, axis=0)
        gg_std = np.std(gg_dist, axis=0)
        gs_mean = np.mean(gs_dist, axis=0)
        gs_std = np.std(gs_dist, axis=0)
        div_mean = np.abs(gs_mean - gg_mean)

        # Find peak divergence
        peak_idx = np.argmax(div_mean)
        peak_time = t_centers[peak_idx]
        peak_val = div_mean[peak_idx]

        return {
            'cell_id': cell_id,
            'go_go_mean': gg_mean,
            'go_go_std': gg_std,
            'go_stop_mean': gs_mean,
            'go_stop_std': gs_std,
            'divergence_mean': div_mean,
            'peak_divergence_time': peak_time,
            'peak_divergence_value': peak_val,
            't_centers': t_centers,
            'success': True,
            'error': None
        }
    except Exception as e:
        return {
            'cell_id': cell_id,
            'success': False,
            'error': str(e)
        }


def _compute_trial_distance(trial1, trial2, epoch, window_size_ms, dt, tau_g, tau_d, kernel_duration):
    """Compute rolling window distance between two trials."""
    # Extract spike times relative to go_cue
    spikes1 = trial1['neural_data'] - trial1['go_cue']
    spikes2 = trial2['neural_data'] - trial2['go_cue']

    # Filter to epoch
    spikes1 = spikes1[(spikes1 >= epoch[0]) & (spikes1 <= epoch[1])]
    spikes2 = spikes2[(spikes2 >= epoch[0]) & (spikes2 <= epoch[1])]

    # Compute SDFs
    t1, sdf1 = _spikes_to_sdf(spikes1, epoch, dt, tau_g, tau_d, kernel_duration)
    t2, sdf2 = _spikes_to_sdf(spikes2, epoch, dt, tau_g, tau_d, kernel_duration)

    # Compute rolling window distance
    window_size = int(window_size_ms / dt)
    n_windows = len(sdf1) - window_size + 1
    distances = np.zeros(n_windows)
    for i in range(n_windows):
        distances[i] = np.sqrt(np.mean((sdf1[i:i + window_size] - sdf2[i:i + window_size])**2))

    window_centers = np.arange(n_windows) + window_size // 2
    window_centers_time = t1[window_centers]

    return distances, window_centers_time


def _spikes_to_sdf(spike_times, epoch, dt, tau_g, tau_d, kernel_duration):
    """Convert spike times to SDF using Pani kernel."""
    t_bins = np.arange(epoch[0], epoch[1] + dt, dt)
    t_centers = t_bins[:-1] + dt / 2
    spike_train, _ = np.histogram(spike_times, bins=t_bins)

    # Create kernel
    t_kernel = np.arange(0, kernel_duration, dt)
    kernel = (1 - np.exp(-t_kernel / tau_g)) * np.exp(-t_kernel / tau_d)
    kernel = kernel / np.sum(kernel)

    # Convolve (causal)
    sdf_full = np.convolve(spike_train, kernel, mode='full')
    sdf = sdf_full[:len(spike_train)] * (1000.0 / dt)

    return t_centers, sdf


# Default configuration
DEFAULT_CONFIG = {
    # SDF kernel parameters (Pani et al. 2022)
    'tau_g': 1.0,           # Growth time constant (ms)
    'tau_d': 20.0,          # Decay time constant (ms)
    'kernel_duration': 20.0, # Kernel duration (ms)
    'dt': 1.0,              # Time step (ms)

    # Analysis parameters
    'epoch': [-50, 250],    # Analysis epoch relative to go_cue (ms)
    'window_size_ms': 10,   # Rolling window size for distance (ms)
    'n_samples': 100,       # Number of trial pair samples
    'random_state': 42,     # Random seed for reproducibility

    # Trial selection
    'direction': 0,         # Direction (0=right, 180=left)
    'ssd_number': 2.0,      # SSD level for STOP/CONT trials
    'min_trials': 5,        # Minimum trials per condition

    # Parallel processing
    'n_jobs': -1,           # Number of parallel jobs (-1 = all cores)
    'verbose': True,        # Print progress information
}


class TrajectoryDivergenceAnalyzer:
    """
    Analyzer for neural trajectory divergence between trial types.

    Implements spike density function (SDF) computation using the Pani et al. (2022)
    causal kernel and computes rolling window distances between trial conditions.

    Parameters
    ----------
    session_data : pd.DataFrame
        Session data containing trial information with columns:
        'cell_ID', 'trial_number', 'type', 'dir', 'ssd_number',
        'trial_failed', 'neural_data', 'go_cue'
    config : dict, optional
        Configuration dictionary. Missing keys use DEFAULT_CONFIG values.

    Attributes
    ----------
    data : pd.DataFrame
        Session data
    config : dict
        Merged configuration
    results : dict
        Analysis results after running population analysis

    Examples
    --------
    >>> analyzer = TrajectoryDivergenceAnalyzer(session_data)
    >>> analyzer.analyze_population()
    >>> analyzer.results['population_divergence_mean']
    """

    def __init__(self, session_data: pd.DataFrame, config: Optional[Dict] = None):
        """Initialize analyzer with session data and configuration."""
        self.data = session_data.copy()
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.results = {}
        self._valid_cell_ids = None

        if self.config['verbose']:
            print(f"TrajectoryDivergenceAnalyzer initialized")
            print(f"  Cells: {self.data['cell_ID'].nunique()}")
            print(f"  Trials: {self.data['trial_number'].nunique()}")
            print(f"  Direction: {self.config['direction']}")
            print(f"  SSD: {self.config['ssd_number']}")

    # =========================================================================
    # SDF Functions
    # =========================================================================

    @staticmethod
    def pani_sdf_kernel(tau_g: float = 1.0, tau_d: float = 20.0,
                        duration: float = 20.0, dt: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create the Pani et al. 2022 spike density function kernel.

        K(t) = [1 - exp(-t/τg)] × exp(-t/τd)

        This is a causal kernel: spikes only influence future time points.

        Parameters
        ----------
        tau_g : float
            Growth time constant (ms), default 1.0 ms
        tau_d : float
            Decay time constant (ms), default 20.0 ms
        duration : float
            Duration of kernel (ms), default 20 ms
        dt : float
            Time step (ms), default 1.0 ms

        Returns
        -------
        t : ndarray
            Time array
        kernel : ndarray
            Kernel values (normalized to sum to 1)
        """
        t = np.arange(0, duration, dt)
        kernel = (1 - np.exp(-t / tau_g)) * np.exp(-t / tau_d)
        kernel = kernel / np.sum(kernel)  # Normalize
        return t, kernel

    def spikes_to_sdf(self, spike_times: np.ndarray,
                      epoch: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert spike times to spike density function using Pani et al. 2022 kernel.

        CAUSAL: A spike at time t only affects time t and later.

        Parameters
        ----------
        spike_times : ndarray
            Spike times (ms) relative to alignment point
        epoch : tuple, optional
            (start, end) of epoch in ms. Uses config if not provided.

        Returns
        -------
        t_centers : ndarray
            Time array (bin centers)
        sdf : ndarray
            Spike density function (spikes/sec)
        """
        if epoch is None:
            epoch = self.config['epoch']

        dt = self.config['dt']
        tau_g = self.config['tau_g']
        tau_d = self.config['tau_d']
        kernel_duration = self.config['kernel_duration']

        # Create time bins
        t_bins = np.arange(epoch[0], epoch[1] + dt, dt)
        t_centers = t_bins[:-1] + dt / 2

        # Create spike train
        spike_train, _ = np.histogram(spike_times, bins=t_bins)

        # Get kernel
        _, kernel = self.pani_sdf_kernel(tau_g, tau_d, kernel_duration, dt)

        # Convolve (causal: only take first N elements)
        sdf_full = np.convolve(spike_train, kernel, mode='full')
        sdf = sdf_full[:len(spike_train)]

        # Convert to spikes/sec
        sdf = sdf * (1000.0 / dt)

        return t_centers, sdf

    # =========================================================================
    # Distance Functions
    # =========================================================================

    @staticmethod
    def rolling_window_distance(sdf1: np.ndarray, sdf2: np.ndarray,
                                 window_size_ms: float = 10,
                                 dt: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute RMSE distance between two SDFs in a rolling window.

        Parameters
        ----------
        sdf1, sdf2 : ndarray
            Spike density functions (same length)
        window_size_ms : float
            Window size in ms
        dt : float
            Time step in ms

        Returns
        -------
        distances : ndarray
            RMSE distance in each window
        window_centers : ndarray
            Index of window centers
        """
        window_size = int(window_size_ms / dt)
        n_windows = len(sdf1) - window_size + 1
        distances = np.zeros(n_windows)

        for i in range(n_windows):
            sdf1_window = sdf1[i:i + window_size]
            sdf2_window = sdf2[i:i + window_size]
            distances[i] = np.sqrt(np.mean((sdf1_window - sdf2_window)**2))

        window_centers = np.arange(n_windows) + window_size // 2
        return distances, window_centers

    def compute_trial_pair_distance(self, trial1_data: pd.Series,
                                     trial2_data: pd.Series,
                                     epoch: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute rolling window distance between two trials.

        Parameters
        ----------
        trial1_data, trial2_data : pd.Series
            Trial data rows containing 'neural_data' and 'go_cue'
        epoch : tuple, optional
            Analysis epoch. Uses config if not provided.

        Returns
        -------
        distances : ndarray
            Distance in each window
        window_centers_time : ndarray
            Time of window centers
        """
        if epoch is None:
            epoch = self.config['epoch']

        # Extract spike times relative to go_cue
        spikes1 = trial1_data['neural_data'] - trial1_data['go_cue']
        spikes2 = trial2_data['neural_data'] - trial2_data['go_cue']

        # Filter to epoch
        spikes1 = spikes1[(spikes1 >= epoch[0]) & (spikes1 <= epoch[1])]
        spikes2 = spikes2[(spikes2 >= epoch[0]) & (spikes2 <= epoch[1])]

        # Compute SDFs
        t1, sdf1 = self.spikes_to_sdf(spikes1, epoch)
        t2, sdf2 = self.spikes_to_sdf(spikes2, epoch)

        # Compute rolling window distance
        distances, window_centers_idx = self.rolling_window_distance(
            sdf1, sdf2, self.config['window_size_ms'], self.config['dt']
        )
        window_centers_time = t1[window_centers_idx]

        return distances, window_centers_time

    def sample_trial_pairs_distances(self, cell_data: pd.DataFrame,
                                      trial_type1: str, trial_type2: str,
                                      n_samples: Optional[int] = None,
                                      random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample random pairs of trials and compute distances.

        Parameters
        ----------
        cell_data : pd.DataFrame
            Data for a single cell
        trial_type1, trial_type2 : str
            Trial types ('GO', 'STOP', or 'CONT')
        n_samples : int, optional
            Number of samples. Uses config if not provided.
        random_state : int, optional
            Random seed. Uses config if not provided.

        Returns
        -------
        all_distances : ndarray
            Distance arrays for all samples (n_samples, n_timepoints)
        t_centers : ndarray
            Time of window centers
        """
        if n_samples is None:
            n_samples = self.config['n_samples']
        if random_state is None:
            random_state = self.config['random_state']

        direction = self.config['direction']
        ssd_number = self.config['ssd_number']
        epoch = self.config['epoch']

        # Filter trials for each type
        trials1 = self.filter_trials(cell_data, trial_type1, direction,
                                      ssd_number if trial_type1 != 'GO' else None)
        trials2 = self.filter_trials(cell_data, trial_type2, direction,
                                      ssd_number if trial_type2 != 'GO' else None)

        if len(trials1) == 0 or len(trials2) == 0:
            raise ValueError(f"No trials found for {trial_type1} or {trial_type2}")

        # Sample pairs
        rng = np.random.RandomState(random_state)
        all_distances = []
        t_centers = None

        for _ in range(n_samples):
            # Random sample one trial from each set
            idx1 = rng.randint(len(trials1))
            idx2 = rng.randint(len(trials2))

            trial1 = trials1.iloc[idx1]
            trial2 = trials2.iloc[idx2]

            distances, t_centers = self.compute_trial_pair_distance(trial1, trial2, epoch)
            all_distances.append(distances)

        return np.array(all_distances), t_centers

    # =========================================================================
    # Trial Filtering Utilities
    # =========================================================================

    @staticmethod
    def filter_trials(data: pd.DataFrame, trial_type: str, direction: int,
                      ssd_number: Optional[float] = None,
                      success_only: bool = True,
                      cell_id: Optional[Any] = None) -> pd.DataFrame:
        """
        Filter trials based on specified conditions.

        Parameters
        ----------
        data : pd.DataFrame
            Session data containing trial information
        trial_type : str
            'GO', 'STOP', or 'CONT'
        direction : int
            0 (right) or 180 (left)
        ssd_number : float, optional
            SSD level (1.0-4.0) for STOP/CONT trials
        success_only : bool
            If True, only include successful trials
        cell_id : optional
            Cell ID to filter by

        Returns
        -------
        pd.DataFrame
            Filtered data
        """
        mask = (data['type'] == trial_type) & (data['dir'] == direction)

        if success_only:
            mask &= (data['trial_failed'] == False)

        if ssd_number is not None:
            mask &= (data['ssd_number'] == ssd_number)

        if cell_id is not None:
            mask &= (data['cell_ID'] == cell_id)

        return data[mask]

    @staticmethod
    def get_trial_numbers(data: pd.DataFrame, trial_type: str, direction: int,
                          ssd_number: Optional[float] = None,
                          success_only: bool = True) -> List[int]:
        """
        Extract unique trial numbers for specified conditions.

        Returns
        -------
        list
            Sorted list of unique trial numbers
        """
        filtered = TrajectoryDivergenceAnalyzer.filter_trials(
            data, trial_type, direction, ssd_number, success_only
        )
        return sorted(filtered['trial_number'].unique())

    # =========================================================================
    # Single Neuron Analysis
    # =========================================================================

    def analyze_single_neuron(self, cell_id: Any,
                               random_state_offset: int = 0) -> Dict[str, Any]:
        """
        Analyze divergence for a single neuron.

        Parameters
        ----------
        cell_id : any
            Cell identifier
        random_state_offset : int
            Offset to add to random state for reproducibility across neurons

        Returns
        -------
        dict
            Results containing:
            - cell_id: Cell identifier
            - go_go_mean/std: GO-GO baseline distances
            - go_stop_mean/std: GO-STOP distances
            - divergence_mean: Absolute difference (GO-STOP - GO-GO)
            - peak_divergence_time/value: Peak divergence location
            - t_centers: Time array
            - success: Whether analysis succeeded
            - error: Error message if failed
        """
        try:
            cell_data = self.data[self.data['cell_ID'] == cell_id]
            random_state = self.config['random_state'] + random_state_offset

            # Sample GO-GO pairs (baseline)
            gg_dist, t_centers = self.sample_trial_pairs_distances(
                cell_data, 'GO', 'GO', random_state=random_state
            )

            # Sample GO-STOP pairs (divergence)
            gs_dist, _ = self.sample_trial_pairs_distances(
                cell_data, 'GO', 'STOP', random_state=random_state
            )

            # Compute statistics
            gg_mean = np.mean(gg_dist, axis=0)
            gg_std = np.std(gg_dist, axis=0)
            gs_mean = np.mean(gs_dist, axis=0)
            gs_std = np.std(gs_dist, axis=0)
            div_mean = np.abs(gs_mean - gg_mean)

            # Find peak divergence
            peak_idx = np.argmax(div_mean)
            peak_time = t_centers[peak_idx]
            peak_val = div_mean[peak_idx]

            return {
                'cell_id': cell_id,
                'go_go_mean': gg_mean,
                'go_go_std': gg_std,
                'go_stop_mean': gs_mean,
                'go_stop_std': gs_std,
                'divergence_mean': div_mean,
                'peak_divergence_time': peak_time,
                'peak_divergence_value': peak_val,
                't_centers': t_centers,
                'success': True,
                'error': None
            }
        except Exception as e:
            return {
                'cell_id': cell_id,
                'success': False,
                'error': str(e)
            }

    # =========================================================================
    # Population Analysis
    # =========================================================================

    def get_valid_cells(self, min_go_trials: Optional[int] = None,
                        min_stop_trials: Optional[int] = None) -> List[Any]:
        """
        Get cell IDs with sufficient trials for analysis.

        Parameters
        ----------
        min_go_trials : int, optional
            Minimum GO trials required. Uses config['min_trials'] if not provided.
        min_stop_trials : int, optional
            Minimum STOP trials required. Uses config['min_trials'] if not provided.

        Returns
        -------
        list
            List of valid cell IDs
        """
        if min_go_trials is None:
            min_go_trials = self.config['min_trials']
        if min_stop_trials is None:
            min_stop_trials = self.config['min_trials']

        direction = self.config['direction']
        ssd_number = self.config['ssd_number']

        valid_cells = []
        for cell_id in self.data['cell_ID'].unique():
            cell_data = self.data[self.data['cell_ID'] == cell_id]

            go_trials = self.filter_trials(cell_data, 'GO', direction)
            stop_trials = self.filter_trials(cell_data, 'STOP', direction, ssd_number)

            if len(go_trials) >= min_go_trials and len(stop_trials) >= min_stop_trials:
                valid_cells.append(cell_id)

        return valid_cells

    def analyze_population(self, cell_ids: Optional[List] = None) -> Dict[str, Any]:
        """
        Analyze divergence across population of neurons using parallel processing.

        Parameters
        ----------
        cell_ids : list, optional
            List of cell IDs to analyze. If None, uses all valid cells.

        Returns
        -------
        dict
            Population results stored in self.results
        """
        if cell_ids is None:
            cell_ids = self.get_valid_cells()
            if self.config['verbose']:
                print(f"Found {len(cell_ids)} valid cells for analysis")

        self._valid_cell_ids = cell_ids
        n_jobs = self.config['n_jobs']
        verbose = self.config['verbose']

        # Pre-chunk data by cell_id (much faster than filtering in each worker)
        if verbose:
            print(f"Pre-chunking data for {len(cell_ids)} neurons...")

        cell_data_chunks = {
            cell_id: self.data[self.data['cell_ID'] == cell_id]
            for cell_id in tqdm(cell_ids, disable=not verbose, desc="Chunking")
        }

        # Parallel processing with pre-chunked data
        if verbose:
            print(f"Analyzing {len(cell_ids)} neurons in parallel...")

        results_list = Parallel(n_jobs=n_jobs)(
            delayed(_analyze_single_neuron_worker)(cell_data_chunks[cell_id], cell_id, self.config, i)
            for i, cell_id in enumerate(tqdm(cell_ids, disable=not verbose, desc="Processing"))
        )

        # Aggregate results
        successful = [r for r in results_list if r['success']]
        failed = [r for r in results_list if not r['success']]

        if verbose:
            print(f"Successfully analyzed: {len(successful)}/{len(cell_ids)} neurons")
            if failed:
                print(f"Failed neurons: {[r['cell_id'] for r in failed]}")

        if not successful:
            raise ValueError("No neurons were successfully analyzed")

        # Stack results
        self.results = {
            'cell_ids': [r['cell_id'] for r in successful],
            'go_go_mean': np.array([r['go_go_mean'] for r in successful]),
            'go_go_std': np.array([r['go_go_std'] for r in successful]),
            'go_stop_mean': np.array([r['go_stop_mean'] for r in successful]),
            'go_stop_std': np.array([r['go_stop_std'] for r in successful]),
            'divergence_mean': np.array([r['divergence_mean'] for r in successful]),
            'peak_divergence_time': np.array([r['peak_divergence_time'] for r in successful]),
            'peak_divergence_value': np.array([r['peak_divergence_value'] for r in successful]),
            't_centers': successful[0]['t_centers'],
            'n_successful': len(successful),
            'n_failed': len(failed),
            'failed_cells': [r['cell_id'] for r in failed],
        }

        # Compute population statistics
        self._compute_population_statistics()

        return self.results

    def _compute_population_statistics(self):
        """Compute population-level statistics from individual neuron results."""
        n_neurons = len(self.results['cell_ids'])

        # GO-GO population stats
        self.results['go_go_population_mean'] = np.mean(self.results['go_go_mean'], axis=0)
        self.results['go_go_population_std'] = np.std(self.results['go_go_mean'], axis=0)
        self.results['go_go_population_sem'] = self.results['go_go_population_std'] / np.sqrt(n_neurons)

        # GO-STOP population stats
        self.results['go_stop_population_mean'] = np.mean(self.results['go_stop_mean'], axis=0)
        self.results['go_stop_population_std'] = np.std(self.results['go_stop_mean'], axis=0)
        self.results['go_stop_population_sem'] = self.results['go_stop_population_std'] / np.sqrt(n_neurons)

        # Divergence population stats
        self.results['divergence_population_mean'] = np.mean(self.results['divergence_mean'], axis=0)
        self.results['divergence_population_std'] = np.std(self.results['divergence_mean'], axis=0)
        self.results['divergence_population_sem'] = self.results['divergence_population_std'] / np.sqrt(n_neurons)

        # Peak divergence stats
        t_centers = self.results['t_centers']
        peak_idx = np.argmax(self.results['divergence_population_mean'])
        self.results['population_peak_divergence_time'] = t_centers[peak_idx]
        self.results['population_peak_divergence_value'] = self.results['divergence_population_mean'][peak_idx]

        if self.config['verbose']:
            print(f"\nPopulation Statistics ({n_neurons} neurons):")
            print(f"  Peak divergence: {self.results['population_peak_divergence_value']:.2f} "
                  f"at {self.results['population_peak_divergence_time']:.1f} ms")
            print(f"  Mean peak time (per neuron): {np.mean(self.results['peak_divergence_time']):.1f} ms "
                  f"(std: {np.std(self.results['peak_divergence_time']):.1f} ms)")

    # =========================================================================
    # Statistics
    # =========================================================================

    def run_paired_ttest(self, window: Tuple[float, float] = (-50, 50),
                         align_to_peak: bool = True) -> Dict[str, float]:
        """
        Run paired t-test comparing GO vs STOP firing around peak divergence.

        Parameters
        ----------
        window : tuple
            Time window (start, end) in ms
        align_to_peak : bool
            If True, window is relative to peak divergence time

        Returns
        -------
        dict
            Statistical results including t-statistic, p-value, Cohen's d
        """
        from scipy.stats import ttest_rel

        if not self.results:
            raise ValueError("Run analyze_population() first")

        t_centers = self.results['t_centers']

        if align_to_peak:
            # Shift window to be relative to each neuron's peak
            go_means = []
            stop_means = []

            for i, _ in enumerate(self.results['cell_ids']):
                peak_time = self.results['peak_divergence_time'][i]
                actual_window = (peak_time + window[0], peak_time + window[1])

                mask = (t_centers >= actual_window[0]) & (t_centers <= actual_window[1])
                if np.sum(mask) > 0:
                    go_means.append(np.mean(self.results['go_go_mean'][i][mask]))
                    stop_means.append(np.mean(self.results['go_stop_mean'][i][mask]))
        else:
            mask = (t_centers >= window[0]) & (t_centers <= window[1])
            go_means = [np.mean(self.results['go_go_mean'][i][mask])
                       for i in range(len(self.results['cell_ids']))]
            stop_means = [np.mean(self.results['go_stop_mean'][i][mask])
                         for i in range(len(self.results['cell_ids']))]

        go_means = np.array(go_means)
        stop_means = np.array(stop_means)

        t_stat, p_val = ttest_rel(stop_means, go_means)
        diff = stop_means - go_means
        cohens_d = np.mean(diff) / np.std(diff)

        stats_results = {
            't_statistic': t_stat,
            'p_value': p_val,
            'cohens_d': cohens_d,
            'n_neurons': len(go_means),
            'go_mean': np.mean(go_means),
            'go_std': np.std(go_means),
            'stop_mean': np.mean(stop_means),
            'stop_std': np.std(stop_means),
            'mean_difference': np.mean(diff),
        }

        if self.config['verbose']:
            print(f"\nPaired t-test Results (n={len(go_means)}):")
            print(f"  Window: {window} ms {'(relative to peak)' if align_to_peak else ''}")
            print(f"  GO mean: {stats_results['go_mean']:.2f} ± {stats_results['go_std']:.2f}")
            print(f"  STOP mean: {stats_results['stop_mean']:.2f} ± {stats_results['stop_std']:.2f}")
            print(f"  t = {t_stat:.4f}, p = {p_val:.4e}")
            print(f"  Cohen's d = {cohens_d:.3f}")

        return stats_results

    # =========================================================================
    # Aligned PSTH Database
    # =========================================================================

    def create_aligned_psth_database(self, cell_class,
                                      trial_types: List[str] = ['GO', 'STOP', 'CONT'],
                                      bin_size: int = 1,
                                      smooth: bool = True,
                                      smooth_ker_size: int = 25) -> pd.DataFrame:
        """
        Create database of PSTHs aligned to each neuron's peak divergence time.

        Parameters
        ----------
        cell_class : class
            Cell class from cell_analysis module (passed to avoid circular import)
        trial_types : list
            Trial types to include
        bin_size : int
            PSTH bin size in ms
        smooth : bool
            Whether to smooth PSTHs
        smooth_ker_size : int
            Smoothing kernel size

        Returns
        -------
        pd.DataFrame
            Database with columns: cell_id, direction, trial_type, ssd_number,
            time (aligned to peak), firing_rate, n_trials, peak_divergence_time, etc.
        """
        if not self.results:
            raise ValueError("Run analyze_population() first")

        direction = self.config['direction']
        ssd_number = self.config['ssd_number']
        epoch = self.config['epoch']
        dir_map = {0: 'Right', 180: 'Left'}

        aligned_data = []

        for idx, cell_id in enumerate(tqdm(self.results['cell_ids'],
                                           disable=not self.config['verbose'],
                                           desc="Creating aligned PSTH database")):
            peak_time = self.results['peak_divergence_time'][idx]
            cell_data = self.data[self.data['cell_ID'] == cell_id]
            cell = cell_class(cell_data, verbose=False)

            # Get mean SSD for this cell
            stop_trials = cell_data[
                (cell_data['type'] == 'STOP') &
                (cell_data['dir'] == direction) &
                (cell_data['ssd_number'] == ssd_number)
            ]
            ssd_mean = np.nan
            if len(stop_trials) > 0 and 'stop_cue' in stop_trials.columns:
                ssd_mean = (stop_trials['stop_cue'] - stop_trials['go_cue']).mean()

            for trial_type in trial_types:
                ssd_for_type = ssd_number if trial_type in ['STOP', 'CONT'] else None

                bin_centers, firing_rate, n_trials = cell.calculate_psth(
                    epok=epoch,
                    bin_size=bin_size,
                    alignment_point='go_cue',
                    trial_type=trial_type,
                    direction=direction,
                    ssd_number=ssd_for_type,
                    success_only=True,
                    smooth=smooth,
                    smooth_ker_size=smooth_ker_size,
                    delta=False,
                    normalize_bins=False
                )

                if bin_centers is not None and len(bin_centers) > 0:
                    shifted_time = bin_centers - peak_time

                    aligned_data.append({
                        'cell_id': cell_id,
                        'direction': direction,
                        'direction_label': dir_map[direction],
                        'trial_type': trial_type,
                        'ssd_number': ssd_for_type,
                        'time': shifted_time,
                        'firing_rate': firing_rate,
                        'n_trials': n_trials,
                        'peak_divergence_time_relative_to_go': peak_time,
                        'go_cue_aligned_time': -peak_time,
                        'stop_signal_aligned_time': ssd_mean - peak_time if not np.isnan(ssd_mean) else np.nan,
                    })

        return pd.DataFrame(aligned_data)

    def get_population_psth_stats(self, psth_database: pd.DataFrame,
                                   trial_type: str,
                                   common_time: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute population average PSTH with interpolation to common time axis.

        Parameters
        ----------
        psth_database : pd.DataFrame
            Database from create_aligned_psth_database()
        trial_type : str
            Trial type to compute stats for
        common_time : ndarray, optional
            Common time axis for interpolation. Default: -250 to 250 ms.

        Returns
        -------
        mean_rate : ndarray
            Population mean firing rate
        sem_rate : ndarray
            Standard error of mean
        n_valid : ndarray
            Number of valid neurons at each time point
        """
        from scipy.interpolate import interp1d

        if common_time is None:
            common_time = np.arange(-250, 251, 1)

        subset = psth_database[psth_database['trial_type'] == trial_type]
        interpolated_rates = []

        for _, row in subset.iterrows():
            t = row['time']
            fr = row['firing_rate']

            if len(t) > 1:
                f = interp1d(t, fr, kind='linear', bounds_error=False, fill_value=np.nan)
                fr_interp = f(common_time)
                interpolated_rates.append(fr_interp)

        if not interpolated_rates:
            return None, None, None

        rate_matrix = np.vstack(interpolated_rates)

        with np.errstate(divide='ignore', invalid='ignore'):
            mean_rate = np.nanmean(rate_matrix, axis=0)
            std_rate = np.nanstd(rate_matrix, axis=0)
            n_valid = np.sum(~np.isnan(rate_matrix), axis=0)
            sem_rate = std_rate / np.sqrt(n_valid)

        return mean_rate, sem_rate, n_valid

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_neuron_data(self, cell_idx: int) -> Dict[str, Any]:
        """
        Get results for a specific neuron by index.

        Parameters
        ----------
        cell_idx : int
            Index into results arrays

        Returns
        -------
        dict
            Neuron-specific results
        """
        if not self.results:
            raise ValueError("Run analyze_population() first")

        return {
            'cell_id': self.results['cell_ids'][cell_idx],
            'go_go_mean': self.results['go_go_mean'][cell_idx],
            'go_go_std': self.results['go_go_std'][cell_idx],
            'go_stop_mean': self.results['go_stop_mean'][cell_idx],
            'go_stop_std': self.results['go_stop_std'][cell_idx],
            'divergence_mean': self.results['divergence_mean'][cell_idx],
            'peak_time': self.results['peak_divergence_time'][cell_idx],
            'peak_value': self.results['peak_divergence_value'][cell_idx],
            't_centers': self.results['t_centers'],
        }

    def summary(self) -> str:
        """Return a summary string of the analysis configuration and results."""
        lines = [
            "TrajectoryDivergenceAnalyzer Summary",
            "=" * 40,
            f"Direction: {self.config['direction']} ({'Right' if self.config['direction'] == 0 else 'Left'})",
            f"SSD: {self.config['ssd_number']}",
            f"Epoch: {self.config['epoch']} ms",
            f"Window size: {self.config['window_size_ms']} ms",
            f"N samples: {self.config['n_samples']}",
        ]

        if self.results:
            lines.extend([
                "",
                "Results:",
                f"  Analyzed neurons: {self.results['n_successful']}",
                f"  Failed neurons: {self.results['n_failed']}",
                f"  Population peak divergence: {self.results['population_peak_divergence_value']:.2f} "
                f"at {self.results['population_peak_divergence_time']:.1f} ms",
            ])
        else:
            lines.append("\nNo results yet. Run analyze_population() first.")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"TrajectoryDivergenceAnalyzer(n_cells={self.data['cell_ID'].nunique()}, direction={self.config['direction']})"

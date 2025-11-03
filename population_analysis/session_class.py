import pandas as pd
import numpy as np
import holoviews as hv
from holoviews import opts
from scipy.ndimage import gaussian_filter1d
from cell_analysis import Cell


class Session:
    """
    Class representing a single recording session with multiple cells.
    Handles population-level analysis within a session.
    Data generation methods are separated from plotting methods.
    """
    
    def __init__(self, session_df, verbose=False):
        """
        Initialize Session with all cells from a single recording session.
        
        Parameters:
        -----------
        session_df : pd.DataFrame
            DataFrame containing all cell-trial data for a single session
            Its columns are: 'cell_ID', 'cell_type', 'maestro_ID', 'problem', 
            'grade', 'filename', 'trial_name', 'reaction_time', 'go_cue', 
            'stop_cue', 'trial_failed', 'ssd_len', 'ssd_number', 'type', 
            'first_relevant_saccade', 'segs_durations', 'segs_times', 
            'trial_length', 'screen_rotation', 'saccades', 'blinks', 'dir', 
            'neural_data', 'session', 'plexon_session', 'trial_number', 'trial_session'
        """
        assert session_df['trial_session'].unique().size == 1, "DataFrame must contain only one session"
        self.data = session_df.copy()
        self.session_id = session_df.iloc[0]['trial_session']
        self.cell_ids = sorted(self.data['cell_ID'].unique())
        self.n_cells = len(self.cell_ids)
        self.trial_ids = sorted(self.data['trial_number'].unique())
        self.n_trials = len(self.trial_ids)
        
        if verbose:
            print(f"Session {self.session_id} initialized:")
            print(f"  - Number of cells: {self.n_cells}")
            print(f"  - Total trials: {self.n_trials}")
            print(f"  - Trial types: {sorted(self.data['type'].unique())}")
            print(f"  - Directions: {sorted(self.data['dir'].unique())}")
    
    def get_cells_with_no_spikes(self, as_percentage=False):
        """
        Get the number or percentage of cells that have no spikes across all trials.
        Uses fully vectorized operations for efficiency.
        
        Parameters:
        -----------
        as_percentage : bool
            If True, return percentage instead of count (default: False)
        
        Returns:
        --------
        int or float : Number or percentage of cells with zero spikes in all trials
        """
        # Group by cell_ID and check if all neural_data entries are empty lists
        # For each cell, get unique values in neural_data and check if only [[]] exists
        cells_with_no_spikes = (
            self.data.groupby('cell_ID')['neural_data']
            .apply(lambda x: x.value_counts().index.tolist() == [[]])
            .sum()
        )
        
        if as_percentage:
            return (cells_with_no_spikes / self.n_cells) * 100 if self.n_cells > 0 else 0.0
        
        return int(cells_with_no_spikes)
    
    def get_cell_psth(self, cell_id, epok=[-500, 1000], bin_size=1,
                      alignment_point='go_cue', trial_type=None, direction=None,
                      ssd_number=None, success_only=True, smooth=True, delta=False,
                      smooth_ker_size=25, normalize_bins=False):
        """
        Calculate PSTH for a specific cell.
        
        Parameters:
        -----------
        cell_id : int
            Cell identifier
        epok : list
            Time window [start, end] in ms
        bin_size : int
            Bin size in ms
        alignment_point : str
            Event to align to
        trial_type : str, optional
            'GO', 'STOP', or 'CONT'
        direction : int, optional
            0 (right) or 180 (left)
        ssd_number : int, optional
            SSD number (1-4)
        success_only : bool
            Include only successful trials
        smooth : bool
            Apply Gaussian smoothing
        smooth_ker_size : int
            Kernel size for Gaussian smoothing (default: 25)
        delta: bool
            if True look center to the mean firing rate
        normalize_bins : bool
            If True, z-score spike counts before calculating firing rate (default: False)
            
        Returns:
        --------
        tuple : (bin_centers, firing_rate, n_trials)
        """
        # Get data for this specific cell
        # Note: No need to copy here since Cell.__init__ will make its own defensive copy
        cell_data = self.data[self.data['cell_ID'] == cell_id]
        
        if len(cell_data) == 0:
            return None, None, 0
        
        # Create a Cell instance for this cell (Cell.__init__ will copy the data)
        cell = Cell(cell_data)
        
        # Use Cell's calculate_psth method
        # Note: Cell uses smooth_ker_size parameter for Gaussian smoothing kernel size
        # Session class uses bin_size for both binning and smoothing (sigma=bin_size)
        bin_centers, firing_rate, n_trials = cell.calculate_psth(
            epok=epok,
            bin_size=bin_size,
            alignment_point=alignment_point,
            trial_type=trial_type,
            direction=direction,
            ssd_number=ssd_number,
            success_only=success_only,
            smooth=smooth,
            smooth_ker_size=smooth_ker_size,  # Match Session's original behavior: sigma=bin_size
            delta=delta,
            normalize_bins=normalize_bins
        )
        
        return bin_centers, firing_rate, n_trials
    
    def get_all_cells_psth(self, epok=[-500, 1000], bin_size=1,
                           alignment_point='go_cue', trial_type=None, direction=None,
                           ssd_number=None, success_only=True, smooth=True, delta=False,
                           smooth_ker_size=25, normalize_bins=False, normalize=False):
        """
        Get PSTH for all cells in the session.
        
        Parameters:
        -----------
        Same as get_cell_psth, plus:
        normalize : bool
            If True, normalize all firing rates by the global maximum across all cells
            
        Returns:
        --------
        tuple : (bin_centers, psth_matrix, cell_ids_used)
            - bin_centers: time bins
            - psth_matrix: 2D array (n_cells X n_bins)
            - cell_ids_used: list of cell IDs with data
        """
        psth_list = []
        cells_used = []
        bin_centers = None
        
        # First pass: collect all PSTHs without normalization
        for cell_id in self.cell_ids:
            bins, firing_rate, n_trials = self.get_cell_psth(
                cell_id=cell_id,
                epok=epok,
                bin_size=bin_size,
                alignment_point=alignment_point,
                trial_type=trial_type,
                direction=direction,
                ssd_number=ssd_number,
                success_only=success_only,
                smooth=smooth,
                smooth_ker_size=smooth_ker_size, 
                delta=delta,
                normalize_bins=normalize_bins
            )
            
            if bins is not None and n_trials > 0:
                psth_list.append(firing_rate)
                cells_used.append(cell_id)
                
                if bin_centers is None:
                    bin_centers = bins
        
        if len(psth_list) == 0:
            return None, None, []
        
        psth_matrix = np.array(psth_list)
        
        # Apply global normalization if requested
        if normalize:
            global_max = psth_matrix.max()
            if global_max > 0:
                psth_matrix = psth_matrix / global_max
        
        return bin_centers, psth_matrix, cells_used
    
    def get_cell_spike_counts(self, cell_id, epok=[-500, 1000], bin_size=1,
                              alignment_point='go_cue', trial_type=None, direction=None,
                              ssd_number=None, success_only=True, normalize=False):
        """
        Get aggregated spike counts for a specific cell.
        
        Parameters:
        -----------
        cell_id : int
            Cell identifier
        epok : list
            Time window [start, end] in ms
        bin_size : int
            Bin size in ms
        alignment_point : str
            Event to align to
        trial_type : str, optional
            'GO', 'STOP', or 'CONT'
        direction : int, optional
            0 (right) or 180 (left)
        ssd_number : int, optional
            SSD number (1-4)
        success_only : bool
            Include only successful trials
        normalize : bool
            If True, z-score normalize spike counts
            
        Returns:
        --------
        tuple : (bin_centers, spike_counts, n_trials)
        """
        # Get data for this specific cell
        # Note: No need to copy here since Cell.__init__ will make its own defensive copy
        cell_data = self.data[self.data['cell_ID'] == cell_id]
        
        if len(cell_data) == 0:
            return None, None, 0
        
        # Create a Cell instance for this cell (Cell.__init__ will copy the data)
        cell = Cell(cell_data)
        
        # Use Cell's aggregate_spikes_by_bins method
        bin_centers, spike_counts, n_trials = cell.aggregate_spikes_by_bins(
            epok=epok,
            bin_size=bin_size,
            alignment_point=alignment_point,
            trial_type=trial_type,
            direction=direction,
            ssd_number=ssd_number,
            success_only=success_only,
            normalize=normalize
        )
        
        return bin_centers, spike_counts, n_trials
    
    def get_all_cells_spike_counts(self, epok=[-500, 1000], bin_size=1,
                                    alignment_point='go_cue', trial_type=None, direction=None,
                                    ssd_number=None, success_only=True, normalize=True):
        """
        Get aggregated spike counts for all cells in the session.
        
        Parameters:
        -----------
        Same as get_cell_spike_counts, plus:
        normalize : bool
            If True, each cell's spike counts are z-score normalized individually
            
        Returns:
        --------
        tuple : (bin_centers, spike_counts_matrix, cell_ids_used)
            - bin_centers: time bins
            - spike_counts_matrix: 2D array (n_cells X n_bins)
            - cell_ids_used: list of cell IDs with data
        """
        spike_counts_list = []
        cells_used = []
        bin_centers = None
        
        # Collect spike counts with per-cell normalization
        for cell_id in self.cell_ids:
            bins, spike_counts, n_trials = self.get_cell_spike_counts(
                cell_id=cell_id,
                epok=epok,
                bin_size=bin_size,
                alignment_point=alignment_point,
                trial_type=trial_type,
                direction=direction,
                ssd_number=ssd_number,
                success_only=success_only,
                normalize=normalize  # Pass normalization to each cell
            )
            
            if bins is not None and n_trials > 0:
                spike_counts_list.append(spike_counts)
                cells_used.append(cell_id)
                
                if bin_centers is None:
                    bin_centers = bins
        
        if len(spike_counts_list) == 0:
            return None, None, []
        
        spike_counts_matrix = np.array(spike_counts_list)
        
        return bin_centers, spike_counts_matrix, cells_used
    
    # ==================== DATA GENERATION METHODS ====================
    
    def get_population_spike_counts_data(self, epok=[-500, 1000], bin_size=1,
                                         alignment_point='go_cue', trial_type=None,
                                         direction=None, ssd_number=None,
                                         success_only=True, normalize=True,
                                         sort_by_peak=True):
        """
        Get spike counts data for all cells in the session.
        DATA GENERATION METHOD - separated from plotting.
        
        Parameters:
        -----------
        epok : list
            Time window [start, end] in ms
        bin_size : int
            Bin size in ms (default: 1)
        alignment_point : str
            Event to align to
        trial_type : str, optional
            'GO', 'STOP', or 'CONT'
        direction : int, optional
            0 (right) or 180 (left)
        ssd_number : int, optional
            SSD number (1-4)
        success_only : bool
            Include only successful trials
        normalize : bool
            If True, z-score normalize spike counts per cell
        sort_by_peak : bool
            Sort cells by time of peak activity
        
        Returns:
        --------
        dict : Dictionary with keys:
            - 'bin_centers': time bins
            - 'spike_counts_matrix': 2D array (n_cells X n_bins), sorted if requested
            - 'cell_ids': list of cell IDs in the order shown
            - 'sort_idx': sorting indices used (if sort_by_peak=True)
            - 'params': dict of parameters used
        """
        # Get spike counts for all cells
        bin_centers, spike_counts_matrix, cells_used = self.get_all_cells_spike_counts(
            epok=epok,
            bin_size=bin_size,
            alignment_point=alignment_point,
            trial_type=trial_type,
            direction=direction,
            ssd_number=ssd_number,
            success_only=success_only,
            normalize=normalize
        )
        
        if spike_counts_matrix is None:
            return None
        
        # Sort by peak time if requested
        sort_idx = None
        if sort_by_peak:
            peak_times = np.argmax(spike_counts_matrix, axis=1)
            sort_idx = np.argsort(peak_times)
            spike_counts_matrix = spike_counts_matrix[sort_idx]
            cells_used = [cells_used[i] for i in sort_idx]
        
        return {
            'bin_centers': bin_centers,
            'spike_counts_matrix': spike_counts_matrix,
            'cell_ids': cells_used,
            'sort_idx': sort_idx,
            'params': {
                'epok': epok,
                'bin_size': bin_size,
                'alignment_point': alignment_point,
                'trial_type': trial_type,
                'direction': direction,
                'ssd_number': ssd_number,
                'success_only': success_only,
                'normalize': normalize,
                'sort_by_peak': sort_by_peak
            }
        }
    
    def get_population_PSTH_single_condition(self, epok=[-500, 1000], bin_size=1,
                                             alignment_point='go_cue', trial_type=None, 
                                             direction=None, ssd_number=None, 
                                             success_only=True, smooth=True, delta=False,
                                             smooth_ker_size=25, normalize_bins=False,
                                             normalize=False, sort_by_peak=True):
        """
        Get population PSTH data for a single condition.
        DATA GENERATION METHOD - separated from plotting.
        
        Parameters:
        -----------
        sort_by_peak : bool
            Sort cells by time of peak activity (ascending order by argmax)
        
        Returns:
        --------
        dict : Dictionary with keys:
            - 'bin_centers': time bins
            - 'psth_matrix': 2D array (n_cells X n_bins), sorted if requested
            - 'cell_ids': list of cell IDs in the order shown
            - 'sort_idx': sorting indices used (if sort_by_peak=True)
            - 'params': dict of parameters used
        """
        # Get PSTH for all cells
        bin_centers, psth_matrix, cells_used = self.get_all_cells_psth(
            epok=epok,
            bin_size=bin_size,
            alignment_point=alignment_point,
            trial_type=trial_type,
            direction=direction,
            ssd_number=ssd_number,
            success_only=success_only,
            smooth=smooth,
            smooth_ker_size=smooth_ker_size,
            delta=delta,
            normalize_bins=normalize_bins,
            normalize=normalize
        )
        
        if psth_matrix is None:
            return None
        
        # Sort by peak time if requested
        sort_idx = None
        if sort_by_peak:
            peak_times = np.argmax(psth_matrix, axis=1)
            sort_idx = np.argsort(peak_times)
            psth_matrix = psth_matrix[sort_idx]
            cells_used = [cells_used[i] for i in sort_idx]
        
        return {
            'bin_centers': bin_centers,
            'psth_matrix': psth_matrix,
            'cell_ids': cells_used,
            'sort_idx': sort_idx,
            'params': {
                'epok': epok,
                'bin_size': bin_size,
                'alignment_point': alignment_point,
                'trial_type': trial_type,
                'direction': direction,
                'ssd_number': ssd_number,
                'success_only': success_only,
                'smooth': smooth,
                'normalize': normalize,
                'sort_by_peak': sort_by_peak
            }
        }
    
    def get_population_PSTHs_left_right(self, epok=[-500, 1000], bin_size=1,
                                       alignment_point='go_cue', trial_type=None,
                                       ssd_number=None, success_only=True, smooth=True,
                                       delta=False, smooth_ker_size=25, 
                                       normalize_bins=False, normalize=False, 
                                       sort_by_peak=True):
        """
        Get population data for left vs right comparison.
        DATA GENERATION METHOD - separated from plotting.
        Cells are ordered by left direction peak, right uses same order.
        
        Returns:
        --------
        dict : Dictionary with keys:
            - 'left': data dict for left direction
            - 'right': data dict for right direction (reordered to match left)
            - 'cell_order': list of cell IDs in display order
        """
        # Get left direction data
        data_left = self.get_population_PSTH_single_condition(
            epok=epok, bin_size=bin_size, alignment_point=alignment_point,
            trial_type=trial_type, direction=180, ssd_number=ssd_number,
            success_only=success_only, smooth=smooth, delta=delta,
            smooth_ker_size=smooth_ker_size, normalize_bins=normalize_bins,
            normalize=normalize, sort_by_peak=sort_by_peak
        )
        
        if data_left is None:
            print("No data for left direction")
            return None
        
        cells_sorted = data_left['cell_ids']
        
        # Get right direction data (unsorted)
        data_right_unsorted = self.get_population_PSTH_single_condition(
            epok=epok, bin_size=bin_size, alignment_point=alignment_point,
            trial_type=trial_type, direction=0, ssd_number=ssd_number,
            success_only=success_only, smooth=smooth, delta=delta,
            smooth_ker_size=smooth_ker_size, normalize_bins=normalize_bins,
            normalize=normalize, sort_by_peak=False
        )
        
        if data_right_unsorted is None:
            print("No data for right direction")
            return None
        
        # Reorder right to match left's cell order
        cell_to_idx_right = {cell: idx for idx, cell in enumerate(data_right_unsorted['cell_ids'])}
        psth_right_sorted = np.zeros_like(data_left['psth_matrix'])
        
        for new_idx, cell_id in enumerate(cells_sorted):
            if cell_id in cell_to_idx_right:
                old_idx = cell_to_idx_right[cell_id]
                psth_right_sorted[new_idx] = data_right_unsorted['psth_matrix'][old_idx]
        
        # Create right data dict with reordered matrix
        data_right = data_right_unsorted.copy()
        data_right['psth_matrix'] = psth_right_sorted
        data_right['cell_ids'] = cells_sorted
        
        return {
            'left': data_left,
            'right': data_right,
            'cell_order': cells_sorted
        }
    
    def get_population_PSTHs_trial_types(self, epok_go=[-500, 1000], epok_stop=[-500, 1000],
                                        bin_size=1, direction=None,
                                        ssd_number=None, success_only=True, smooth=True,
                                        normalize=True):
        """
        Get population data for GO, STOP, and CONT trial comparison.
        DATA GENERATION METHOD - separated from plotting.
        - GO: aligned to go_cue
        - STOP: aligned to stop_cue
        - CONT: aligned to stop_cue
        All three use same cell ordering (based on GO trial peaks).
        
        Returns:
        --------
        dict : Dictionary with keys:
            - 'go': data dict for GO trials
            - 'stop': data dict for STOP trials (reordered to match GO)
            - 'cont': data dict for CONT trials (reordered to match GO)
            - 'cell_order': list of cell IDs in display order
        """
        # Get GO trials (aligned to go_cue) and sort by peak
        data_go = self.get_population_PSTH_single_condition(
            epok=epok_go, bin_size=bin_size, alignment_point='go_cue',
            trial_type='GO', direction=direction, ssd_number=None,
            success_only=success_only, smooth=smooth, delta=False,
            smooth_ker_size=25, normalize=normalize, sort_by_peak=True
        )

        if data_go is None:
            print("No GO trial data")
            return None
        
        cells_sorted = data_go['cell_ids']
        
        # Get STOP trials (aligned to stop_cue, unsorted)
        data_stop_unsorted = self.get_population_PSTH_single_condition(
            epok=epok_stop, bin_size=bin_size, alignment_point='stop_cue',
            trial_type='STOP', direction=direction, ssd_number=ssd_number,
            success_only=success_only, smooth=smooth, delta=False,
            smooth_ker_size=25, normalize=normalize, sort_by_peak=True
        )
        
        # Get CONT trials (aligned to stop_cue, unsorted)
        data_cont_unsorted = self.get_population_PSTH_single_condition(
            epok=epok_stop, bin_size=bin_size, alignment_point='stop_cue',
            trial_type='CONT', direction=direction, ssd_number=ssd_number,
            success_only=success_only, smooth=smooth, delta=False,
            smooth_ker_size=25, normalize=normalize, sort_by_peak=True
        )
        
        # Reorder STOP to match GO's cell order
        if data_stop_unsorted is not None:
            cell_to_idx_stop = {cell: idx for idx, cell in enumerate(data_stop_unsorted['cell_ids'])}
            psth_stop_sorted = np.zeros((len(cells_sorted), data_stop_unsorted['psth_matrix'].shape[1]))
            
            for new_idx, cell_id in enumerate(cells_sorted):
                if cell_id in cell_to_idx_stop:
                    old_idx = cell_to_idx_stop[cell_id]
                    psth_stop_sorted[new_idx] = data_stop_unsorted['psth_matrix'][old_idx]
            
            data_stop = data_stop_unsorted.copy()
            data_stop['psth_matrix'] = psth_stop_sorted
            data_stop['cell_ids'] = cells_sorted
        else:
            data_stop = None
        
        # Reorder CONT to match GO's cell order
        if data_cont_unsorted is not None:
            cell_to_idx_cont = {cell: idx for idx, cell in enumerate(data_cont_unsorted['cell_ids'])}
            psth_cont_sorted = np.zeros((len(cells_sorted), data_cont_unsorted['psth_matrix'].shape[1]))
            
            for new_idx, cell_id in enumerate(cells_sorted):
                if cell_id in cell_to_idx_cont:
                    old_idx = cell_to_idx_cont[cell_id]
                    psth_cont_sorted[new_idx] = data_cont_unsorted['psth_matrix'][old_idx]
            
            data_cont = data_cont_unsorted.copy()
            data_cont['psth_matrix'] = psth_cont_sorted
            data_cont['cell_ids'] = cells_sorted
        else:
            data_cont = None
        
        return {
            'go': data_go,
            'stop': data_stop,
            'cont': data_cont,
            'cell_order': cells_sorted
        }
    
    # ==================== PLOTTING METHODS ====================
    
    def plot_population_PSTH_heatmap(self, data=None, **kwargs):
        """
        Plot a heatmap of all cells' activity in the session.
        Can use pre-generated data or generate new data.
        
        Parameters:
        -----------
        data : dict, optional
            Pre-generated data from get_population_PSTH_single_condition()
            If None, will generate data using **kwargs
        **kwargs : dict
            Parameters for get_population_PSTH_single_condition() if data is None
        
        Returns:
        --------
        hv.Image : Heatmap plot
        """
        # Generate data if not provided
        if data is None:
            data = self.get_population_PSTH_single_condition(**kwargs)
        
        if data is None:
            print("No data found for specified conditions")
            return None
        
        psth_matrix = data['psth_matrix']
        bin_centers = data['bin_centers']
        params = data['params']
        cell_ids = data['cell_ids']
        sort_idx = data.get('sort_idx', cell_ids)
        print(f'sort_idx: {sort_idx}')
        print(f'cell_ids: {cell_ids}')
        
        # Create DataFrame for heatmap with named axes
        psth_df = pd.DataFrame(
            psth_matrix,
            columns=bin_centers,
            # index=sort_idx if sort_idx is not None else cell_ids
        )
        psth_df.columns.name = 'Time (ms)'
        psth_df.index.name = 'Cell ID'
        
        # Create heatmap - hvplot will automatically use the axis names
        heatmap = psth_df.hvplot.heatmap().opts(
            opts.HeatMap(
                cmap='Plasma',
                colorbar=True,
                width=800,
                height=600,
                xlabel=f'Time from {params["alignment_point"]} (ms)',
                ylabel='Neurons',
                title=f'Session {self.session_id} - {params["trial_type"] or "All"} trials - Dir {params["direction"] if params["direction"] is not None else "Both"}',
                invert_yaxis=False,
                tools=['hover'],
                clabel='Firing Rate',
                xlim=(params['epok'][0], params['epok'][1])
            )
        )
        
        return heatmap
    
    def plot_population_spike_counts_heatmap(self, data=None, **kwargs):
        """
        Plot a heatmap of all cells' spike counts in the session.
        Can use pre-generated data or generate new data.
        
        Parameters:
        -----------
        data : dict, optional
            Pre-generated data from get_population_spike_counts_data()
            If None, will generate data using **kwargs
        **kwargs : dict
            Parameters for get_population_spike_counts_data() if data is None
        
        Returns:
        --------
        hv.HeatMap : Heatmap plot
        """
        # Generate data if not provided
        if data is None:
            data = self.get_population_spike_counts_data(**kwargs)
        
        if data is None:
            print("No data found for specified conditions")
            return None
        
        spike_counts_matrix = data['spike_counts_matrix']
        bin_centers = data['bin_centers']
        params = data['params']
        cell_ids = data['cell_ids']
        
        # Create DataFrame for heatmap with named axes
        spike_counts_df = pd.DataFrame(
            spike_counts_matrix,
            columns=bin_centers,
            # index=cell_ids
        )
        spike_counts_df.columns.name = 'Time (ms)'
        spike_counts_df.index.name = 'Cell ID'
        
        # Create heatmap - hvplot will automatically use the axis names
        heatmap = spike_counts_df.hvplot.heatmap(
            xmarks_muted=True,
            xmarks_visible=False
        ).opts(
            opts.HeatMap(
                cmap='Plasma',
                colorbar=True,
                width=800,
                height=600,
                xlabel=f'Time from {params["alignment_point"]} (ms)',
                ylabel='Neurons',
                title=f'Session {self.session_id} - Spike Counts (bin={params["bin_size"]}ms)\n{params["trial_type"] or "All"} trials - Dir {params["direction"] if params["direction"] is not None else "Both"}',
                invert_yaxis=False,
                tools=['hover'],
                clabel='Spike Count' if not params['normalize'] else 'Normalized Spike Count',
                xlim=(params['epok'][0], params['epok'][1])
            )
        )

        return heatmap
    
    def plot_left_right_PSTH_comparison(self, data=None, **kwargs):
        """
        Plot left vs right direction heatmaps with same cell ordering.
        Cells are ordered by left direction peak, right uses same order.
        
        Parameters:
        -----------
        data : dict, optional
            Pre-generated data from get_population_PSTHs_left_right()
            If None, will generate data using **kwargs
        **kwargs : dict
            Parameters for get_population_PSTHs_left_right() if data is None
        
        Returns:
        --------
        hv.Layout : Side-by-side heatmaps
        """
        # Generate data if not provided
        if data is None:
            data = self.get_population_PSTHs_left_right(**kwargs)
        
        if data is None:
            return None
        
        data_left = data['left']
        data_right = data['right']
        params_left = data_left['params']
        cell_ids = data_left['cell_ids']
        
        # Create DataFrames for heatmaps with named axes
        n_cells = data_left['psth_matrix'].shape[0]
        
        psth_left_df = pd.DataFrame(
            data_left['psth_matrix'],
            columns=data_left['bin_centers'],
            index=cell_ids
        )
        psth_left_df.columns.name = 'Time (ms)'
        psth_left_df.index.name = 'Cell ID'
        
        psth_right_df = pd.DataFrame(
            data_right['psth_matrix'],
            columns=data_right['bin_centers'],
            index=cell_ids
        )
        psth_right_df.columns.name = 'Time (ms)'
        psth_right_df.index.name = 'Cell ID'
        
        # Create heatmaps - hvplot will automatically use the axis names
        heatmap_left = psth_left_df.hvplot.heatmap().opts(
            opts.HeatMap(
                cmap='Plasma',
                colorbar=True,
                width=400,
                height=600,
                xlabel=f'Time from {params_left["alignment_point"]} (ms)',
                ylabel='Neurons',
                title=f'Left (180°) - {params_left["trial_type"] or "All"}',
                invert_yaxis=False,
                tools=['hover'],
                clabel='Firing Rate',
                xlim=(params_left['epok'][0], params_left['epok'][1])
            )
        )
        
        heatmap_right = psth_right_df.hvplot.heatmap().opts(
            opts.HeatMap(
                cmap='Plasma',
                colorbar=True,
                width=400,
                height=600,
                xlabel=f'Time from {params_left["alignment_point"]} (ms)',
                ylabel='Neurons',
                title=f'Right (0°) - {params_left["trial_type"] or "All"}',
                invert_yaxis=False,
                tools=['hover'],
                clabel='Firing Rate',
                xlim=(params_left['epok'][0], params_left['epok'][1])
            )
        )
        
        return (heatmap_left + heatmap_right).cols(2)
    
    def plot_trial_type_PSTH_comparison(self, data_left=None, data_right=None, **kwargs):
        """
        Plot GO, STOP, and CONT trials with appropriate alignments in 3X2 grid.
        - Rows: GO, STOP, CONT trial types
        - Columns: Left (180°), Right (0°) directions
        - GO: aligned to go_cue
        - STOP: aligned to stop_cue
        - CONT: aligned to stop_cue
        
        All plots use the same cell ordering (based on GO left direction peaks).
        
        Parameters:
        -----------
        data_left : dict, optional
            Pre-generated data for left direction from get_population_PSTHs_trial_types()
        data_right : dict, optional
            Pre-generated data for right direction from get_population_PSTHs_trial_types()
        **kwargs : dict
            Parameters for get_population_PSTHs_trial_types() if data not provided
            Note: 'direction' parameter will be ignored as both directions are plotted
        
        Returns:
        --------
        hv.Layout : 3X2 grid of heatmaps
        """
        # Remove direction parameter if provided in kwargs
        kwargs.pop('direction', None)
        
        # Generate left direction data if not provided
        if data_left is None:
            data_left = self.get_population_PSTHs_trial_types(direction=180, **kwargs)
        
        # Generate right direction data if not provided  
        if data_right is None:
            data_right = self.get_population_PSTHs_trial_types(direction=0, **kwargs)
        
        if data_left is None or data_right is None:
            print("Missing data for one or both directions")
            return None
        
        # Use left direction's cell ordering for both
        cell_order = data_left['cell_order']
        n_cells = len(cell_order)
        
        # Reorder right direction data to match left
        for trial_key in ['go', 'stop', 'cont']:
            if data_right[trial_key] is not None and data_left[trial_key] is not None:
                right_data = data_right[trial_key]
                left_cell_order = data_left['cell_order']
                
                # Create mapping from cell_id to index in right data
                cell_to_idx = {cell: idx for idx, cell in enumerate(right_data['cell_ids'])}
                
                # Reorder right matrix to match left's cell order
                reordered_matrix = np.zeros_like(data_left[trial_key]['psth_matrix'])
                for new_idx, cell_id in enumerate(left_cell_order):
                    if cell_id in cell_to_idx:
                        old_idx = cell_to_idx[cell_id]
                        reordered_matrix[new_idx] = right_data['psth_matrix'][old_idx]
                
                data_right[trial_key]['psth_matrix'] = reordered_matrix
                data_right[trial_key]['cell_ids'] = left_cell_order
        
        # Create heatmaps for each trial type and direction
        plots = []
        
        for trial_type, trial_key in [('GO', 'go'), ('STOP', 'stop'), ('CONT', 'cont')]:
            data_l = data_left[trial_key]
            data_r = data_right[trial_key]
            
            if data_l is None or data_r is None:
                # Create placeholder if data missing
                plots.append(hv.Text(0, 0, f'No {trial_type} data').opts(width=400, height=300))
                plots.append(hv.Text(0, 0, f'No {trial_type} data').opts(width=400, height=300))
                continue
            
            params_l = data_l['params']
            params_r = data_r['params']
            cell_ids = data_l['cell_ids']
            
            # Determine alignment and xlabel
            if trial_type == 'GO':
                alignment = 'go_cue'
                xlabel = 'Time from go_cue (ms)'
            else:
                alignment = 'stop_cue'
                xlabel = 'Time from stop_cue (ms)'
            
            # Create DataFrames for heatmaps with named axes
            psth_left_df = pd.DataFrame(
                data_l['psth_matrix'],
                columns=data_l['bin_centers'],
                index=cell_ids
            )
            psth_left_df.columns.name = 'Time (ms)'
            psth_left_df.index.name = 'Cell ID'
            
            psth_right_df = pd.DataFrame(
                data_r['psth_matrix'],
                columns=data_r['bin_centers'],
                index=cell_ids
            )
            psth_right_df.columns.name = 'Time (ms)'
            psth_right_df.index.name = 'Cell ID'
            
            # Left direction heatmap - hvplot will automatically use the axis names
            heatmap_left = psth_left_df.hvplot.heatmap().opts(
                opts.HeatMap(
                    cmap='Plasma',
                    colorbar=True,
                    width=400,
                    height=300,
                    xlabel=xlabel,
                    ylabel='Neurons',
                    title=f'{trial_type} - Left (180°)',
                    invert_yaxis=False,
                    tools=['hover'],
                    clabel='Firing Rate',
                    xlim=(params_l['epok'][0], params_l['epok'][1])
                )
            )
            
            # Right direction heatmap - hvplot will automatically use the axis names
            heatmap_right = psth_right_df.hvplot.heatmap().opts(
                opts.HeatMap(
                    cmap='Plasma',
                    colorbar=True,
                    width=400,
                    height=300,
                    xlabel=xlabel,
                    ylabel='Neurons',
                    title=f'{trial_type} - Right (0°)',
                    invert_yaxis=False,
                    tools=['hover'],
                    clabel='Firing Rate',
                    xlim=(params_r['epok'][0], params_r['epok'][1])
                )
            )
            
            plots.extend([heatmap_left, heatmap_right])
        
        # Create 3×2 layout (3 rows, 2 columns)
        return hv.Layout(plots).cols(2)
    
    def plot_trial_type_PSTH_by_ssd(self, trial_type='STOP', ssd_numbers=None, **kwargs):
        """
        Plot STOP or CONT trials separated by SSD number, for both left and right directions.
        Creates a 42 grid with:
        - Rows: SSD numbers (SSD1, SSD2, SSD3, SSD4)
        - Columns: Left (180°) and Right (0°) directions
        
        All plots use the same cell ordering (based on GO left direction peaks).
        
        Parameters:
        -----------
        trial_type : str, optional
            'STOP' or 'CONT'. Default is 'STOP'
        ssd_numbers : list, optional
            List of SSD numbers to plot. If None, uses all available SSDs
        **kwargs : dict
            Parameters for data generation:
            - epok_stop : list, time window for STOP/CONT trials
            - bin_size : int, bin size in ms
            - success_only : bool, include only successful trials
            - smooth : bool, apply Gaussian smoothing
            - normalize : bool, normalize firing rates
        
        Returns:
        --------
        hv.Layout : 4X2 grid of heatmaps (SSD by direction)
        """
        # Validate trial_type
        if trial_type not in ['STOP', 'CONT']:
            print(f"Invalid trial_type '{trial_type}'. Must be 'STOP' or 'CONT'")
            return None
        
        # Get available SSD numbers if not specified
        if ssd_numbers is None:
            stop_cont_data = self.data[self.data['type'] == trial_type]
            ssd_numbers = sorted([x for x in stop_cont_data['ssd_number'].unique() if pd.notna(x)])
        
        # Get reference cell ordering from GO trials (left direction)
        epok_go = kwargs.get('epok_go', kwargs.get('epok_stop', [-200, 700]))
        bin_size = kwargs.get('bin_size', 10)
        success_only = kwargs.get('success_only', True)
        smooth = kwargs.get('smooth', True)
        normalize = kwargs.get('normalize', True)
        
        # Get GO data for cell ordering
        data_go = self.get_population_PSTH_single_condition(
            epok=epok_go, bin_size=bin_size, alignment_point='go_cue',
            trial_type='GO', direction=180, ssd_number=None,
            success_only=success_only, smooth=smooth, delta=False,
            smooth_ker_size=25, normalize=normalize, sort_by_peak=True
        )
        
        if data_go is None:
            print("No GO trial data for establishing cell order")
            return None
        
        cell_order = data_go['cell_ids']
        n_cells = len(cell_order)
        
        # Get epoch for STOP/CONT trials
        epok_stop = kwargs.get('epok_stop', [-200, 700])
        
        # Create plots for each SSD, with left/right as columns
        plots = []
        
        for ssd in ssd_numbers:
            # Create left and right plots for this SSD
            for direction, dir_label in [(180, 'Left'), (0, 'Right')]:
                # Get data for this condition
                data = self.get_population_PSTH_single_condition(
                    epok=epok_stop, bin_size=bin_size, alignment_point='stop_cue',
                    trial_type=trial_type, direction=direction, ssd_number=ssd,
                    success_only=success_only, smooth=smooth, delta=False,
                    smooth_ker_size=25, normalize=normalize, sort_by_peak=True  
                )
                
                if data is None or len(data['cell_ids']) == 0:
                    # Create placeholder
                    plots.append(
                        hv.Text(0, 0, f'No data').opts(
                            width=400, height=250, 
                            title=f'{trial_type} SSD {int(ssd)} - {dir_label} ({direction}°)'
                        )
                    )
                    continue
                
                # Reorder to match reference cell order
                cell_to_idx = {cell: idx for idx, cell in enumerate(data['cell_ids'])}
                reordered_matrix = np.zeros((len(cell_order), data['psth_matrix'].shape[1]))
                
                for new_idx, cell_id in enumerate(cell_order):
                    if cell_id in cell_to_idx:
                        old_idx = cell_to_idx[cell_id]
                        reordered_matrix[new_idx] = data['psth_matrix'][old_idx]
                
                params = data['params']
                
                # Create DataFrame for heatmap with named axes
                psth_df = pd.DataFrame(
                    reordered_matrix,
                    columns=data['bin_centers'],
                    index=cell_order
                )
                psth_df.columns.name = 'Time (ms)'
                psth_df.index.name = 'Cell ID'
                
                # Create heatmap - hvplot will automatically use the axis names
                heatmap = psth_df.hvplot.heatmap().opts(
                    opts.HeatMap(
                        cmap='Plasma',
                        colorbar=True,
                        width=400,
                        height=250,
                        xlabel='Time from stop_cue (ms)',
                        ylabel='Neurons',
                        title=f'{trial_type} SSD {int(ssd)} - {dir_label} ({direction}°)',
                        invert_yaxis=False,
                        tools=['hover'],
                        fontsize={'title': 10, 'labels': 9, 'ticks': 8},
                        clabel='Firing Rate',
                        xlim=(params['epok'][0], params['epok'][1])
                    )
                )
                
                plots.append(heatmap)
        
        # Create layout with 2 columns (Left, Right)
        return hv.Layout(plots).cols(2)

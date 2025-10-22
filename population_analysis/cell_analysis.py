"""
Cell Analysis Module for MSN Population Analysis

This module contains classes for analyzing Medium Spiny Neuron (MSN) recordings
during a countermanding stop-signal task (CSST).

Classes:
    - MSNCell: Single-cell analysis with raster plots, histograms, and PSTHs
    - PopulationAnalyzer: Population-level analysis across all cells
    - Session: Session-level population analysis with normalized heatmaps

Author: Barak
Date: October 2025
"""

import pandas as pd
import numpy as np
import holoviews as hv
from holoviews import opts
from scipy.ndimage import gaussian_filter1d
import hvplot.pandas  # Enable hvplot for pandas DataFrames
from scipy.stats import zscore
from numpy.exceptions import AxisError


class MSNCell:
    """
    Class representing a single MSN cell and its activity across trials.
    Handles spike alignment, raster plotting, and PSTH generation.
    """
    
    # Color schemes for different trial conditions
    DIRECTION_COLORS = {0: '#1f77b4', 180: '#ff7f0e'}  # Blue for right (0°), Orange for left (180°)
    TYPE_COLORS = {'GO': '#2ca02c', 'STOP': '#d62728', 'CONT': '#9467bd'}  # Green, Red, Purple
    SSD_COLORS = {
        1: '#000000', 2: '#0072B2', 3: '#D55E00', 4: '#009E73',
        'GO': '#2ca02c',  # Green for GO trials
        'combined': '#1f77b4' 
    }  # Different colors for each SSD
    
    
    def __init__(self, cell_df, verbose=False):
        """
        Initialize MSN cell with its trial data.
        
        Parameters:
        -----------
        cell_df : pd.DataFrame
            DataFrame containing all trials for a single cell
        """
        self.data = cell_df.copy().reset_index(drop=True)
        self.cell_id = cell_df.iloc[0]['cell_ID']
        self.cell_type = cell_df.iloc[0]['cell_type']
        self.sessions = cell_df['trial_session'].unique()
        
        # Sort data by relevant columns for organized plotting
        self.data = self.data.sort_values(by=['type', 'dir', 'ssd_number', 'trial_failed']).reset_index(drop=True)
        
        # Trial characteristics
        self.trial_types = sorted(self.data['type'].unique())
        self.directions = sorted(self.data['dir'].unique())
        self.ssd_numbers = sorted(self.data['ssd_number'].dropna().unique())
        
        if verbose:
            print(f"Cell {self.cell_id} initialized:")
            print(f"  - Total trials: {len(self.data)}")
            print(f"  - Trial types: {self.trial_types}")
            print(f"  - Directions: {self.directions}")
            print(f"  - SSD numbers: {self.ssd_numbers}")
    
    def align_spikes_to_event(self, alignment_point='go_cue'):
        """
        Align spike times to a specific event (go_cue, stop_cue, or saccade onset).
        
        Parameters:
        -----------
        alignment_point : str or float
            Event to align to: 'go_cue', 'stop_cue', 'first_relevant_saccade', or a specific time
        
        Returns:
        --------
        pd.Series : Series of aligned spike times for each trial
        """
        def get_alignment_time(row):
            if alignment_point == 'go_cue':
                return row['go_cue']
            elif alignment_point == 'stop_cue':
                return row['stop_cue'] if not pd.isna(row['stop_cue']) else row['go_cue']
            elif alignment_point == 'first_relevant_saccade':
                saccade = row['first_relevant_saccade']
                if isinstance(saccade, (list, np.ndarray)) and len(saccade) > 0:
                    return saccade[0]
                return np.nan
            else:
                return alignment_point
        
        def align_spikes(row):
            t_0 = get_alignment_time(row)
            if pd.isna(t_0):
                return np.array([])
            spikes = np.array(row['neural_data'], dtype=float)
            return spikes - t_0
        
        aligned = self.data.apply(align_spikes, axis=1)
        self.data[f'spikes_aligned_to_{alignment_point}'] = aligned
        return aligned
    
    def filter_trials(self, trial_type=None, direction=None, ssd_number=None, 
                     success_only=False, failed_only=False):
        """
        Filter trials based on various criteria.
        
        Parameters:
        -----------
        trial_type : str, optional
            'GO', 'STOP', or 'CONT'
        direction : int, optional
            0 (right) or 180 (left)
        ssd_number : int, optional
            SSD number (1-4)
        success_only : bool
            Include only successful trials
        failed_only : bool
            Include only failed trials
        
        Returns:
        --------
        pd.DataFrame : Filtered DataFrame
        """
        filtered = self.data.copy()
        
        if trial_type is not None:
            filtered = filtered[filtered['type'] == trial_type]
        
        if direction is not None:
            filtered = filtered[filtered['dir'] == direction]
        
        if ssd_number is not None:
            filtered = filtered[filtered['ssd_number'] == ssd_number]
        
        if success_only:
            filtered = filtered[filtered['trial_failed'] == False]
        
        if failed_only:
            filtered = filtered[filtered['trial_failed'] == True]
        
        return filtered
    
    def plot_raster(self, alignment_point='go_cue', epok=[-200, 500], 
                   color_by='type', **filter_kwargs):
        """
        Create a raster plot of spike times across trials.
        
        Parameters:
        -----------
        alignment_point : str
            Event to align to: 'go_cue', 'stop_cue', or 'first_relevant_saccade'
        epok : list
            Time window [start, end] in ms relative to alignment point
        color_by : str
            How to color trials: 'type', 'direction', 'ssd', 'outcome'
        **filter_kwargs : dict
            Additional filtering criteria
        
        Returns:
        --------
        hv.NdOverlay : Raster plot
        """
        # Filter trials if requested
        if filter_kwargs:
            plot_data = MSNCell(self.filter_trials(**filter_kwargs))
        else:
            plot_data = self
        
        # Align spikes
        col_name = f'spikes_aligned_to_{alignment_point}'
        if col_name not in plot_data.data.columns:
            plot_data.align_spikes_to_event(alignment_point)
        
        # Create raster plot
        overlay = hv.Curve([])
        col_names = np.array(range(epok[0], epok[1] + 1, 1))
        spikes_arr = pd.DataFrame(
            np.zeros(
                (plot_data.data.shape[0], col_names.size)
            ),
            columns=col_names
        )
        
        plot_data.data = plot_data.data.sort_values(
            by=['type', 'ssd_number', 'dir']
        ).reset_index(drop=True)

        for i, (idx, row) in enumerate(plot_data.data.iterrows()):
            col = plot_data.data[f'spikes_aligned_to_{alignment_point}'].iloc[idx]
            col = np.unique(col[(col >= epok[0]) & (col <= epok[1])].astype(int))
            color_int = row['ssd_number'] if not pd.isna(row['ssd_number']) else 1
            color_int = (color_int * 2) if row['dir'] == 180 else color_int
            spikes_arr.loc[idx, col] = int(color_int)
            if spikes_arr.columns.value_counts().max() > 1:
                print(f"Warning: Multiple spikes in the same ms for trial index {idx}")
                raise ValueError(col, i, idx, row)

        # First color is white for no spikes
        colors = ['#ffffff'] + [
            self.SSD_COLORS[key] for key in range(1,5)
        ]
        
        overlay *= spikes_arr.hvplot.heatmap(x='columns', y='index').opts(
            cmap=colors, colorbar=False, width=800, height=600
        )

        # Configure plot
        title = f"Cell {self.cell_id} - Raster Plot (aligned to {alignment_point})"
        if filter_kwargs:
            filter_str = ', '.join([f"{k}={v}" for k, v in filter_kwargs.items()])
            title += f" | Filters: {filter_str}"
        
        plot = overlay.opts(
            opts.HeatMap(
                xlabel=f'Time from {alignment_point} (ms)',
                ylabel='Trial #',
                title=title,
                width=800, height=600,
                xlim=(epok[0], epok[1])
            )
        )
        
        return plot
    
    def plot_raster_by_type_direction(self, epok=[-200, 500], 
                                      alignment_point='go_cue', show_legend=True):
        """
        Create separate raster plots for each trial type and direction combination.
        Uses the heatmap-based raster plot method to ensure even presentation of all trials.
        - Only successful trials (trial_failed = False)
        - Color coded by SSD number and direction
        - Sorted by SSD number within each plot
        
        Parameters:
        -----------
        epok : list
            Time window [start, end] in ms (default: [-200, 700])
        alignment_point : str
            Event to align to: 'go_cue', 'stop_cue', or 'first_relevant_saccade'
        show_legend : bool
            Whether to display color legend (default: True)
        
        Returns:
        --------
        dict : Nested dictionary {direction: {trial_type: plot}}
        """
        plots = {}
        for direction in self.directions:
            dir_label = "Right (0°)" if direction == 0 else "Left (180°)"
            plots[direction] = {}

            trials = ['GO', 'STOP', 'CONT']
            if alignment_point != 'go_cue':
                trials.remove('GO')
            
            for trial_type in trials:
                # Use the plot_raster method with appropriate filters
                # This delegates to the heatmap-based implementation
                try:
                    plot = self.plot_raster(
                        alignment_point=alignment_point,
                        epok=epok,
                        color_by='ssd',  # Color by SSD number
                        direction=direction,
                        trial_type=trial_type,
                        success_only=True
                    )
                    
                    # Customize the title to match the original format
                    legend_text = ""
                    if show_legend:
                        if trial_type == 'GO':
                            legend_text = "\nColors: Intensity indicates trial presence"
                        else:
                            legend_text = "\nColors: Different intensities for SSD1-4"
                    
                    title_text = f"Cell {self.cell_id} - {trial_type} trials - {dir_label} (Success Only){legend_text}"
                    
                    # Update plot options with custom title and dimensions
                    plot = plot.opts(
                        opts.HeatMap(
                            title=title_text,
                            ylabel='Trial # (sorted by SSD)',
                            height=300
                        )
                    )
                    
                    plots[direction][trial_type] = plot
                    
                except Exception as e:
                    # If there's no data for this combination, skip it
                    print(f"No data for {trial_type} - {dir_label}: {e}")
                    continue
        
        return plots
    
    def aggregate_spikes_by_bins(self, epok=[-200, 500], bin_size=11,
                                 alignment_point='go_cue', trial_type=None, 
                                 direction=None, ssd_number=None, 
                                 success_only=True, normalize=False):
        """
        Aggregate spikes into bins for a specific set of trials.
        
        Parameters:
        -----------
        epok : list
            Time window [start, end] in ms (default: [-200, 500])
        bin_size : int
            Bin size in ms (default: 10)
        alignment_point : str
            Event to align to: 'go_cue', 'stop_cue', or 'first_relevant_saccade'
        trial_type : str, optional
            'GO', 'STOP', or 'CONT'
        direction : int, optional
            0 (right) or 180 (left)
        ssd_number : int, optional
            SSD number (1-4)
        success_only : bool
            Include only successful trials (default: True)
        normalize : bool
            If True, normalize spike counts to [0, 1] (default: False)
        
        Returns:
        --------
        tuple : (bin_centers, spike_counts, n_trials)
            - bin_centers: array of bin center times
            - spike_counts: aggregated spike counts per bin
            - n_trials: number of trials used
        """
        # Align spikes if not already done
        col_name = f'spikes_aligned_to_{alignment_point}'
        if col_name not in self.data.columns:
            self.align_spikes_to_event(alignment_point)
        
        # Filter trials
        filtered_data = self.filter_trials(
            trial_type=trial_type,
            direction=direction,
            ssd_number=ssd_number,
            success_only=success_only
        )
        
        if len(filtered_data) == 0:
            return None, None, 0
        
        # Create bins
        bins = np.arange(epok[0], epok[1] + bin_size, bin_size)
        bin_centers = bins[:-1] + bin_size / 2
        
        # Count spikes in each bin across all trials
        spike_counts = np.zeros(len(bins) - 1)
        for _, row in filtered_data.iterrows():
            spikes = row[col_name]
            counts, _ = np.histogram(spikes, bins=bins)
            spike_counts += counts
        
        # Normalize if requested
        if normalize:
            try:
                spike_counts = zscore(spike_counts, axis=1)
            except AxisError:
                spike_counts = zscore(spike_counts)
        
        return bin_centers, spike_counts, len(filtered_data)
    
    def calculate_psth(self, epok=[-200, 500], bin_size=10,
                      alignment_point='go_cue', trial_type=None, direction=None,
                      ssd_number=None, success_only=True, smooth=True, delta=False,
                      smooth_ker_size=25, normalize_bins=False):
        """
        Calculate PSTH (peri-stimulus time histogram) with firing rate and Gaussian smoothing.
        
        Parameters:
        -----------
        epok : list
            Time window [start, end] in ms (default: [-200, 700])
        bin_size : int
            Bin size in ms (default: 10)
        alignment_point : str
            Event to align to: 'go_cue', 'stop_cue', or 'first_relevant_saccade'
        trial_type : str, optional
            'GO', 'STOP', or 'CONT'
        direction : int, optional
            0 (right) or 180 (left)
        ssd_number : int, optional
            SSD number (1-4)
        success_only : bool
            Include only successful trials (default: True)
        smooth : bool
            If True, apply Gaussian smoothing with sigma=bin_size (default: True)
        smooth_ker_size : int
            Kernel size for Gaussian smoothing (default: 25)
        delta: bool
            if True looke center to the mean firing rate
        normalize_bins : bool
            If True, z-score spike counts before calculating firing rate (default: False)
        
        Returns:
        --------
        tuple : (bin_centers, firing_rate, n_trials)
            - bin_centers: array of bin center times
            - firing_rate: firing rate in spikes/sec (smoothed if smooth=True)
            - n_trials: number of trials used
        """
        # Get spike counts from aggregate_spikes_by_bins
        bin_centers, spike_counts, n_trials = self.aggregate_spikes_by_bins(
            epok=epok,
            bin_size=bin_size,
            alignment_point=alignment_point,
            trial_type=trial_type,
            direction=direction,
            ssd_number=ssd_number,
            success_only=success_only,
            normalize=normalize_bins
        )
        
        if bin_centers is None:
            return None, None, 0
        
        # Convert spike counts to firing rate (spikes/sec)
        # spike_counts is total spikes across all trials
        firing_rate = (spike_counts / n_trials) / (bin_size / 1000)
        if delta:
            firing_rate = firing_rate - np.mean(firing_rate)
        
        # Apply Gaussian smoothing if requested
        if smooth:
            firing_rate = gaussian_filter1d(firing_rate, sigma=smooth_ker_size)
        
        return bin_centers, firing_rate, n_trials
    
    def plot_histogram_by_type_direction(self, epok=[-200, 500], bin_size=1,
                                        alignment_point='go_cue', separate_ssd=False,
                                        normalize=False):
        """
        Create histogram plots for each trial type and direction.
        - Only successful trials (trial_failed = False)
        - Can be combined across all SSDs or separate for each SSD
        - Simple spike count histogram (not converted to firing rate like PSTH)
        
        Parameters:
        -----------
        epok : list
            Time window [start, end] in ms (default: [-200, 700])
        bin_size : int
            Bin size in ms (default: 10)
        alignment_point : str
            Event to align to: 'go_cue', 'stop_cue', or 'first_relevant_saccade'
        separate_ssd : bool
            If True, create separate histograms for each SSD. If False, combine all SSDs.
        normalize : bool
            If True, normalize spike counts to [0, 1] (default: False)
        
        Returns:
        --------
        dict : Nested dictionary {direction: {trial_type: plot}}
        """
        # Color palette for SSD numbers
        ssd_colors = self.SSD_COLORS

        plots = {}
        
        for direction in self.directions:
            dir_label = "Right (0°)" if direction == 0 else "Left (180°)"
            plots[direction] = {}
            
            for trial_type in ['GO', 'STOP', 'CONT']:
                # Check if we have data for this combination
                test_data = self.filter_trials(
                    direction=direction,
                    trial_type=trial_type,
                    success_only=True
                )
                
                if len(test_data) == 0:
                    continue
                
                if separate_ssd and trial_type != 'GO':
                    # Create separate curves for each SSD
                    overlay = hv.NdOverlay()
                    
                    ssd_numbers = sorted(test_data['ssd_number'].dropna().unique())
                    for ssd_num in ssd_numbers:
                        bin_centers, spike_counts, n_trials = self.aggregate_spikes_by_bins(
                            epok=epok,
                            bin_size=bin_size,
                            alignment_point=alignment_point,
                            trial_type=trial_type,
                            direction=direction,
                            ssd_number=ssd_num,
                            success_only=True,
                            normalize=normalize
                        )
                        
                        if bin_centers is None:
                            continue
                        
                        # Create histogram for this SSD
                        # Convert to edges format for hv.Histogram
                        edges = np.concatenate([bin_centers - bin_size/2, [bin_centers[-1] + bin_size/2]])
                        hist = hv.Histogram(
                            (edges, spike_counts),
                            kdims='Time', 
                            vdims='Spike Count' if not normalize else 'Normalized Spike Count',
                            label=f'SSD{int(ssd_num)} (n={n_trials})'
                        ).opts(
                            color=ssd_colors.get(int(ssd_num), '#7f7f7f'),
                            alpha=0.6,
                            line_width=0
                        )
                        overlay *= hist
                    
                    # Add vertical line at t=0
                    zero_line = hv.VLine(0).opts(
                        color='red', line_width=2, line_dash='dashed', alpha=0.7
                    )
                    
                    # Combine and configure
                    ylabel = 'Normalized Spike Count' if normalize else 'Spike Count'
                    plot = (overlay * zero_line).opts(
                        opts.Histogram(tools=['hover'], alpha=0.6),
                        opts.NdOverlay(
                            xlabel=f'Time from {alignment_point} (ms)',
                            ylabel=ylabel,
                            title=f"Cell {self.cell_id} - {trial_type} trials - {dir_label} (Success Only)\nHistogram by SSD (bin={bin_size}ms)",
                            width=800, height=300,
                            legend_position='right',
                            show_grid=True,
                            xlim=(epok[0], epok[1])
                        ),
                        opts.VLine(color='red', line_width=2, line_dash='dashed')
                    )
                    
                else:
                    # Combined histogram for all SSDs or GO trials
                    bin_centers, spike_counts, n_trials = self.aggregate_spikes_by_bins(
                        epok=epok,
                        bin_size=bin_size,
                        alignment_point=alignment_point,
                        trial_type=trial_type,
                        direction=direction,
                        success_only=True,
                        normalize=normalize
                    )
                    
                    if bin_centers is None:
                        continue
                    
                    # Create histogram
                    color = ssd_colors['GO'] if trial_type == 'GO' else ssd_colors['combined']
                    # Convert to edges format for hv.Histogram
                    edges = np.concatenate([bin_centers - bin_size/2, [bin_centers[-1] + bin_size/2]])
                    hist_bars = hv.Histogram(
                        (edges, spike_counts),
                        kdims='Time', 
                        vdims='Spike Count' if not normalize else 'Normalized Spike Count'
                    ).opts(
                        color=color,
                        alpha=0.7,
                        tools=['hover'],
                        line_width=0
                    )
                    
                    # Add vertical line at t=0
                    zero_line = hv.VLine(0).opts(
                        color='red', line_width=2, line_dash='dashed', alpha=0.7
                    )
                    
                    # Combine and configure
                    ssd_label = f" (all SSDs, n={n_trials})" if trial_type != 'GO' else f" (n={n_trials})"
                    ylabel = 'Normalized Spike Count' if normalize else 'Spike Count'
                    plot = (hist_bars * zero_line).opts(
                        opts.Histogram(
                            xlabel=f'Time from {alignment_point} (ms)',
                            ylabel=ylabel,
                            title=f"Cell {self.cell_id} - {trial_type} trials - {dir_label} (Success Only){ssd_label}\nHistogram (bin={bin_size}ms)",
                            width=800, height=300,
                            show_grid=True,
                            xlim=(epok[0], epok[1])
                        ),
                        opts.VLine(color='red', line_width=2, line_dash='dashed')
                    )
                
                plots[direction][trial_type] = plot
        
        return plots
    
    def plot_psth_by_type_direction(self, epok=[-200, 500], bin_size=10,
                                    alignment_point='go_cue', 
                                    separate_ssd=False, smooth=True, 
                                    smooth_ker_size=25, delta=False, 
                                    normalize_bins=False):
        """
        Create PSTH (peri-stimulus time histogram) plots for each trial type and direction.
        - Only successful trials (trial_failed = False)
        - Can be combined across all SSDs or separate for each SSD
        - Firing rate with Gaussian smoothing (sigma=bin_size)
        
        Parameters:
        -----------
        epok : list
            Time window [start, end] in ms (default: [-200, 700])
        bin_size : int
            Bin size in ms (default: 10)
        alignment_point : str
            Event to align to: 'go_cue', 'stop_cue', or 'first_relevant_saccade'
        separate_ssd : bool
            If True, create separate PSTHs for each SSD. If False, combine all SSDs.
        smooth : bool
            If True, apply Gaussian smoothing with sigma=bin_size (default: True)
        smooth_ker_size : int
            Kernel size for Gaussian smoothing (default: 25)
        delta: bool
            if True looke center to the mean firing rate
        
        Returns:
        --------
        dict : Nested dictionary {direction: {trial_type: plot}}
        """
        # Color palette for SSD numbers
        ssd_colors = self.SSD_COLORS
        
        plots = {}
        
        for direction in self.directions:
            dir_label = "Right (0°)" if direction == 0 else "Left (180°)"
            plots[direction] = {}
            
            for trial_type in ['GO', 'STOP', 'CONT']:
                # Check if we have data for this combination
                test_data = self.filter_trials(
                    direction=direction,
                    trial_type=trial_type,
                    success_only=True
                )
                
                if len(test_data) == 0:
                    continue
                
                if separate_ssd and trial_type != 'GO':
                    # Create separate curves for each SSD
                    plot_elements = {}
                    
                    ssd_numbers = sorted(test_data['ssd_number'].dropna().unique())
                    for ssd_num in ssd_numbers:
                        bin_centers, firing_rate, n_trials = self.calculate_psth(
                            epok=epok,
                            bin_size=bin_size,
                            alignment_point=alignment_point,
                            trial_type=trial_type,
                            direction=direction,
                            ssd_number=ssd_num,
                            success_only=True,
                            smooth=smooth,
                            smooth_ker_size=smooth_ker_size,
                            delta=delta,
                            normalize_bins=normalize_bins
                        )
                        
                        if bin_centers is None:
                            continue
                        
                        # Create curve for this SSD
                        curve = hv.Curve(
                            (bin_centers, firing_rate),
                            kdims='Time', 
                            vdims='Firing Rate (spikes/s)',
                            label=f'SSD{int(ssd_num)} (n={n_trials})'
                        ).opts(
                            color=ssd_colors.get(ssd_num, '#7f7f7f'),
                            line_width=2, tools=['hover']
                        )
                        plot_elements[f'SSD{int(ssd_num)}'] = curve
                    
                    # Add vertical line at t=0
                    zero_line = hv.VLine(0).opts(
                        color='red', line_width=2, line_dash='dashed', alpha=0.7
                    )
                    
                    # Combine and configure
                    smoothed_label = " (smoothed)" if smooth else ""
                    plot = (hv.NdOverlay(plot_elements) * zero_line).opts(
                        opts.NdOverlay(
                            xlabel=f'Time from {alignment_point} (ms)',
                            ylabel='Firing Rate (spikes/s)',
                            title=f"Cell {self.cell_id} - {trial_type} trials - {dir_label} (Success Only)\nPSTH by SSD (bin={bin_size}ms{smoothed_label})",
                            width=800, height=300,
                            legend_position='top',
                            show_grid=True,
                            xlim=(epok[0], epok[1])
                        ),
                    )
                    
                else:
                    # Combined PSTH for all SSDs or GO trials
                    bin_centers, firing_rate, n_trials = self.calculate_psth(
                        epok=epok,
                        bin_size=bin_size,
                        alignment_point=alignment_point,
                        trial_type=trial_type,
                        direction=direction,
                        success_only=True,
                        smooth=smooth, 
                        smooth_ker_size=smooth_ker_size,
                        delta=delta,
                        normalize_bins=normalize_bins
                    )
                    
                    if bin_centers is None:
                        continue
                    
                    # Create curve
                    color = ssd_colors['GO'] if trial_type == 'GO' else ssd_colors['combined']
                    psth_curve = hv.Curve(
                        (bin_centers, firing_rate),
                        kdims='Time', 
                        vdims='Firing Rate (spikes/s)'
                    ).opts(
                        color=color,
                        line_width=2,
                        tools=['hover']
                    )
                    
                    # Add vertical line at t=0
                    zero_line = hv.VLine(0).opts(
                        color='red', line_width=2, line_dash='dashed', alpha=0.7
                    )
                    
                    # Combine and configure
                    ssd_label = f" (all SSDs, n={n_trials})" if trial_type != 'GO' else f" (n={n_trials})"
                    smoothed_label = " (smoothed)" if smooth else ""
                    plot = (psth_curve * zero_line).opts(
                        opts.Curve(
                            xlabel=f'Time from {alignment_point} (ms)',
                            ylabel='Firing Rate (spikes/s)',
                            title=f"Cell {self.cell_id} - {trial_type} trials - {dir_label} (Success Only){ssd_label}\nPSTH (bin={bin_size}ms{smoothed_label})",
                            width=800, height=300,
                            show_grid=True,
                            xlim=(epok[0], epok[1])
                        ),
                        opts.VLine(color='red', line_width=2, line_dash='dashed')
                    )
                
                plots[direction][trial_type] = plot
        
        return plots


class PopulationAnalyzer:
    """
    Class for analyzing populations of MSN cells.
    Replicates Pani et al. Figure 1B style analysis.
    """
    
    def __init__(self, cell_trial_df):
        """
        Initialize with full cell-trial database.
        
        Parameters:
        -----------
        cell_trial_df : pd.DataFrame
            Full MSN cell-trial database
        """
        self.data = cell_trial_df
        self.cell_ids = sorted(self.data['cell_ID'].unique())
        self.n_cells = len(self.cell_ids)
        self.n_sessions = self.data['trial_session'].nunique()
        
        print(f"PopulationAnalyzer initialized with {self.n_cells} cells")
        print(f"Total trials: {len(self.data)}")
        print(f"Number of sessions: {self.n_sessions}")
    
    def get_cell(self, cell_id):
        """
        Get an MSNCell object for a specific cell.
        
        Parameters:
        -----------
        cell_id : str
            Cell identifier
        
        Returns:
        --------
        MSNCell : Cell object
        """
        cell_data = self.data[self.data['cell_ID'] == cell_id]
        return MSNCell(cell_data)
    
    def plot_figure_1b_style(self, cell_id=None, epok=[-500, 1500], 
                            ssd_to_show=None):
        """
        Create a Figure 1B style raster plot showing:
        - Separate panels for each direction
        - Different trial types (GO, CONT, STOP)
        - Aligned to go_cue with stop_cue markers
        - Color coded by trial outcome
        
        Parameters:
        -----------
        cell_id : str, optional
            Specific cell to analyze. If None, uses first cell.
        epok : list
            Time window [start, end] in ms
        ssd_to_show : int, optional
            Specific SSD number to show. If None, shows all.
        
        Returns:
        --------
        dict : Dictionary with plots for each direction
        """
        if cell_id is None:
            cell_id = self.cell_ids[0]
        
        cell = self.get_cell(cell_id)
        
        # Align spikes to go_cue
        cell.align_spikes_to_event('go_cue')
        
        plots = {}
        
        for direction in cell.directions:
            dir_label = "Right (0°)" if direction == 0 else "Left (180°)"
            
            # Filter by direction
            dir_data = cell.filter_trials(direction=direction)
            if ssd_to_show is not None:
                dir_data = dir_data[
                    (dir_data['type'].isin(['GO'])) | 
                    (dir_data['ssd_number'] == ssd_to_show)
                ]
            
            # Sort trials: GO, then CONT (successful), then STOP (by success/fail, by SSD)
            dir_data = dir_data.sort_values(
                by=['type', 'trial_failed', 'ssd_number'],
                key=lambda x: x.map({
                    'GO': 0, 'CONT': 1, 'STOP': 2, 
                    False: 0, True: 1
                }) if x.name in ['type', 'trial_failed'] else x
            ).reset_index(drop=True)
            
            # Create raster
            overlay = hv.NdOverlay()
            trial_markers = []  # Store trial type boundaries
            
            for i, (idx, row) in enumerate(dir_data.iterrows()):
                spikes = row['spikes_aligned_to_go_cue']
                spikes_in_epok = spikes[(spikes >= epok[0]) & (spikes <= epok[1])]
                
                # Determine color based on trial type and outcome
                if row['type'] == 'GO':
                    color = '#2ca02c'  # Green for GO
                elif row['type'] == 'CONT':
                    color = '#1f77b4' if not row['trial_failed'] else '#ff9896'  # Blue/light red
                else:  # STOP
                    color = '#d62728' if row['trial_failed'] else '#9467bd'  # Red (fail) / Purple (success)
                
                # Plot spikes
                if len(spikes_in_epok) > 0:
                    spike_plot = hv.Spikes(
                        spikes_in_epok, kdims='Time'
                    ).opts(
                        position=i, color=color, spike_length=0.9, line_width=1.5
                    )
                    overlay *= spike_plot
                
                # Mark stop_cue with vertical line segment for STOP/CONT trials
                if row['type'] in ['STOP', 'CONT'] and not pd.isna(row['stop_cue']):
                    stop_time = row['stop_cue'] - row['go_cue']
                    if epok[0] <= stop_time <= epok[1]:
                        stop_marker = hv.Curve(
                            [(stop_time, i-0.4), (stop_time, i+0.4)],
                            kdims='Time', vdims='Trial'
                        ).opts(color='black', line_width=2, alpha=0.6)
                        overlay *= stop_marker
            
            # Add vertical line at t=0 (go_cue)
            go_line = hv.VLine(0).opts(color='red', line_width=2, line_dash='dashed', alpha=0.7)
            
            # Combine and configure
            plot = (overlay * go_line).opts(
                opts.NdOverlay(
                    xlabel='Time from Go Cue (ms)',
                    ylabel='Trial #',
                    title=f"Cell {cell_id} - {dir_label}",
                    width=1000, height=600,
                    show_legend=False,
                    show_grid=True,
                    xlim=(epok[0], epok[1]),
                    ylim=(-1, len(dir_data))
                ),
                opts.VLine(color='red', line_width=2, line_dash='dashed')
            )
            
            plots[direction] = plot
        
        return plots
    
    def plot_psth(self, cell_id, alignment_point='go_cue', bin_size=50, 
                 epok=[-500, 1500], **filter_kwargs):
        """
        Create a PSTH (peri-stimulus time histogram) for a cell.
        
        Parameters:
        -----------
        cell_id : str
            Cell identifier
        alignment_point : str
            Event to align to
        bin_size : int
            Bin size in ms
        epok : list
            Time window [start, end] in ms
        **filter_kwargs : dict
            Trial filtering criteria
        
        Returns:
        --------
        hv.Curve : PSTH plot
        """
        cell = self.get_cell(cell_id)
        
        # Align spikes
        col_name = f'spikes_aligned_to_{alignment_point}'
        if col_name not in cell.data.columns:
            cell.align_spikes_to_event(alignment_point)
        
        # Filter trials
        if filter_kwargs:
            plot_data = cell.filter_trials(**filter_kwargs)
        else:
            plot_data = cell.data
        
        # Create bins
        bins = np.arange(epok[0], epok[1] + bin_size, bin_size)
        bin_centers = bins[:-1] + bin_size / 2
        
        # Count spikes in each bin
        spike_counts = np.zeros(len(bins) - 1)
        for _, row in plot_data.iterrows():
            spikes = row[col_name]
            counts, _ = np.histogram(spikes, bins=bins)
            spike_counts += counts
        
        # Convert to firing rate (spikes/sec)
        firing_rate = (spike_counts / len(plot_data)) / (bin_size / 1000)
        
        # Create plot
        psth_curve = hv.Curve(
            (bin_centers, firing_rate), kdims='Time', vdims='Firing Rate'
        ).opts(
            xlabel=f'Time from {alignment_point} (ms)',
            ylabel='Firing Rate (spikes/s)',
            title=f"PSTH - Cell {cell_id}",
            width=1000, height=400,
            color='#1f77b4',
            line_width=2,
            show_grid=True
        )
        
        # Add vertical line at t=0
        zero_line = hv.VLine(0).opts(
            color='red', line_width=2, line_dash='dashed', alpha=0.7
        )
        
        return (psth_curve * zero_line).opts(
            opts.VLine(color='red', line_width=2, line_dash='dashed')
        )


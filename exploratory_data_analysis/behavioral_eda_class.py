"""
Behavioral Exploratory Data Analysis Class

This class provides methods to create plots and analyze behavioral data from 
stop signal task experiments. It separates data processing methods from plotting 
methods for better organization and reusability.

Author: Generated from behavioral_eda.ipynb notebook
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt
import holoviews as hv
from holoviews import opts
import hvplot.pandas
import panel as pn
import ast
from typing import Optional, Dict, List, Tuple


class BehavioralEDA:
    """
    A class for performing exploratory data analysis on behavioral data from 
    stop signal task experiments.
    
    This class loads behavioral data from pickle files and provides methods to:
    1. Generate DataFrames for various analyses
    2. Create visualizations of behavioral performance
    3. Compute reaction time measures
    4. Analyze signal delay effects
    """
    
    def __init__(self, file_path: str):
        """
        Initialize the BehavioralEDA class with data from a pickle file.
        
        Parameters:
        -----------
        file_path : str
            Path to the pickle file containing behavioral data
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Load the data
        self.df = pd.read_pickle(self.file_path)
        self.monkey = self._extract_monkey_name()
        
        # Initialize holoviews settings
        self._setup_holoviews()
        
        # Check if reaction time processing is needed - need all required columns
        required_rt_columns = ['computed_rt', 'rt_type', 'signal_delay']
        has_all_rt_columns = all(col in self.df.columns for col in required_rt_columns)
        
        self._rt_processed = has_all_rt_columns
        
        print(f"Loaded data for {self.monkey}")
        print(f"Total trials: {len(self.df):,}")
        print(f"Date range: {self.df['trial_session'].str[:8].min()} to {self.df['trial_session'].str[:8].max()}")
        
        has_reaction_time = 'reaction_time' in self.df.columns
        if self._rt_processed:
            print("✓ All RT processing columns already available")
        elif has_reaction_time:
            print("✓ Reaction time data available, will add derived columns as needed")
        else:
            print("⚠ Reaction time data needs to be processed from saccade data")

    def _extract_monkey_name(self) -> str:
        """Extract monkey name from file path"""
        filename = self.file_path.name
        if 'fiona' in filename.lower():
            return 'fiona'
        elif 'yasmin' in filename.lower():
            return 'yasmin'
        else:
            return 'unknown'
    
    def _setup_holoviews(self):
        """Setup holoviews configuration"""
        try:
            hv.extension('bokeh')
            pn.extension('bokeh')
        except:
            pass  # Skip if bokeh not available
        
        self.font_dict = {'title': 16, 'labels': 14, 'ticks': 12, 'legend': 12}
        
        # Set default options
        hv.opts.defaults(
            hv.opts.Curve(width=600, height=400, tools=['hover'], fontsize=self.font_dict),
            hv.opts.Scatter(width=600, height=400, size=8, tools=['hover'], fontsize=self.font_dict),
            hv.opts.Histogram(width=600, height=400, fontsize=self.font_dict),
            hv.opts.Bars(width=600, height=400, fontsize=self.font_dict),
        )

    # ==========================================
    # DATA PROCESSING METHODS
    # ==========================================
    
    def get_basic_summary(self) -> Dict:
        """
        Get basic overview of the behavioral data.
        
        Returns:
        --------
        dict : Dictionary containing basic statistics
        """
        summary = {
            'total_trials': len(self.df),
            'trial_types': self.df['type'].value_counts().to_dict(),
            'directions': self.df['direction'].value_counts().to_dict(),
            'overall_success_rate': (1 - self.df['trial_failed'].mean()) * 100,
            'experimental_sets': self.df['set'].value_counts().to_dict()
        }
        return summary
    
    def get_trial_summary_data(self) -> pd.DataFrame:
        """
        Create summary statistics for trial types and outcomes.
        
        Returns:
        --------
        pd.DataFrame : DataFrame with trial counts by type and outcome
        """
        trial_summary = self.df.groupby(['type', 'trial_failed']).size().reset_index(name='count')
        trial_summary['outcome'] = trial_summary['trial_failed'].map({False: 'Success', True: 'Failed'})
        return trial_summary
    
    def get_success_rates_data(self) -> pd.DataFrame:
        """
        Calculate success rates by trial type.
        
        Returns:
        --------
        pd.DataFrame : DataFrame with success rates by trial type
        """
        success_rates = self.df.groupby('type').agg({
            'trial_failed': ['count', 'sum', 'mean']
        }).round(3)
        success_rates.columns = ['total_trials', 'failed_trials', 'failure_rate']
        success_rates['success_rate'] = (1 - success_rates['failure_rate']) * 100
        success_rates['failure_rate'] *= 100
        return success_rates.reset_index()
    
    def get_trial_percentage_data(self) -> pd.DataFrame:
        """
        Get trial outcomes as percentages.
        
        Returns:
        --------
        pd.DataFrame : Melted DataFrame with percentage outcomes
        """
        trial_pct = self.df.groupby('type').apply(
            lambda x: pd.Series({
                'Success': (1 - x['trial_failed'].mean()) * 100,
                'Failed': x['trial_failed'].mean() * 100
            })
        ).reset_index()
        
        trial_pct_melted = trial_pct.melt(id_vars='type', var_name='outcome', value_name='percentage')
        return trial_pct_melted
    
    def get_direction_summary_data(self) -> pd.DataFrame:
        """
        Get trial summary by direction and type.
        
        Returns:
        --------
        pd.DataFrame : DataFrame with direction and trial type analysis
        """
        direction_summary = self.df.groupby(['type', 'direction', 'trial_failed']).size().reset_index(name='count')
        direction_summary['outcome'] = direction_summary['trial_failed'].map({False: 'Success', True: 'Failed'})
        return direction_summary
    
    def get_direction_success_rates(self) -> pd.DataFrame:
        """
        Get success rates by type and direction.
        
        Returns:
        --------
        pd.DataFrame : DataFrame with success rates by type and direction
        """
        dir_success = self.df.groupby(['type', 'direction']).agg({
            'trial_failed': ['count', 'mean']
        }).round(3)
        dir_success.columns = ['total_trials', 'failure_rate']
        dir_success['success_rate'] = (1 - dir_success['failure_rate']) * 100
        return dir_success.reset_index()
    
    def get_signal_delay_performance_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get performance data by signal delay for STOP and CONT trials.
        
        Returns:
        --------
        tuple : (stop_performance, cont_performance) DataFrames
        """
        # Ensure RT is processed
        if not self._rt_processed:
            self._process_reaction_times()
        
        signal_perf_data = self.df[
            self.df['type'].isin(['STOP', 'CONT']) & 
            self.df['signal_delay'].notna()
        ]
        
        # STOP trial performance
        stop_performance = signal_perf_data[signal_perf_data['type'] == 'STOP'].groupby('ssd_number').agg({
            'trial_failed': ['count', 'sum', 'mean']
        }).round(3)
        stop_performance.columns = ['total_trials', 'failed_trials', 'error_rate']
        stop_performance['error_percentage'] = stop_performance['error_rate'] * 100
        stop_performance = stop_performance.reset_index()
        
        # CONT trial performance
        cont_performance = signal_perf_data[signal_perf_data['type'] == 'CONT'].groupby('ssd_number').agg({
            'trial_failed': ['count', 'sum', 'mean']
        }).round(3)
        cont_performance.columns = ['total_trials', 'failed_trials', 'failure_rate']
        cont_performance['correct_percentage'] = (1 - cont_performance['failure_rate']) * 100
        cont_performance = cont_performance.reset_index()
        
        # Add SSD lengths
        ssd_len_col = []
        for g, gdf in self.df.groupby('ssd_number'):
            # print(f"SSD Number: {g}")
            ssd_len_col.append(gdf['ssd_len'].value_counts().idxmax())
            print(g, ": ",gdf['ssd_len'].value_counts().idxmax())
        ssd_len_col.sort()

        stop_performance['ssd_len'] = ssd_len_col
        cont_performance['ssd_len'] = ssd_len_col
        
        return stop_performance, cont_performance
    
    def get_rt_scatter_data(self) -> pd.DataFrame:
        """
        Get data for RT scatter plots.
        
        Returns:
        --------
        pd.DataFrame : DataFrame with mean RTs by session and type
        """
        if not self._rt_processed:
            self._process_reaction_times()
        
        rt_scatter_data = self.df[
            self.df['computed_rt'].notna() & 
            self.df['rt_type'].isin(['GO_RT', 'Continue_RT', 'Error_Stop_RT'])
        ]
        
        scatter_df = rt_scatter_data.groupby(['rt_type', 'trial_session']).agg({
            'computed_rt': ['mean']
        })
        scatter_df.columns = ['mean_rt']
        scatter_df = scatter_df.reset_index()
        
        return scatter_df
    
    def get_rt_distribution_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get data for RT distribution plots.
        
        Returns:
        --------
        tuple : (cont_df, stop_df) with RT distribution data
        """
        if not self._rt_processed:
            self._process_reaction_times()
        
        failed_stop_trials = self.df[
            self.df['type'].isin(['STOP']) & 
            self.df['trial_failed'].isin([True])
        ]
        successful_cont_trials = self.df[
            self.df['type'].isin(['CONT']) & 
            self.df['trial_failed'].isin([False])
        ]
        
        tot_cont_trials = len(self.df[self.df['type'].isin(['CONT'])])
        tot_stop_trials = len(self.df[self.df['type'].isin(['STOP'])])
        
        def frame_it(df, normalizer, ssd_prefix):
            tmp_df = df.groupby(['computed_rt', 'ssd_number']).size()
            tmp_df = tmp_df.reset_index().rename(columns={
                'computed_rt': 'Reaction Time', 
                'ssd_number': 'SSD Number', 
                0: 'Count'
            })
            tmp_df['percentage'] = (tmp_df['Count'] / normalizer) * 100
            tmp_df['SSD Number'] = tmp_df.apply(
                lambda row: f'{ssd_prefix}{int(row["SSD Number"])}', axis=1
            )
            tmp_df = tmp_df.groupby(['Reaction Time', 'SSD Number'])['percentage'].sum().reset_index()
            
            # Group into 20ms bins
            tmp_df['Reaction Time Bin'] = (tmp_df['Reaction Time'] // 20) * 20
            tmp_df = tmp_df.groupby(['Reaction Time Bin', 'SSD Number'], as_index=False)['percentage'].sum()
            return tmp_df
        
        cont_df = frame_it(successful_cont_trials, tot_cont_trials, 'CSD')
        stop_df = frame_it(failed_stop_trials, tot_stop_trials, 'SSD')
        
        return cont_df, stop_df

    # ==========================================
    # UTILITY METHODS
    # ==========================================
    
    def _extract_saccade_start(self, saccade_array) -> float:
        """
        Extract saccade start time from first_relevant_saccade array.
        
        Parameters:
        -----------
        saccade_array : various types
            Saccade data in various formats
            
        Returns:
        --------
        float : Saccade start time or NaN
        """
        # Handle None values
        if saccade_array is None:
            return np.nan
        
        # Handle numpy arrays directly
        if isinstance(saccade_array, np.ndarray):
            if len(saccade_array) >= 2:
                return float(saccade_array[0])
            else:
                return np.nan
        
        # Handle lists and tuples
        if isinstance(saccade_array, (list, tuple)):
            if len(saccade_array) >= 2:
                return float(saccade_array[0])
            else:
                return np.nan
        
        # Handle string representations
        if isinstance(saccade_array, str):
            saccade_array = saccade_array.strip()
            if saccade_array == '' or saccade_array == 'nan':
                return np.nan
            try:
                # Try to evaluate string representation of array
                saccade_data = ast.literal_eval(saccade_array)
                if isinstance(saccade_data, (list, tuple)) and len(saccade_data) >= 2:
                    return float(saccade_data[0])
                else:
                    return np.nan
            except:
                return np.nan
        
        # Handle pandas NA/NaN values
        try:
            if pd.isna(saccade_array):
                return np.nan
        except:
            pass
        
        # If we get here, we couldn't parse it
        return np.nan
    
    def _check_flag_consistency(self, row) -> bool:
        """
        Check flag consistency for trial success/failure.
        
        Parameters:
        -----------
        row : pd.Series
            DataFrame row
            
        Returns:
        --------
        bool : Whether trial failed
        """
        bit = 2 if (row['type'] != 'STOP') else 11
        return not bool(row['flags'] & (1 << bit))
    
    def _process_reaction_times(self):
        """Process reaction times from saccade data if not already present."""
        if self._rt_processed:
            return
            
        print("Processing reaction times and adding to original DataFrame...")
        
        # Check if reaction_time column exists, if so use it
        if 'reaction_time' in self.df.columns:
            print("✓ Using existing reaction_time column")
            self.df['computed_rt'] = self.df['reaction_time']
        else:
            print("Computing reaction times from saccade data...")
            # Extract saccade start times only if needed
            self.df['saccade_start'] = self.df['first_relevant_saccade'].apply(
                self._extract_saccade_start
            )
            
            # Initialize computed_rt column
            self.df['computed_rt'] = np.nan
            
            # Calculate RTs efficiently using vectorized operations where possible
            valid_saccade_mask = self.df['saccade_start'].notna() & self.df['go_cue'].notna()
            
            # Calculate RT for valid entries
            rt_values = self.df.loc[valid_saccade_mask, 'saccade_start'] - self.df.loc[valid_saccade_mask, 'go_cue']
            
            # Only keep positive RTs
            positive_rt_mask = rt_values > 0
            self.df.loc[valid_saccade_mask, 'computed_rt'] = np.where(
                positive_rt_mask, rt_values, np.nan
            )
        
        # Initialize RT type column
        if 'rt_type' not in self.df.columns:
            self.df['rt_type'] = ''
            
            # Classify RT types using vectorized operations
            go_success_mask = (self.df['type'] == 'GO') & (~self.df['trial_failed'])
            stop_error_mask = (self.df['type'] == 'STOP') & (self.df['trial_failed'])
            cont_mask = (self.df['type'] == 'CONT')
            
            self.df.loc[go_success_mask, 'rt_type'] = 'GO_RT'
            self.df.loc[stop_error_mask, 'rt_type'] = 'Error_Stop_RT'
            self.df.loc[cont_mask, 'rt_type'] = 'Continue_RT'
            
            # Set remaining as 'Other'
            other_mask = ~(go_success_mask | stop_error_mask | cont_mask)
            self.df.loc[other_mask, 'rt_type'] = 'Other'
        
        # Calculate signal delays if not already present
        if 'signal_delay' not in self.df.columns:
            self.df['signal_delay'] = np.nan
            
            # Calculate signal delays for STOP and CONT trials
            signal_trial_mask = (
                self.df['type'].isin(['STOP', 'CONT']) & 
                self.df['stop_cue'].notna() & 
                self.df['go_cue'].notna()
            )
            
            signal_delays = self.df.loc[signal_trial_mask, 'stop_cue'] - self.df.loc[signal_trial_mask, 'go_cue']
            
            # Only keep positive signal delays
            positive_signal_mask = signal_delays > 0
            self.df.loc[signal_trial_mask, 'signal_delay'] = np.where(
                positive_signal_mask, signal_delays, np.nan
            )
        
        self._rt_processed = True
        print("✓ Reaction time processing completed and added to original DataFrame")

    # ==========================================
    # PLOTTING METHODS
    # ==========================================
    
    def plot_trial_distribution(self) -> hv.element.chart.Bars:
        """
        Create a bar plot of trial distribution by type and outcome.
        
        Returns:
        --------
        holoviews plot : Stacked bar chart
        """
        trial_summary = self.get_trial_summary_data()
        
        plot = trial_summary.hvplot.bar(
            x='type', y='count', by='outcome',
            stacked=True,
            title=f'{self.monkey.title()} - Trial Distribution by Type and Outcome',
            xlabel='Trial Type',
            ylabel='Number of Trials',
            width=600, height=400,
            color=['#2E8B57', '#CD5C5C'],  # Green for success, red for failed
            legend='top_right'
        )
        
        plot.opts(fontsize=self.font_dict)
        return plot
    
    def plot_success_rates_percentage(self) -> hv.element.chart.Bars:
        """
        Create a bar plot of success rates as percentages.
        
        Returns:
        --------
        holoviews plot : Stacked percentage bar chart
        """
        trial_pct_melted = self.get_trial_percentage_data()
        
        plot = trial_pct_melted.hvplot.bar(
            x='type', y='percentage', by='outcome',
            stacked=True,
            title=f'{self.monkey.title()} - Success Rate by Trial Type (%)',
            xlabel='Trial Type',
            ylabel='Percentage of Trials',
            width=600, height=400,
            color=['#2E8B57', '#CD5C5C'],
            legend='top',
            ylim=(0, 100)
        )
        
        return plot
    
    def plot_direction_analysis(self) -> hv.element.chart.Bars:
        """
        Create a bar plot of successful trials by type and direction.
        
        Returns:
        --------
        holoviews plot : Bar chart by direction
        """
        direction_summary = self.get_direction_summary_data()
        
        plot = direction_summary[direction_summary['outcome'] == 'Success'].hvplot.bar(
            x='type', y='count', by='direction',
            title=f'{self.monkey.title()} - Successful Trials by Type and Direction',
            xlabel='Trial Type',
            ylabel='Number of Successful Trials',
            width=600, height=400,
            legend='top_right'
        )
        
        return plot
    
    def plot_trial_length_distribution(self) -> hv.core.overlay.NdOverlay:
        """
        Create histogram of trial length distribution by type.
        
        Returns:
        --------
        holoviews plot : Histogram overlay
        """
        plot = self.df.hvplot.hist(
            y='trial_length', by='type',
            bins=50, alpha=0.7,
            title=f'{self.monkey.title()} - Trial Length Distribution by Type',
            xlabel='Trial Length (ms)',
            ylabel='Frequency',
            width=800, height=400,
            legend='top_right'
        )
        
        return plot
    
    def plot_go_cue_timing(self) -> hv.core.overlay.NdOverlay:
        """
        Create histogram of go cue timing distribution by type.
        
        Returns:
        --------
        holoviews plot : Histogram overlay
        """
        plot = self.df.hvplot.hist(
            y='go_cue', by='type',
            bins=50, alpha=0.7,
            title=f'{self.monkey.title()} - Go Cue Timing Distribution by Type',
            xlabel='Go Cue Time (ms)',
            ylabel='Frequency',
            width=800, height=400,
            legend='top_right'
        )
        
        return plot
    
    def plot_signal_delay_performance(self) -> hv.core.overlay.Overlay:
        """
        Create line plot replicating Figure 1b - Stop and Continue performance.
        
        Returns:
        --------
        holoviews plot : Line plot overlay
        """
        stop_performance, cont_performance = self.get_signal_delay_performance_data()
        
        stop_plot = stop_performance.hvplot.line(
            x='ssd_len', y='error_percentage',
            color='red', line_width=3,
            label=f'Error stop ({self.monkey.title()})',
            markers=True, marker_size=8
        ) * stop_performance.hvplot.scatter(
            x='ssd_len', y='error_percentage',
            color='red', size=150, alpha=0.5,
            marker='triangle'
        )
        
        cont_plot = cont_performance.hvplot.line(
            x='ssd_len', y='correct_percentage',
            color='blue', line_width=3,
            label=f'Correct continue ({self.monkey.title()})',
            markers=True, marker_size=8, #line_dash='dashed'
        ) * cont_performance.hvplot.scatter(
            x='ssd_len', y='correct_percentage',
            color='blue', size=100, alpha=0.5,
            marker='o'
        )
        
        # Combine plots
        plot_fig1b = (stop_plot * cont_plot).opts(
            title=f'{self.monkey.title()} - Stop and Continue Performance (Figure 1b)',
            xlabel='Stop/continue signal delay (ms)',
            ylabel='Percentage of saccades',
            width=700, height=400,
            ylim=(0, 100),
            legend_position='top',
            show_grid=True,
            fontsize=self.font_dict,
        )
        
        return plot_fig1b
    
    def plot_rt_scatter(self) -> hv.core.overlay.Overlay:
        """
        Create scatter plot of session mean RTs.
        
        Returns:
        --------
        holoviews plot : Scatter plot with diagonal reference
        """
        scatter_df = self.get_rt_scatter_data()
        
        # Filter for GO_RT vs Continue_RT
        scatter_data = scatter_df[scatter_df['rt_type'].isin(['GO_RT', 'Continue_RT'])]
        scatter_pivot = scatter_data.pivot(
            index='trial_session', columns='rt_type', values='mean_rt'
        ).reset_index()
        
        scatter_plot = scatter_pivot.hvplot.scatter(
            x='GO_RT', y='Continue_RT',
            xlabel='Mean RT (GO_RT)',
            width=700, height=400,
            alpha=0.7,
            color='purple',
            label=f'Continue continue RT {self.monkey}',
            legend=True
        )
        
        # Filter for GO_RT vs Error_Stop_RT
        scatter_data_error = scatter_df[scatter_df['rt_type'].isin(['GO_RT', 'Error_Stop_RT'])]
        scatter_pivot_error = scatter_data_error.pivot(
            index='trial_session', columns='rt_type', values='mean_rt'
        ).reset_index()
        
        scatter_plot_error = scatter_pivot_error.hvplot.scatter(
            x='GO_RT', y='Error_Stop_RT',
            title=f'{self.monkey} Session mean RT',
            ylabel='Mean RT (Error_Stop_RT / Continue_RT)',
            width=700, height=400,
            alpha=0.7,
            color='green',
            legend=True,
            label=f'Error stop RT {self.monkey}'
        )
        
        # Add diagonal line
        min_rt = min(scatter_df['mean_rt'])
        max_rt = max(scatter_df['mean_rt'])
        diagonal_line = hv.Curve([(min_rt, min_rt), (max_rt, max_rt)]).opts(
            color='black', line_dash='solid'
        )
        
        combined_plot = (scatter_plot * scatter_plot_error * diagonal_line).opts(
            width=600, height=400, legend_position='top_left', show_legend=True
        )
        
        return combined_plot
    
    def plot_rt_distributions(self) -> hv.core.overlay.Overlay:
        """
        Create line plot of RT distributions (Figure 1d).
        
        Returns:
        --------
        holoviews plot : Line plot overlay
        """
        cont_df, stop_df = self.get_rt_distribution_data()
        
        # Create histogram for continue trials
        cont_plot = cont_df.hvplot.line(
            x='Reaction Time Bin', y='percentage', by='SSD Number',
            title=f'{self.monkey.title()} - Continue and error stop RT',
            xlabel='Reaction time (ms)',
            ylabel='Percentage of total trials',
            width=800, height=400,
            line_dash='dashed',
            line_width=3,
        )
        
        # Create histogram for error stop trials
        stop_plot = stop_df.hvplot.line(
            x='Reaction Time Bin', y='percentage', by='SSD Number',
            line_width=3,
            xlim=(0, 600)
        )
        
        combined_plot = (cont_plot * stop_plot).opts(legend_position='top_right')
        return combined_plot

    # ==========================================
    # SUMMARY METHODS
    # ==========================================
    
    def print_summary_stats(self):
        """Print comprehensive summary statistics."""
        print("=== BEHAVIORAL DATA OVERVIEW ===")
        
        summary = self.get_basic_summary()
        print(f"\nTrial Types:")
        for trial_type, count in summary['trial_types'].items():
            print(f"  {trial_type}: {count:,}")
        
        print(f"\nDirections:")
        for direction, count in summary['directions'].items():
            print(f"  {direction}: {count:,}")
        
        print(f"\nOverall Success Rate: {summary['overall_success_rate']:.1f}%")
        
        success_rates = self.get_success_rates_data()
        print("\nSuccess rates by trial type:")
        print(success_rates[['type', 'success_rate', 'total_trials']])
        
        if not self._rt_processed:
            self._process_reaction_times()
            
        print("\n=== RT STATISTICS BY TYPE ===")
        if self.df['computed_rt'].notna().sum() > 0:
            rt_stats = self.df[self.df['computed_rt'].notna()].groupby('rt_type')['computed_rt'].agg([
                'count', 'mean', 'std', 'min', 'max'
            ]).round(1)
            print(rt_stats)
        else:
            print("No valid RTs computed")
"""
Neural Population Analysis Panel App

This app demonstrates neural population analysis using Non-homogeneous Poisson Processes.
It simulates a neuron that spikes at different rates during different time periods:
- Baseline rate (lambda_1) from -200 to 0 ms
- Enhanced rate (lambda_2) from 0 to 200 ms  
- Return to baseline from 200 to 500 ms

The spikes follow a Poisson distribution as described in:
https://www.randomservices.org/random/poisson/Nonhomogeneous.html
"""

import numpy as np
import panel as pn
import param
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import io
import base64

# Enable Panel extensions
pn.extension('matplotlib')

class NonHomogeneousPoissonProcess:
    """
    Implementation of Non-homogeneous Poisson Process for neural spike simulation.
    
    Based on the theory from: https://www.randomservices.org/random/poisson/Nonhomogeneous.html
    """
    
    def __init__(self, time_start=-200, time_end=500):
        """
        Initialize the process with time boundaries in milliseconds.
        
        Args:
            time_start: Start time in ms (default: -200)
            time_end: End time in ms (default: 500)
        """
        self.time_start = time_start
        self.time_end = time_end
        
    def rate_function(self, t, lambda_1, lambda_2):
        """
        Define the time-varying rate function λ(t).
        
        Args:
            t: Time array in ms
            lambda_1: Baseline firing rate (spikes/ms)
            lambda_2: Enhanced firing rate (spikes/ms)
            
        Returns:
            Rate function values at times t
        """
        # Ensure t is numpy array for proper indexing
        t = np.asarray(t)
        rate = np.zeros_like(t, dtype=float)
        
        # Baseline period: -200 to 0 ms
        baseline_mask = (t >= -200) & (t < 0)
        rate[baseline_mask] = lambda_1
        
        # Enhanced period: 0 to 200 ms
        enhanced_mask = (t >= 0) & (t < 200)
        rate[enhanced_mask] = lambda_2
        
        # Return to baseline: 200 to 500 ms
        return_mask = (t >= 200) & (t <= 500)
        rate[return_mask] = lambda_1
        
        return rate
    
    def generate_spikes(self, lambda_1, lambda_2, n_trials=1, dt=0.1):
        """
        Generate spike trains using the thinning algorithm for non-homogeneous Poisson processes.
        
        Args:
            lambda_1: Baseline firing rate (spikes/ms)
            lambda_2: Enhanced firing rate (spikes/ms)
            n_trials: Number of spike trains to generate
            dt: Time resolution in ms
            
        Returns:
            List of spike time arrays for each trial
        """
        # Find maximum rate for thinning algorithm
        max_rate = max(lambda_1, lambda_2)
        
        spike_trains = []
        
        for trial in range(n_trials):
            # Generate candidate spike times using homogeneous Poisson with max rate
            total_time = self.time_end - self.time_start
            expected_spikes = int(max_rate * total_time * 2)  # Overestimate
            
            # Generate inter-spike intervals
            intervals = np.random.exponential(1.0 / max_rate, expected_spikes)
            
            # Convert to spike times
            candidate_times = np.cumsum(intervals) + self.time_start
            
            # Only keep times within our window
            candidate_times = candidate_times[candidate_times <= self.time_end]
            
            # Apply thinning: accept each spike with probability λ(t)/λ_max
            accepted_spikes = []
            
            for spike_time in candidate_times:
                actual_rate = self.rate_function(np.array([spike_time]), lambda_1, lambda_2)[0]
                accept_prob = actual_rate / max_rate
                
                if np.random.random() < accept_prob:
                    accepted_spikes.append(spike_time)
            
            spike_trains.append(np.array(accepted_spikes))
        
        return spike_trains


class NeuralAnalysisApp(param.Parameterized):
    """
    Panel application for neural population analysis demonstration.
    """
    
    # Parameters that can be controlled by user
    lambda_1 = param.Number(default=0.005, bounds=(0.001, 0.02), step=0.001,
                           doc="Baseline firing rate (spikes/ms)")
    
    lambda_2 = param.Number(default=0.015, bounds=(0.001, 0.05), step=0.001,
                           doc="Enhanced firing rate (spikes/ms)")
    
    n_trials = param.Integer(default=20, bounds=(1, 100), step=1,
                            doc="Number of spike trains to simulate")
    
    def __init__(self, **params):
        super().__init__(**params)
        self.poisson_process = NonHomogeneousPoissonProcess()
        
    def create_plots(self):
        """Create the main visualization plots."""
        
        # Generate spike trains
        spike_trains = self.poisson_process.generate_spikes(
            self.lambda_1, self.lambda_2, self.n_trials
        )
        
        # Create figure with subplots
        fig = Figure(figsize=(12, 8))
        
        # Plot 1: Rate function
        ax1 = fig.add_subplot(3, 1, 1)
        time_axis = np.linspace(-200, 500, 1000)
        rate_values = self.poisson_process.rate_function(time_axis, self.lambda_1, self.lambda_2)
        
        ax1.plot(time_axis, rate_values * 1000, 'b-', linewidth=2, label='Rate λ(t)')
        ax1.axvline(x=0, color='r', linestyle='--', alpha=0.7, label='Stimulus onset')
        ax1.axvline(x=200, color='r', linestyle='--', alpha=0.7, label='Stimulus offset')
        
        # Add colored background regions
        ax1.axvspan(-200, 0, alpha=0.2, color='gray', label='Baseline')
        ax1.axvspan(0, 200, alpha=0.2, color='orange', label='Enhanced')
        ax1.axvspan(200, 500, alpha=0.2, color='gray')
        
        ax1.set_ylabel('Firing Rate\n(spikes/sec)')
        ax1.set_title('Non-homogeneous Poisson Process: Neural Firing Rate')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Spike raster
        ax2 = fig.add_subplot(3, 1, 2)
        
        for i, spikes in enumerate(spike_trains):
            if len(spikes) > 0:
                ax2.scatter(spikes, [i] * len(spikes), s=1, c='black', alpha=0.7)
        
        ax2.axvline(x=0, color='r', linestyle='--', alpha=0.7)
        ax2.axvline(x=200, color='r', linestyle='--', alpha=0.7)
        ax2.axvspan(-200, 0, alpha=0.1, color='gray')
        ax2.axvspan(0, 200, alpha=0.1, color='orange')
        ax2.axvspan(200, 500, alpha=0.1, color='gray')
        
        ax2.set_ylabel('Trial Number')
        ax2.set_title(f'Spike Raster Plot ({self.n_trials} trials)')
        ax2.set_ylim(-0.5, self.n_trials - 0.5)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: PSTH (Peri-Stimulus Time Histogram)
        ax3 = fig.add_subplot(3, 1, 3)
        
        # Combine all spikes
        all_spikes = np.concatenate([spikes for spikes in spike_trains if len(spikes) > 0])
        
        if len(all_spikes) > 0:
            bins = np.linspace(-200, 500, 50)
            counts, bin_edges = np.histogram(all_spikes, bins=bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_width = bin_edges[1] - bin_edges[0]
            
            # Convert to firing rate (spikes/sec)
            firing_rate = counts / (self.n_trials * bin_width / 1000)
            
            ax3.bar(bin_centers, firing_rate, width=bin_width*0.8, alpha=0.7, color='navy')
        
        ax3.axvline(x=0, color='r', linestyle='--', alpha=0.7)
        ax3.axvline(x=200, color='r', linestyle='--', alpha=0.7)
        ax3.axvspan(-200, 0, alpha=0.1, color='gray')
        ax3.axvspan(0, 200, alpha=0.1, color='orange')
        ax3.axvspan(200, 500, alpha=0.1, color='gray')
        
        ax3.set_xlabel('Time (ms)')
        ax3.set_ylabel('Firing Rate\n(spikes/sec)')
        ax3.set_title('Peri-Stimulus Time Histogram (PSTH)')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig
    
    def get_stats_text(self):
        """Generate statistics text for display."""
        spike_trains = self.poisson_process.generate_spikes(
            self.lambda_1, self.lambda_2, self.n_trials
        )
        
        # Calculate statistics
        baseline_spikes = []
        enhanced_spikes = []
        return_spikes = []
        
        for spikes in spike_trains:
            baseline_spikes.extend(spikes[(spikes >= -200) & (spikes < 0)])
            enhanced_spikes.extend(spikes[(spikes >= 0) & (spikes < 200)])
            return_spikes.extend(spikes[(spikes >= 200) & (spikes <= 500)])
        
        # Calculate average firing rates
        baseline_rate = len(baseline_spikes) / (self.n_trials * 0.2)  # 200ms period
        enhanced_rate = len(enhanced_spikes) / (self.n_trials * 0.2)  # 200ms period
        return_rate = len(return_spikes) / (self.n_trials * 0.3)     # 300ms period
        
        stats_text = f"""
        **Simulation Statistics:**
        
        **Parameters:**
        - Baseline rate (λ₁): {self.lambda_1:.3f} spikes/ms ({self.lambda_1*1000:.1f} spikes/sec)
        - Enhanced rate (λ₂): {self.lambda_2:.3f} spikes/ms ({self.lambda_2*1000:.1f} spikes/sec)
        - Number of trials: {self.n_trials}
        
        **Observed Rates:**
        - Baseline period (-200 to 0 ms): {baseline_rate:.1f} spikes/sec
        - Enhanced period (0 to 200 ms): {enhanced_rate:.1f} spikes/sec
        - Return period (200 to 500 ms): {return_rate:.1f} spikes/sec
        
        **Analysis:**
        - Fold change: {enhanced_rate/baseline_rate if baseline_rate > 0 else 'N/A'}
        - Total spikes: {sum(len(spikes) for spikes in spike_trains)}
        """
        
        return stats_text


def create_app():
    """Create the Panel application."""
    
    # Create the neural analysis object
    neural_app = NeuralAnalysisApp()
    
    # Create parameter controls
    controls = pn.Param(
        neural_app,
        parameters=['lambda_1', 'lambda_2', 'n_trials'],
        widgets={
            'lambda_1': pn.widgets.FloatSlider,
            'lambda_2': pn.widgets.FloatSlider,
            'n_trials': pn.widgets.IntSlider
        },
        show_name=False,
        sizing_mode='stretch_width'
    )
    
    # Create dynamic plot function
    @pn.depends(neural_app.param.lambda_1, neural_app.param.lambda_2, neural_app.param.n_trials)
    def get_plot():
        return pn.pane.Matplotlib(neural_app.create_plots(), sizing_mode='stretch_width')
    
    # Create dynamic stats function  
    @pn.depends(neural_app.param.lambda_1, neural_app.param.lambda_2, neural_app.param.n_trials)
    def get_stats():
        return pn.pane.Markdown(neural_app.get_stats_text(), sizing_mode='stretch_width')
    
    # Create information panel
    info_text = """
    # Neural Population Analysis Demo
    
    This application demonstrates **Non-homogeneous Poisson Processes** for modeling neural spike trains.
    
    ## How it works:
    
    1. **Rate Function**: The neuron fires at different rates during three periods:
       - **Baseline** (-200 to 0 ms): Rate λ₁ 
       - **Enhanced** (0 to 200 ms): Rate λ₂ (stimulus response)
       - **Return** (200 to 500 ms): Back to baseline λ₁
    
    2. **Spike Generation**: Uses the thinning algorithm to generate spikes following the time-varying rate function.
    
    3. **Visualization**: Shows the rate function, spike raster plot, and PSTH (Peri-Stimulus Time Histogram).
    
    ## Controls:
    Use the sliders below to adjust the parameters and observe how they affect neural activity patterns.
    
    ---
    """
    
    # Layout the application
    app = pn.template.MaterialTemplate(
        title="Neural Population Analysis - Non-homogeneous Poisson Process",
        sidebar=[
            pn.pane.Markdown(info_text),
            "## Parameters",
            controls,
            "## Statistics", 
            get_stats
        ],
        main=[get_plot],
        header_background='#2596be',
    )
    
    return app


if __name__ == "__main__":
    # For running the app
    app = create_app()
    app.servable()
    
    # Show the app
    app.show(port=5007, open=False)
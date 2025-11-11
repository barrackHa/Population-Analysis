"""
Test suite for cell_analysis module using pytest

Creates synthetic test data to validate Cell and PopulationAnalyzer functionality.

Run tests with: pytest pytest_test_cell_class.py -v

Interactive Visualizations:
The cells marked with #%% can be run interactively in VS Code's scientific mode
to visualize the test data using the built-in Cell plotting methods:
  - plot_raster() - Creates raster plots colored by type/direction
  - plot_raster_by_type_direction() - Organized grid of raster plots
  - plot_psth() - Firing rate analysis (PSTH)
"""

import pandas as pd
import numpy as np
import pytest
from cell_analysis import Cell, PopulationAnalyzer


@pytest.fixture
def test_cell_data():
    """
    Create synthetic test data for a single cell with controlled spike patterns.
    
    Test data specifications (all successful trials):
    1. GO trials:
       - go_cue at 200 ms
       - Left (dir=180): spikes at 220 and 240 ms
       - Right (dir=0): spikes at 320 and 340 ms
    
    2. STOP trials:
       - go_cue at 200 ms, stop_cue at 500 ms
       - Left (dir=180): spikes at 250 and 550 ms
       - Right (dir=0): spikes at 320 and 540 ms
    
    3. CONT trials:
       - go_cue at 200 ms, stop_cue at 500 ms
       - Left (dir=180): spikes at 250 and 550 ms
       - Right (dir=0): spikes at 320 and 540 ms
    """
    test_cell_id = 9999
    test_session = 'test_session'
    
    trials = [
        # Trial 1: GO Left
        {
            'cell_ID': test_cell_id,
            'cell_type': 'MSN',
            'trial_session': test_session,
            'type': 'GO',
            'dir': 180,
            'trial_failed': False,
            'go_cue': 200,
            'stop_cue': np.nan,
            'first_relevant_saccade': 260,
            'ssd_number': np.nan,
            'neural_data': [220, 240],
        },
        # Trial 2: GO Right
        {
            'cell_ID': test_cell_id,
            'cell_type': 'MSN',
            'trial_session': test_session,
            'type': 'GO',
            'dir': 0,
            'trial_failed': False,
            'go_cue': 200,
            'stop_cue': np.nan,
            'first_relevant_saccade': 360,
            'ssd_number': np.nan,
            'neural_data': [320, 340],
        },
        # Trial 3: STOP Left
        {
            'cell_ID': test_cell_id,
            'cell_type': 'MSN',
            'trial_session': test_session,
            'type': 'STOP',
            'dir': 180,
            'trial_failed': False,
            'go_cue': 200,
            'stop_cue': 500,
            'first_relevant_saccade': np.nan,
            'ssd_number': 1,
            'neural_data': [250, 550],
        },
        # Trial 4: STOP Right
        {
            'cell_ID': test_cell_id,
            'cell_type': 'MSN',
            'trial_session': test_session,
            'type': 'STOP',
            'dir': 0,
            'trial_failed': False,
            'go_cue': 200,
            'stop_cue': 500,
            'first_relevant_saccade': np.nan,
            'ssd_number': 1,
            'neural_data': [320, 540],
        },
        # Trial 5: CONT Left
        {
            'cell_ID': test_cell_id,
            'cell_type': 'MSN',
            'trial_session': test_session,
            'type': 'CONT',
            'dir': 180,
            'trial_failed': False,
            'go_cue': 200,
            'stop_cue': 500,
            'first_relevant_saccade': 570,
            'ssd_number': 1,
            'neural_data': [250, 550],
        },
        # Trial 6: CONT Right
        {
            'cell_ID': test_cell_id,
            'cell_type': 'MSN',
            'trial_session': test_session,
            'type': 'CONT',
            'dir': 0,
            'trial_failed': False,
            'go_cue': 200,
            'stop_cue': 500,
            'first_relevant_saccade': 560,
            'ssd_number': 1,
            'neural_data': [320, 540],
        },
    ]
    
    return pd.DataFrame(trials)


@pytest.fixture
def test_cell(test_cell_data):
    """Create a Cell instance with test data"""
    return Cell(test_cell_data)


class TestCell:
    """Test suite for Cell class"""
    
    def test_initialization(self, test_cell):
        """Test that Cell initializes correctly"""
        assert test_cell.cell_id == 9999
        assert test_cell.cell_type == 'MSN'
        assert len(test_cell.data) == 6
        assert set(test_cell.trial_types) == {'GO', 'STOP', 'CONT'}
        assert set(test_cell.directions) == {0, 180}
    
    def test_filter_trials_by_type(self, test_cell):
        """Test filtering trials by type"""
        go_trials = test_cell.filter_trials(trial_type='GO')
        assert len(go_trials) == 2
        assert all(go_trials['type'] == 'GO')
        
        stop_trials = test_cell.filter_trials(trial_type='STOP')
        assert len(stop_trials) == 2
        assert all(stop_trials['type'] == 'STOP')
        
        cont_trials = test_cell.filter_trials(trial_type='CONT')
        assert len(cont_trials) == 2
        assert all(cont_trials['type'] == 'CONT')
    
    def test_filter_trials_by_direction(self, test_cell):
        """Test filtering trials by direction"""
        left_trials = test_cell.filter_trials(direction=180)
        assert len(left_trials) == 3
        assert all(left_trials['dir'] == 180)
        
        right_trials = test_cell.filter_trials(direction=0)
        assert len(right_trials) == 3
        assert all(right_trials['dir'] == 0)
    
    def test_filter_trials_combined(self, test_cell):
        """Test filtering trials with multiple criteria"""
        stop_right = test_cell.filter_trials(trial_type='STOP', direction=0)
        assert len(stop_right) == 1
        assert stop_right.iloc[0]['type'] == 'STOP'
        assert stop_right.iloc[0]['dir'] == 0
    
    def test_align_spikes_to_go_cue(self, test_cell):
        """Test spike alignment to go cue."""
        # Test GO Left trials - don't modify test_cell.data, use filtered data directly
        go_left_data = test_cell.filter_trials(trial_type='GO', direction=180)
        aligned_spikes_left = go_left_data.apply(
            lambda row: np.array(row['neural_data']) - row['go_cue'], axis=1
        )
        
        # First spike should be around 20ms after go_cue (220-200)
        first_trial_spikes = aligned_spikes_left.iloc[0]
        assert len(first_trial_spikes) == 2
        np.testing.assert_array_almost_equal(first_trial_spikes, [20, 40])
        
        # Test GO Right trials
        go_right_data = test_cell.filter_trials(trial_type='GO', direction=0)
        aligned_spikes_right = go_right_data.apply(
            lambda row: np.array(row['neural_data']) - row['go_cue'], axis=1
        )
        
        # First spike should be around 120ms after go_cue (320-200)
        first_trial_spikes = aligned_spikes_right.iloc[0]
        assert len(first_trial_spikes) == 2
        np.testing.assert_array_almost_equal(first_trial_spikes, [120, 140])
    
    def test_align_spikes_to_stop_cue(self, test_cell):
        """Test spike alignment to stop cue."""
        # Test STOP Left trials - don't modify test_cell.data, use filtered data directly
        stop_left_data = test_cell.filter_trials(trial_type='STOP', direction=180)
        aligned_spikes_left = stop_left_data.apply(
            lambda row: np.array(row['neural_data']) - row['stop_cue'], axis=1
        )
        
        # Spikes at 250 and 550ms, stop_cue at 500ms -> aligned: -250, +50
        first_trial_spikes = aligned_spikes_left.iloc[0]
        assert len(first_trial_spikes) == 2
        np.testing.assert_array_almost_equal(first_trial_spikes, [-250, 50])
        
        # Test CONT Right trials with stop_cue alignment
        cont_right_data = test_cell.filter_trials(trial_type='CONT', direction=0)
        aligned_spikes_right = cont_right_data.apply(
            lambda row: np.array(row['neural_data']) - row['stop_cue'], axis=1
        )
        
        # Spikes at 320 and 540ms, stop_cue at 500ms -> aligned: -180, +40
        first_trial_spikes = aligned_spikes_right.iloc[0]
        assert len(first_trial_spikes) == 2
        np.testing.assert_array_almost_equal(first_trial_spikes, [-180, 40])
    
    def test_spike_binning(self, test_cell):
        """Test spike aggregation into bins using aggregate_spikes_by_bins method."""
        # Method signature: aggregate_spikes_by_bins(epok=[-200, 700], bin_size=10, 
        #                   alignment_point='go_cue', trial_type=None, ...)
        
        bin_centers, spike_counts, n_trials = test_cell.aggregate_spikes_by_bins(
            epok=[0, 200],
            bin_size=50,
            alignment_point='go_cue',
            trial_type='GO',
            success_only=True  # Default is True, all our trials are successful
        )
        
        # We have 2 GO trials (1 Left, 1 Right)
        assert n_trials == 2
        
        # Check that spikes are correctly binned
        # Bin [0-50ms]: should have spikes from Left trials (20,40ms)
        # Bin [100-150ms]: should have spikes from Right trials (120,140ms)
        assert len(spike_counts) == len(bin_centers)
        assert spike_counts[0] > 0  # First bin should have spikes from Left trials
        assert spike_counts[2] > 0  # Third bin should have spikes from Right trials
    
    def test_calculate_psth(self, test_cell):
        """Test PSTH calculation"""
        # Calculate PSTH for GO trials aligned to go_cue
        # Returns tuple: (bin_centers, firing_rate, n_trials)
        bin_centers, firing_rate, n_trials = test_cell.calculate_psth(
            alignment_point='go_cue',
            trial_type='GO',
            bin_size=50,
            epok=[0, 200],
            smooth=False  # Disable smoothing for simpler test
        )
        
        # We have 2 GO trials
        assert n_trials == 2
        
        # Check that we got valid arrays
        assert bin_centers is not None
        assert firing_rate is not None
        assert len(bin_centers) == len(firing_rate)
        
        # Firing rate should be positive in bins with spikes
        assert firing_rate[0] > 0  # First bin has Left trial spikes
        assert firing_rate[2] > 0  # Third bin has Right trial spikes


class TestPopulationAnalyzer:
    """Test suite for PopulationAnalyzer class"""
    
    def test_initialization(self, test_cell_data):
        """Test PopulationAnalyzer initialization"""
        # PopulationAnalyzer expects a DataFrame, not a list
        analyzer = PopulationAnalyzer(test_cell_data)
        assert analyzer.n_cells == 1
        assert 9999 in analyzer.cell_ids
    
    def test_get_cell(self, test_cell_data):
        """Test getting a specific cell"""
        # PopulationAnalyzer expects a DataFrame, not a list
        analyzer = PopulationAnalyzer(test_cell_data)
        cell = analyzer.get_cell(9999)
        assert cell is not None
        assert cell.cell_id == 9999
        assert cell.cell_type == 'MSN'


if __name__ == '__main__':
    # Run tests with verbose output when executed directly
    pytest.main([__file__, "-v", "--tb=short"])


# %% [markdown]
# # Visualization of Test Data and Results
# 
# The following cells provide interactive visualizations using the built-in
# plotting methods of the Cell class. Run these cells individually in VS Code's interactive mode.

# %% Create test data for visualization
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cell_analysis import Cell

# Set style for better-looking plots
plt.rcParams['figure.figsize'] = (12, 8)

# Create test data
test_cell_id = 9999
test_session = 'test_session'

trials = [
    # Trial 1: GO Left
    {
        'cell_ID': test_cell_id,
        'cell_type': 'MSN',
        'trial_session': test_session,
        'type': 'GO',
        'dir': 180,
        'trial_failed': False,
        'go_cue': 200,
        'stop_cue': np.nan,
        'first_relevant_saccade': 260,
        'ssd_number': np.nan,
        'neural_data': [220, 240],
    },
    # Trial 2: GO Right
    {
        'cell_ID': test_cell_id,
        'cell_type': 'MSN',
        'trial_session': test_session,
        'type': 'GO',
        'dir': 0,
        'trial_failed': False,
        'go_cue': 200,
        'stop_cue': np.nan,
        'first_relevant_saccade': 360,
        'ssd_number': np.nan,
        'neural_data': [320, 340],
    },
    # Trial 3: STOP Left
    {
        'cell_ID': test_cell_id,
        'cell_type': 'MSN',
        'trial_session': test_session,
        'type': 'STOP',
        'dir': 180,
        'trial_failed': False,
        'go_cue': 200,
        'stop_cue': 500,
        'first_relevant_saccade': np.nan,
        'ssd_number': 1,
        'neural_data': [250, 550],
    },
    # Trial 4: STOP Right
    {
        'cell_ID': test_cell_id,
        'cell_type': 'MSN',
        'trial_session': test_session,
        'type': 'STOP',
        'dir': 0,
        'trial_failed': False,
        'go_cue': 200,
        'stop_cue': 500,
        'first_relevant_saccade': np.nan,
        'ssd_number': 1,
        'neural_data': [320, 540],
    },
    # Trial 5: CONT Left
    {
        'cell_ID': test_cell_id,
        'cell_type': 'MSN',
        'trial_session': test_session,
        'type': 'CONT',
        'dir': 180,
        'trial_failed': False,
        'go_cue': 200,
        'stop_cue': 500,
        'first_relevant_saccade': 570,
        'ssd_number': 1,
        'neural_data': [250, 550],
    },
    # Trial 6: CONT Right
    {
        'cell_ID': test_cell_id,
        'cell_type': 'MSN',
        'trial_session': test_session,
        'type': 'CONT',
        'dir': 0,
        'trial_failed': False,
        'go_cue': 200,
        'stop_cue': 500,
        'first_relevant_saccade': 560,
        'ssd_number': 1,
        'neural_data': [320, 540],
    },
]

viz_data = pd.DataFrame(trials)
viz_cell = Cell(viz_data)

print(f"Created test cell {viz_cell.cell_id} with {len(viz_cell.data)} trials")
print(f"Trial types: {viz_cell.trial_types}")
print(f"Directions: {viz_cell.directions}")

# %% Plot 1: Raster plot colored by trial type
# Using the built-in plot_raster method
raster_type = viz_cell.plot_raster(
    alignment_point='go_cue',
    epok=[-100, 400],
    color_by='type'
)
raster_type

# %% Plot 2: Raster plot colored by direction
raster_direction = viz_cell.plot_raster(
    alignment_point='go_cue',
    epok=[-100, 400],
    color_by='direction'
)
raster_direction

# %% Plot 3: Raster plots by type and direction (organized grid)
# Using the built-in plot_raster_by_type_direction method
raster_grid = viz_cell.plot_raster_by_type_direction(
    alignment_point='go_cue',
    epok=[-100, 400],
    show_legend=True
)
raster_grid

# %% Plot 4: PSTH for GO trials with different alignments
# Plot PSTH aligned to go_cue using plot_psth_by_type_direction
psth_go = viz_cell.plot_psth_by_type_direction(
    alignment_point='go_cue',
    epok=[-50, 200],
    bin_size=20,
    separate_ssd=False,
    smooth=True
)
psth_go

# %% Plot 5: Compare STOP and CONT trials aligned to stop_cue
# STOP and CONT trials visualized using plot_psth_by_type_direction
psth_stop_cont = viz_cell.plot_psth_by_type_direction(
    alignment_point='stop_cue',
    epok=[-300, 300],
    bin_size=50,
    separate_ssd=True,  # Show separate lines for each SSD
    smooth=True
)
psth_stop_cont

# %% Plot 6: Histogram plots for different trial types
# Using plot_histogram_by_type_direction to show spike counts
histograms = viz_cell.plot_histogram_by_type_direction(
    alignment_point='go_cue',
    epok=[-50, 200],
    bin_size=20,
    separate_ssd=False,
    normalize=False
)
histograms

# %% Plot 7: Summary statistics with matplotlib
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Test Data Summary Statistics', fontsize=16, fontweight='bold')

# Plot 1: Trial counts by type
ax = axes[0, 0]
trial_counts = viz_cell.data['type'].value_counts()
bars = ax.bar(trial_counts.index, trial_counts.values, 
              color=['#1f77b4', '#2ca02c', '#d62728'], alpha=0.7, edgecolor='black')
ax.set_ylabel('Number of Trials', fontsize=11)
ax.set_title('Trial Counts by Type', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Plot 2: Trial counts by direction
ax = axes[0, 1]
direction_counts = viz_cell.data['dir'].value_counts()
direction_labels = ['Right (0°)', 'Left (180°)']
bars = ax.bar(direction_labels, [direction_counts.get(0, 0), direction_counts.get(180, 0)], 
              color=['#ff7f0e', '#9467bd'], alpha=0.7, edgecolor='black')
ax.set_ylabel('Number of Trials', fontsize=11)
ax.set_title('Trial Counts by Direction', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Plot 3: Spike count distribution
ax = axes[1, 0]
spike_counts = viz_cell.data['neural_data'].apply(len)
ax.hist(spike_counts, bins=5, color='teal', alpha=0.7, edgecolor='black', linewidth=1.2)
ax.set_xlabel('Number of Spikes per Trial', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Spike Count Distribution', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
mean_spikes = spike_counts.mean()
ax.axvline(mean_spikes, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_spikes:.1f}')
ax.legend(fontsize=10)

# Plot 4: Event timing summary
ax = axes[1, 1]
trial_types = ['GO', 'STOP', 'CONT']
go_cue_times = []
stop_cue_times = []

for tt in trial_types:
    filtered = viz_cell.data[viz_cell.data['type'] == tt]
    if len(filtered) > 0:
        go_cue_times.append(filtered['go_cue'].mean())
        stop_mean = filtered['stop_cue'].mean()
        stop_cue_times.append(stop_mean if not pd.isna(stop_mean) else 0)
    else:
        go_cue_times.append(0)
        stop_cue_times.append(0)

x = np.arange(len(trial_types))
width = 0.35

bars1 = ax.bar(x - width/2, go_cue_times, width, label='Go Cue', 
              color='gray', alpha=0.7, edgecolor='black')
bars2 = ax.bar(x + width/2, stop_cue_times, width, label='Stop Cue',
              color='orange', alpha=0.7, edgecolor='black')

ax.set_ylabel('Time (ms)', fontsize=11)
ax.set_title('Average Event Timing by Trial Type', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(trial_types)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("✨ Visualization Complete!")
print("="*60)
print(f"Cell ID: {viz_cell.cell_id}")
print(f"Total Trials: {len(viz_cell.data)}")
print(f"Trial Types: {', '.join(viz_cell.trial_types)}")
print(f"Directions: {viz_cell.directions}")
print("\nUse the built-in plotting methods:")
print("  - plot_raster() for raster plots")
print("  - plot_raster_by_type_direction() for organized grid")
print("  - plot_psth() for firing rate analysis")
print("="*60)

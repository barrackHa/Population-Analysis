"""
Interactive Panel Dashboard for Xie Decoder Trial Animation

Allows user to:
1. Select direction (Right/Left)
2. Select trial type (STOP/GO)
3. Select specific trial number
4. Play/pause animation of trajectory through 2D hidden space
"""

import panel as pn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from pathlib import Path
import pickle
import torch
import torch.nn as nn

pn.extension()

# ============================================================================
# 1. Define Xie Decoder Architecture
# ============================================================================

class XieDecoder(nn.Module):
    """Xie-style decoder with 2D hidden layer."""
    def __init__(self, n_neurons, hidden_dim=2, n_classes=2):
        super().__init__()
        self.n_neurons = n_neurons
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        self.W = nn.Linear(n_neurons, hidden_dim, bias=True)
        self.M = nn.Linear(hidden_dim, n_classes, bias=False)

    def forward(self, x):
        h = self.W(x)
        logits = self.M(h)
        return logits, h


# ============================================================================
# 2. Load Data and Models
# ============================================================================

# Configuration
SESSION_ID = 'fi211025a'
SSD_NUM = 2

# Paths - use script location to find project root
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DATA_DIR = PROJECT_ROOT / 'data' / 'decoder_data'
MODELS_DIR = PROJECT_ROOT / 'data' / 'decoder_models'
CELL_DF_PATH = PROJECT_ROOT / 'data' / 'unified_cell_trial_data' / 'msn_fiona_cell_trial_data.pkl'

# Load prepared data
data_file = DATA_DIR / f'decoder_data_{SESSION_ID}_ssd{SSD_NUM}.pkl'
with open(data_file, 'rb') as f:
    prepared_data = pickle.load(f)

normalized_datasets = prepared_data['datasets']

# Load full cell_df
cell_df = pd.read_pickle(CELL_DF_PATH)
session_data = cell_df[cell_df['trial_session'] == SESSION_ID].copy()

# Direction mapping
dir_map = {0: 'Right', 180: 'Left'}
directions = [0, 180]

# Load Xie models
xie_models = {}

for direction in directions:
    dir_name = dir_map[direction].lower()
    xie_file = MODELS_DIR / f'xie_decoder_{SESSION_ID}_ssd{SSD_NUM}_{dir_name}.pt'
    checkpoint = torch.load(xie_file, map_location='cpu')

    n_neurons = checkpoint['n_neurons']
    hidden_dim = checkpoint['hidden_dim']
    model = XieDecoder(n_neurons, hidden_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    W_weights = model.W.weight.data.cpu().numpy()
    W_bias = model.W.bias.data.cpu().numpy()

    xie_models[direction] = {
        'model': model,
        'W_weights': W_weights,
        'W_bias': W_bias
    }

print("✓ Data and models loaded")


# ============================================================================
# 3. Compute Epoch Centroids
# ============================================================================

epoch_centroids = {}

for direction in directions:
    X_train = normalized_datasets[direction]['X_train']
    y_train = normalized_datasets[direction]['y_train']

    W_weights = xie_models[direction]['W_weights']
    W_bias = xie_models[direction]['W_bias']

    train_projections_2d = X_train @ W_weights.T + W_bias

    fix_mask = y_train == 0
    go_mask = y_train == 1

    fix_projections = train_projections_2d[fix_mask]
    go_projections = train_projections_2d[go_mask]

    fix_centroid_mean = fix_projections.mean(axis=0)
    fix_centroid_std = fix_projections.std(axis=0)

    go_centroid_mean = go_projections.mean(axis=0)
    go_centroid_std = go_projections.std(axis=0)

    epoch_centroids[direction] = {
        'FIX_state': {
            'mean': fix_centroid_mean,
            'std': fix_centroid_std,
        },
        'GO_state': {
            'mean': go_centroid_mean,
            'std': go_centroid_std,
        }
    }

print("✓ Epoch centroids computed")


# ============================================================================
# 4. Time-Resolved 2D Projection Function
# ============================================================================

def compute_time_resolved_2d_projection(session_data, trial_number, direction,
                                       W_weights, W_bias, cell_ids,
                                       train_mean, train_std,
                                       time_window=(-200, 300), bin_size=108, step_size=10):
    """
    Compute time-resolved projection of a trial onto 2D hidden layer.
    """
    trial_data = session_data[
        (session_data['trial_number'] == trial_number) &
        (session_data['dir'] == direction)
    ].copy()

    if len(trial_data) == 0:
        return None, None

    bin_edges = np.arange(time_window[0], time_window[1] + step_size, step_size)
    time_bins = bin_edges

    n_bins = len(time_bins)
    n_neurons = len(cell_ids)
    firing_rates = np.zeros((n_bins, n_neurons))

    for neuron_idx, cell_id in enumerate(cell_ids):
        cell_trial = trial_data[trial_data['cell_ID'] == cell_id]

        if len(cell_trial) == 0:
            continue

        spike_times = cell_trial['neural_data'].iloc[0]
        go_cue_time = cell_trial['go_cue'].iloc[0]
        spike_times_rel = spike_times - go_cue_time

        if spike_times_rel is None or len(spike_times_rel) == 0:
            continue

        for bin_idx in range(n_bins):
            bin_start = bin_edges[bin_idx]
            bin_end = bin_start + bin_size

            spikes_in_bin = np.sum((spike_times_rel >= bin_start) &
                                   (spike_times_rel < bin_end))
            duration_sec = bin_size / 1000.0
            firing_rates[bin_idx, neuron_idx] = spikes_in_bin / duration_sec

    firing_rates_normalized = (firing_rates - train_mean) / (train_std + 1e-10)
    projection_2d = firing_rates_normalized @ W_weights.T + W_bias

    return time_bins, projection_2d


# ============================================================================
# 5. Get Available Trials
# ============================================================================

def get_available_trials(direction, trial_type):
    """Get list of available trial numbers for given direction and type."""
    if trial_type == 'STOP':
        trials = session_data[
            (session_data['type'] == 'STOP') &
            (session_data['trial_failed'] == False) &
            (session_data['dir'] == direction) &
            (session_data['ssd_number'] == SSD_NUM)
        ]['trial_number'].unique()
    else:  # GO
        test_meta = normalized_datasets[direction]['meta_test']
        trials = test_meta[test_meta['label'] == 1]['trial_number'].unique()

    return sorted(trials.tolist())


# ============================================================================
# 6. Create Dashboard Components
# ============================================================================

# Widgets
direction_selector = pn.widgets.Select(
    name='Direction',
    options={'Right (0°)': 0, 'Left (180°)': 180},
    value=0,
    width=200
)

trial_type_selector = pn.widgets.Select(
    name='Trial Type',
    options=['STOP', 'GO'],
    value='STOP',
    width=200
)

trial_selector = pn.widgets.Select(
    name='Trial Number',
    options=[],
    width=200
)

time_slider = pn.widgets.IntSlider(
    name='Time Point',
    start=0,
    end=0,
    value=0,
    width=600
)

play_button = pn.widgets.Button(
    name='▶ Play',
    button_type='success',
    width=100
)

reset_button = pn.widgets.Button(
    name='⟲ Reset',
    button_type='primary',
    width=100
)

speed_slider = pn.widgets.FloatSlider(
    name='Speed',
    start=0.1,
    end=2.0,
    value=1.0,
    step=0.1,
    width=200
)

# Matplotlib figure
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
matplotlib_pane = pn.pane.Matplotlib(fig, dpi=100, tight=True)

# Status text
status_text = pn.pane.Markdown("Select a direction and trial to begin")

# State variables
state = {
    'playing': False,
    'current_trajectory': None,
    'current_time_bins': None,
    'callback': None,
    'current_direction': 0,
    'current_trial_type': 'STOP',
    'current_trial': None
}


# ============================================================================
# 7. Plotting Function
# ============================================================================

def update_plot():
    """Update the plot with current state."""
    ax.clear()

    direction = state['current_direction']

    if state['current_trajectory'] is None:
        ax.text(0.5, 0.5, 'No trajectory loaded',
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        matplotlib_pane.param.trigger('object')
        return

    # Get centroids
    centroids = epoch_centroids[direction]
    fix_centroid = centroids['FIX_state']
    go_centroid = centroids['GO_state']

    # Plot FIX state centroid
    fix_ellipse = Ellipse(
        fix_centroid['mean'],
        width=2 * fix_centroid['std'][0],
        height=2 * fix_centroid['std'][1],
        facecolor='blue',
        alpha=0.3,
        edgecolor='blue',
        linewidth=2
    )
    ax.add_patch(fix_ellipse)
    ax.scatter(fix_centroid['mean'][0], fix_centroid['mean'][1],
               s=200, marker='s', color='blue', edgecolor='black',
               linewidth=2, label='FIX state', zorder=10)

    # Plot GO state centroid
    go_ellipse = Ellipse(
        go_centroid['mean'],
        width=2 * go_centroid['std'][0],
        height=2 * go_centroid['std'][1],
        facecolor='red',
        alpha=0.3,
        edgecolor='red',
        linewidth=2
    )
    ax.add_patch(go_ellipse)
    ax.scatter(go_centroid['mean'][0], go_centroid['mean'][1],
               s=200, marker='s', color='red', edgecolor='black',
               linewidth=2, label='GO state', zorder=10)

    # Get trajectory and current time point
    traj = state['current_trajectory']
    time_bins = state['current_time_bins']
    current_idx = time_slider.value

    # Plot full trajectory (faded)
    ax.plot(traj[:, 0], traj[:, 1],
            color='gray', alpha=0.3, linewidth=2, linestyle='--',
            label='Full trajectory')

    # Plot trajectory up to current point
    if current_idx > 0:
        ax.plot(traj[:current_idx+1, 0], traj[:current_idx+1, 1],
                color='#2E86AB', linewidth=3, alpha=0.8,
                label='Current path')

    # Plot current position
    current_pos = traj[current_idx]
    ax.scatter(current_pos[0], current_pos[1],
               s=300, marker='o', color='yellow', edgecolor='black',
               linewidth=3, zorder=15, label=f'Current ({time_bins[current_idx]:.0f}ms)')

    # Plot start and end
    ax.scatter(traj[0, 0], traj[0, 1],
               s=150, marker='o', color='green', edgecolor='black',
               linewidth=2, zorder=12, label='Start')
    ax.scatter(traj[-1, 0], traj[-1, 1],
               s=150, marker='X', color='purple', edgecolor='black',
               linewidth=2, zorder=12, label='End')

    # Formatting
    ax.set_xlabel('Dimension 1', fontsize=12)
    ax.set_ylabel('Dimension 2', fontsize=12)
    ax.set_title(f'{dir_map[direction]} Direction - {state["current_trial_type"]} Trial {state["current_trial"]}\n'
                 f'Time: {time_bins[current_idx]:.0f} ms relative to GO cue',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.axvline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)

    # Set axis limits based on data
    all_x = np.concatenate([
        [fix_centroid['mean'][0], go_centroid['mean'][0]],
        traj[:, 0]
    ])
    all_y = np.concatenate([
        [fix_centroid['mean'][1], go_centroid['mean'][1]],
        traj[:, 1]
    ])

    x_margin = (all_x.max() - all_x.min()) * 0.2
    y_margin = (all_y.max() - all_y.min()) * 0.2

    ax.set_xlim([all_x.min() - x_margin, all_x.max() + x_margin])
    ax.set_ylim([all_y.min() - y_margin, all_y.max() + y_margin])

    matplotlib_pane.param.trigger('object')


# ============================================================================
# 8. Callback Functions
# ============================================================================

def update_trial_list(_event=None):
    """Update available trials when direction or trial type changes."""
    direction = direction_selector.value
    trial_type = trial_type_selector.value

    trials = get_available_trials(direction, trial_type)
    trial_selector.options = trials

    if len(trials) > 0:
        trial_selector.value = trials[0]

    status_text.object = f"Found {len(trials)} {trial_type} trials for {dir_map[direction]} direction"


def load_trial(_event=None):
    """Load the selected trial and compute trajectory."""
    direction = direction_selector.value
    trial_type = trial_type_selector.value
    trial_num = trial_selector.value

    if trial_num is None:
        return

    # Stop any ongoing animation
    stop_animation()

    # Get model components
    W_weights = xie_models[direction]['W_weights']
    W_bias = xie_models[direction]['W_bias']
    cell_ids = normalized_datasets[direction]['cell_ids']

    neuron_stats = normalized_datasets[direction]['neuron_stats']
    train_mean = neuron_stats['mean']
    train_std = neuron_stats['std']

    # Compute trajectory
    time_bins, proj_2d = compute_time_resolved_2d_projection(
        session_data,
        trial_num,
        direction,
        W_weights,
        W_bias,
        cell_ids,
        train_mean,
        train_std,
        time_window=(-200, 300),
        bin_size=108,
        step_size=10
    )

    if proj_2d is None:
        status_text.object = f"⚠️ Failed to load trial {trial_num}"
        return

    # Update state
    state['current_trajectory'] = proj_2d
    state['current_time_bins'] = time_bins
    state['current_direction'] = direction
    state['current_trial_type'] = trial_type
    state['current_trial'] = trial_num

    # Update slider
    time_slider.start = 0
    time_slider.end = len(time_bins) - 1
    time_slider.value = 0

    # Update plot
    update_plot()

    status_text.object = f"✓ Loaded {trial_type} trial {trial_num} ({len(time_bins)} time points)"


def on_slider_change(_event):
    """Update plot when slider changes."""
    if state['current_trajectory'] is not None:
        update_plot()


def animate_step():
    """Advance one frame in the animation."""
    if not state['playing']:
        return

    if time_slider.value < time_slider.end:
        time_slider.value += 1
    else:
        # Reached the end
        stop_animation()


def play_pause(_event):
    """Toggle play/pause."""
    if state['playing']:
        stop_animation()
    else:
        start_animation()


def start_animation():
    """Start the animation."""
    state['playing'] = True
    play_button.name = '⏸ Pause'

    # Calculate interval based on speed (100ms base interval)
    interval_ms = int(100 / speed_slider.value)

    # Schedule periodic callback
    state['callback'] = pn.state.add_periodic_callback(
        animate_step,
        period=interval_ms
    )


def stop_animation():
    """Stop the animation."""
    state['playing'] = False
    play_button.name = '▶ Play'

    if state['callback'] is not None:
        state['callback'].stop()
        state['callback'] = None


def reset_animation(_event):
    """Reset to beginning."""
    stop_animation()
    time_slider.value = 0


# ============================================================================
# 9. Wire Up Callbacks
# ============================================================================

direction_selector.param.watch(update_trial_list, 'value')
trial_type_selector.param.watch(update_trial_list, 'value')
trial_selector.param.watch(load_trial, 'value')
time_slider.param.watch(on_slider_change, 'value')
play_button.on_click(play_pause)
reset_button.on_click(reset_animation)

# Initialize trial list
update_trial_list()


# ============================================================================
# 10. Create Dashboard Layout
# ============================================================================

dashboard = pn.template.FastListTemplate(
    title='Xie Decoder Trial Trajectory Animator',
    sidebar=[
        pn.pane.Markdown('## Controls'),
        direction_selector,
        trial_type_selector,
        trial_selector,
        pn.layout.Divider(),
        pn.pane.Markdown('## Animation'),
        pn.Row(play_button, reset_button),
        speed_slider,
        pn.layout.Divider(),
        pn.pane.Markdown('## Time Control'),
        time_slider,
        pn.layout.Divider(),
        status_text,
        pn.layout.Divider(),
        pn.pane.Markdown('''
        ### Instructions
        1. Select direction (Right/Left)
        2. Select trial type (STOP/GO)
        3. Select specific trial number
        4. Use slider to scrub through time
        5. Or click Play to animate

        ### Visualization
        - **Blue square**: FIX state centroid
        - **Red square**: GO state centroid
        - **Yellow circle**: Current position
        - **Green circle**: Trial start (-200ms)
        - **Purple X**: Trial end (300ms)
        ''')
    ],
    main=[
        matplotlib_pane
    ],
    accent_base_color='#2E86AB',
    header_background='#2E86AB'
)

# Serve the dashboard
if __name__ == '__main__':
    dashboard.show(port=5006)

"""
Parallel processing helpers for condition-concatenated PCA analysis.

This module contains worker functions for parallel PSTH extraction.
Separate module is required for ProcessPoolExecutor to pickle functions properly.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from cell_analysis import Cell


def extract_neuron_psths_all_conditions(cell_id, cell_df, config):
    """
    Extract PSTHs for all 6 conditions (3 types × 2 directions) for a single neuron using Cell class.

    Parameters:
    -----------
    cell_id : int
        Cell identifier
    cell_df : DataFrame
        Full cell database
    config : dict
        Configuration parameters

    Returns:
    --------
    psths : dict
        Dictionary with keys: 'GO_L', 'GO_R', 'STOP_L', 'STOP_R', 'CONT_L', 'CONT_R'
        Each value is an array of firing rates
    has_all_conditions : bool
        True if neuron has data for all 6 conditions
    """
    # Get cell data and create Cell object
    cell_data = cell_df[cell_df['cell_ID'] == cell_id]
    cell = Cell(cell_data, verbose=False)

    psths = {}
    has_all_conditions = True

    # Extract for each trial type and direction combination
    for trial_type, epok, alignment in [('GO', config['go_epok'], 'go_cue'),
                                         ('STOP', config['stop_epok'], 'stop_cue'),
                                         ('CONT', config['cont_epok'], 'stop_cue')]:
        for direction in config['directions']:
            dir_label = 'L' if direction == 180 else 'R'
            key = f'{trial_type}_{dir_label}'

            _, firing_rate, n_trials = cell.calculate_psth(
                epok=epok,
                bin_size=config['bin_size'],
                alignment_point=alignment,
                trial_type=trial_type,
                direction=direction,
                ssd_number=None,
                success_only=True,
                smooth=False,
                delta=False,
                normalize_bins=False
            )

            # Check if we have data for this condition
            if firing_rate is None or n_trials == 0:
                has_all_conditions = False
                # Fill with zeros for now
                n_bins = config['go_n_bins'] if trial_type == 'GO' else \
                         config['stop_n_bins'] if trial_type == 'STOP' else \
                         config['cont_n_bins']
                firing_rate = np.zeros(n_bins)

            psths[key] = firing_rate

    return psths, has_all_conditions


def process_cell_worker(args):
    """
    Worker function for parallel PSTH extraction.

    Parameters:
    -----------
    args : tuple
        (cell_id, cell_df, config)

    Returns:
    --------
    tuple : (cell_id, psths, has_all_conditions, trial_counts_dict)
    """
    cell_id, cell_df, config = args

    # Extract PSTHs
    psths, has_all_conditions = extract_neuron_psths_all_conditions(cell_id, cell_df, config)

    # Count trials for each condition
    cell_data = cell_df[cell_df['cell_ID'] == cell_id]
    trial_counts_dict = {}

    for trial_type in ['GO', 'STOP', 'CONT']:
        for direction in config['directions']:
            dir_label = 'L' if direction == 180 else 'R'
            key = f'{trial_type}_{dir_label}'
            n_trials = len(cell_data[
                (cell_data['type'] == trial_type) &
                (cell_data['dir'] == direction) &
                (cell_data['trial_failed'] == False)
            ])
            trial_counts_dict[key] = n_trials

    return cell_id, psths, has_all_conditions, trial_counts_dict

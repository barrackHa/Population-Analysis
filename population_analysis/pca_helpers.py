"""
Helper functions for multi-session PCA analysis.
Worker functions for parallel processing must be in a separate module for pickling.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path


def extract_session_psth_worker(session_id, pickle_path, epok, bin_size,
                                alignment_point, smooth_ker_size, normalize):
    """
    Worker function to extract PSTH matrices from a single session.
    This function must be standalone (top-level) for ProcessPoolExecutor pickling.

    Returns only the essential data (PSTH matrices) and immediately discards Session object.
    """
    # Import classes
    sys.path.insert(0, str(Path.cwd().parent))
    from session_class import Session

    try:
        # Load only this session's data
        cell_df = pd.read_pickle(pickle_path)
        session_data = cell_df[cell_df['trial_session'] == session_id]

        # Create session and filter
        session = Session(session_data, verbose=False)
        session.drop_cells_with_missing_trial_type_or_dir_data()
        n_cells = len(session.cell_ids)

        # Extract GO trials
        go_data = session.get_population_PSTHs_left_right(
            epok=epok,
            bin_size=bin_size,
            alignment_point=alignment_point,
            trial_type='GO',
            success_only=True,
            smooth=True,
            smooth_ker_size=smooth_ker_size,
            delta=True,
            normalize_bins=False,
            normalize=normalize,
            sort_by_peak=False
        )

        # Extract STOP trials (SSD=1)
        stop_data = session.get_population_PSTHs_left_right(
            epok=epok,
            bin_size=bin_size,
            alignment_point=alignment_point,
            trial_type='STOP',
            success_only=True,
            ssd_number=1,
            smooth=True,
            smooth_ker_size=smooth_ker_size,
            delta=True,
            normalize_bins=False,
            normalize=normalize,
            sort_by_peak=False
        )

        # Extract only what we need - PSTH matrices and cell IDs
        result = {
            'session_id': session_id,
            'n_cells': n_cells,
            'go_left': go_data['left']['psth_matrix'],
            'go_right': go_data['right']['psth_matrix'],
            'stop_left': stop_data['left']['psth_matrix'],
            'stop_right': stop_data['right']['psth_matrix'],
            'cell_ids': go_data['left']['cell_ids'],  # Same for all conditions
            'bin_centers': go_data['left']['bin_centers']
        }

        # Explicitly delete Session object to free memory
        del session
        del session_data
        del cell_df
        del go_data
        del stop_data

        return result

    except Exception as e:
        return {
            'session_id': session_id,
            'error': str(e),
            'n_cells': 0
        }

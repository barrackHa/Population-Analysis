"""
Helper functions for multi-session PCA analysis.
Worker functions for parallel processing must be in a separate module for pickling.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path


def extract_session_psth_worker(session_id, pickle_path=None, session_df=None,
                                epok=None, bin_size=None,
                                alignment_point=None, smooth_ker_size=None,
                                normalize=None, compute_average=True, ssd_number=1):
    """
    Worker function to extract PSTH matrices from a single session.
    This function must be standalone (top-level) for ProcessPoolExecutor pickling.

    Parameters:
    -----------
    session_id : str
        Session identifier
    pickle_path : str or Path, optional
        Path to pickle file containing full dataset. Used if session_df not provided.
    session_df : pd.DataFrame, optional
        Pre-loaded DataFrame for this specific session. More efficient than pickle_path.
        If provided, takes precedence over pickle_path.
    epok : list
        Epoch [start, end] in ms
    bin_size : int
        Bin size in ms
    alignment_point : str
        Alignment point (e.g., 'go_cue')
    smooth_ker_size : int
        Smoothing kernel size in ms
    normalize : str or bool
        Normalization method
    compute_average : bool
        If True, compute and return average PSTH across all GO trials (left + right).
        This can be used as a baseline to subtract from condition-specific activity.
    ssd_number : int or None
        Stop signal delay level to extract (1-4). Default: 1 (shortest SSD).
        If None, combines all SSDs.

    Returns only the essential data (PSTH matrices) and immediately discards Session object.
    """
    # Import classes
    sys.path.insert(0, str(Path.cwd().parent))
    from session_class import Session

    try:
        # Get session data - prioritize session_df for efficiency
        if session_df is not None:
            # Use pre-loaded session data (efficient for parallel processing)
            session_data = session_df
        elif pickle_path is not None:
            # Load from pickle (backward compatibility)
            cell_df = pd.read_pickle(pickle_path)
            session_data = cell_df[cell_df['trial_session'] == session_id]
            del cell_df  # Free memory immediately
        else:
            raise ValueError("Either pickle_path or session_df must be provided")

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

        # Extract STOP trials
        stop_data = session.get_population_PSTHs_left_right(
            epok=epok,
            bin_size=bin_size,
            alignment_point=alignment_point,
            trial_type='STOP',
            success_only=True,
            ssd_number=ssd_number,
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

        # Compute average PSTH if requested (GO trials, both directions combined)
        if compute_average:
            avg_data = session.get_population_PSTH_single_condition(
                epok=epok,
                bin_size=bin_size,
                alignment_point=alignment_point,
                trial_type='GO',
                direction=None,  # Both directions combined
                success_only=True,
                smooth=True,
                smooth_ker_size=smooth_ker_size,
                delta=True,
                normalize_bins=False,
                normalize=normalize,
                sort_by_peak=False
            )
            result['avg_psth'] = avg_data['psth_matrix']

        # Explicitly delete objects to free memory
        del session
        del session_data
        del go_data
        del stop_data

        return result

    except Exception as e:
        return {
            'session_id': session_id,
            'error': str(e),
            'n_cells': 0
        }

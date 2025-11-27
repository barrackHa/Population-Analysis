"""
Helper functions for multi-session PCA analysis.
Worker functions for parallel processing must be in a separate module for pickling.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path


def _filter_result_to_cells(result_dict, target_cell_ids):
    """
    Filter PSTH result dictionary to only include specified cells (in order).

    Parameters:
    -----------
    result_dict : dict
        Result dictionary from _extract_psths_from_session
    target_cell_ids : list
        List of cell IDs to keep (in desired order)

    Returns:
    --------
    dict : Filtered result with same structure
    """
    # Build cell index mapping
    cell_to_idx = {cell: idx for idx, cell in enumerate(result_dict['cell_ids'])}

    # Get indices for target cells
    indices = [cell_to_idx[cell] for cell in target_cell_ids]

    # Filter all matrices
    filtered = {
        'n_cells': len(indices),
        'n_cells_dropped_ssd': result_dict.get('n_cells_dropped_ssd', 0),
        'go_left': result_dict['go_left'][indices],
        'go_right': result_dict['go_right'][indices],
        'stop_left': result_dict['stop_left'][indices],
        'stop_right': result_dict['stop_right'][indices],
        'cell_ids': target_cell_ids,
        'bin_centers': result_dict['bin_centers']
    }

    if 'avg_psth' in result_dict:
        filtered['avg_psth'] = result_dict['avg_psth'][indices]

    return filtered


def _extract_psths_from_session(session, epok, bin_size, delta, alignment_point,
                                smooth_ker_size, normalize, compute_average, ssd_number):
    """
    Helper function to extract PSTH matrices from a Session object.
    Used by extract_session_psth_worker for both split and non-split modes.

    Returns dict with GO/STOP matrices and metadata.
    """
    # Extract GO trials
    go_data = session.get_population_PSTHs_left_right(
        epok=epok,
        bin_size=bin_size,
        alignment_point=alignment_point,
        trial_type='GO',
        success_only=True,
        smooth=True,
        smooth_ker_size=smooth_ker_size,
        delta=delta,
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
        delta=delta,
        normalize_bins=False,
        normalize=normalize,
        sort_by_peak=False
    )

    # Get cell IDs from both GO and STOP extractions
    go_cell_ids = set(go_data['left']['cell_ids'])
    stop_cell_ids = set(stop_data['left']['cell_ids'])

    # Find cells that have data for BOTH GO and STOP conditions
    common_cells = go_cell_ids & stop_cell_ids
    n_dropped = len(go_cell_ids) - len(common_cells)

    if n_dropped > 0:
        # Filter matrices to only include common cells
        go_cell_to_idx = {cell: idx for idx, cell in enumerate(go_data['left']['cell_ids'])}
        stop_cell_to_idx = {cell: idx for idx, cell in enumerate(stop_data['left']['cell_ids'])}

        # Get ordered list of common cells (preserve GO order)
        common_cells_ordered = [c for c in go_data['left']['cell_ids'] if c in common_cells]

        # Filter GO matrices
        go_indices = [go_cell_to_idx[c] for c in common_cells_ordered]
        go_left_filtered = go_data['left']['psth_matrix'][go_indices]
        go_right_filtered = go_data['right']['psth_matrix'][go_indices]

        # Filter STOP matrices
        stop_indices = [stop_cell_to_idx[c] for c in common_cells_ordered]
        stop_left_filtered = stop_data['left']['psth_matrix'][stop_indices]
        stop_right_filtered = stop_data['right']['psth_matrix'][stop_indices]

        result = {
            'n_cells': len(common_cells_ordered),
            'n_cells_dropped_ssd': n_dropped,
            'go_left': go_left_filtered,
            'go_right': go_right_filtered,
            'stop_left': stop_left_filtered,
            'stop_right': stop_right_filtered,
            'cell_ids': common_cells_ordered,
            'bin_centers': go_data['left']['bin_centers']
        }
    else:
        # No mismatch - use original data
        result = {
            'n_cells': len(go_cell_ids),
            'n_cells_dropped_ssd': 0,
            'go_left': go_data['left']['psth_matrix'],
            'go_right': go_data['right']['psth_matrix'],
            'stop_left': stop_data['left']['psth_matrix'],
            'stop_right': stop_data['right']['psth_matrix'],
            'cell_ids': go_data['left']['cell_ids'],
            'bin_centers': go_data['left']['bin_centers']
        }

    # Compute average PSTH if requested
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
            delta=delta,
            normalize_bins=False,
            normalize=normalize,
            sort_by_peak=False
        )
        # Filter avg_psth to match filtered cells if needed
        if n_dropped > 0:
            avg_cell_to_idx = {cell: idx for idx, cell in enumerate(avg_data['cell_ids'])}
            avg_indices = [avg_cell_to_idx[c] for c in common_cells_ordered]
            result['avg_psth'] = avg_data['psth_matrix'][avg_indices]
        else:
            result['avg_psth'] = avg_data['psth_matrix']

    return result


def extract_session_psth_worker(session_id, pickle_path=None, session_df=None,
                                epok=None, bin_size=None, delta=True,
                                alignment_point=None, smooth_ker_size=None,
                                normalize=None, compute_average=True, ssd_number=1,
                                split_train_test=False, test_fraction=0.5, random_state=None):
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
    split_train_test : bool, optional
        If True, split trials into train and test sets before extracting PSTHs.
        Default: False
    test_fraction : float, optional
        Fraction of trials to use for testing (0.0 to 1.0). Default: 0.5
    random_state : int, optional
        Random seed for reproducible train/test splits. Default: None

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
        n_cells_original = len(session.cell_ids)

        # Handle train/test splitting
        if split_train_test:
            # Split session into train and test
            train_session, test_session = session.split_to_train_test(
                test_fraction=test_fraction,
                random_state=random_state
            )

            # Extract PSTHs from train session
            train_result = _extract_psths_from_session(
                train_session, epok, bin_size, delta, alignment_point,
                smooth_ker_size, normalize, compute_average, ssd_number
            )

            # Extract PSTHs from test session
            test_result = _extract_psths_from_session(
                test_session, epok, bin_size, delta, alignment_point,
                smooth_ker_size, normalize, compute_average, ssd_number
            )

            # Find common cells between train and test
            # (They may differ because of SSD filtering on different trial subsets)
            train_cells = set(train_result['cell_ids'])
            test_cells = set(test_result['cell_ids'])
            common_cells = train_cells & test_cells

            # Calculate dropped cells for alignment
            n_train_only = len(train_cells - test_cells)
            n_test_only = len(test_cells - train_cells)
            n_dropped_for_alignment = n_train_only + n_test_only

            # Filter both train and test to common cells if needed
            if train_cells != test_cells:
                # Preserve order from train
                common_cells_ordered = [c for c in train_result['cell_ids']
                                       if c in common_cells]

                # Filter both train and test to same cells
                train_result = _filter_result_to_cells(train_result, common_cells_ordered)
                test_result = _filter_result_to_cells(test_result, common_cells_ordered)

            # Package results
            result = {
                'session_id': session_id,
                'split_mode': True,
                'n_cells_original': n_cells_original,
                'n_cells_dropped_for_alignment': n_dropped_for_alignment,
                'train': train_result,
                'test': test_result
            }

            # Cleanup
            del session, train_session, test_session, session_data

        else:
            # Original behavior: no split
            psth_result = _extract_psths_from_session(
                session, epok, bin_size, delta, alignment_point,
                smooth_ker_size, normalize, compute_average, ssd_number
            )

            # Package results
            result = {
                'session_id': session_id,
                'split_mode': False,
                'n_cells_original': n_cells_original,
                'n_cells': psth_result['n_cells'],
                'n_cells_dropped_ssd': psth_result['n_cells_dropped_ssd'],
                'go_left': psth_result['go_left'],
                'go_right': psth_result['go_right'],
                'stop_left': psth_result['stop_left'],
                'stop_right': psth_result['stop_right'],
                'cell_ids': psth_result['cell_ids'],
                'bin_centers': psth_result['bin_centers']
            }

            if 'avg_psth' in psth_result:
                result['avg_psth'] = psth_result['avg_psth']

            # Cleanup
            del session, session_data

        return result

    except Exception as e:
        return {
            'session_id': session_id,
            'error': str(e),
            'n_cells': 0,
            'split_mode': split_train_test
        }

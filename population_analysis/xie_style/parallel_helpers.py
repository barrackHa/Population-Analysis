"""
Helper functions for parallel Lasso regression fitting.

This module must be separate to allow ProcessPoolExecutor to pickle
the worker functions properly.
"""

import numpy as np
from sklearn.linear_model import LassoCV


def fit_lasso_with_splits(X, y, n_splits=100, cv_folds=5, random_state=None):
    """
    Fit Lasso with repeated half-splits for stability.

    Parameters:
    -----------
    X : ndarray (n_trials, n_features)
        Design matrix
    y : ndarray (n_trials,)
        Response vector for one cell
    n_splits : int
        Number of half-splits
    cv_folds : int
        Cross-validation folds for alpha selection
    random_state : int
        Random seed

    Returns:
    --------
    beta_mean : ndarray (n_features,)
        Mean coefficients across splits
    beta_std : ndarray (n_features,)
        Std of coefficients across splits
    """
    n_trials, n_features = X.shape
    beta_samples = []

    rng = np.random.RandomState(random_state)

    for split in range(n_splits):
        # Random half-split
        indices = rng.permutation(n_trials)
        split_size = n_trials // 2
        idx_A = indices[:split_size]
        idx_B = indices[split_size:2*split_size]

        # Fit on split A
        lasso_A = LassoCV(cv=cv_folds, random_state=random_state, max_iter=5000)
        lasso_A.fit(X[idx_A], y[idx_A])
        beta_samples.append(lasso_A.coef_)

        # Fit on split B
        lasso_B = LassoCV(cv=cv_folds, random_state=random_state, max_iter=5000)
        lasso_B.fit(X[idx_B], y[idx_B])
        beta_samples.append(lasso_B.coef_)

    # Average across splits
    beta_samples = np.array(beta_samples)
    beta_mean = beta_samples.mean(axis=0)
    beta_std = beta_samples.std(axis=0)

    return beta_mean, beta_std


def fit_single_cell_worker(args):
    """
    Worker function for parallel processing of a single cell.

    Parameters:
    -----------
    args : tuple
        (cell_idx, y, X, n_splits, cv_folds, random_state)

    Returns:
    --------
    tuple : (cell_idx, beta_mean, beta_std)
    """
    cell_idx, y, X, n_splits, cv_folds, random_state = args

    # Add cell_idx to random_state for reproducibility
    cell_random_state = random_state + cell_idx if random_state is not None else None

    beta_mean, beta_std = fit_lasso_with_splits(
        X, y,
        n_splits=n_splits,
        cv_folds=cv_folds,
        random_state=cell_random_state
    )

    return cell_idx, beta_mean, beta_std

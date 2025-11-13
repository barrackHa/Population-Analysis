"""
Multi-Session PCA Analysis Class

Refactored from multi_session_PCA_analysis.ipynb for cleaner, modular analysis.
Designed to support rolling window PCA analysis and easy parameter tuning.

Author: Barak & Claude
Date: November 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import json
import warnings
from mpl_toolkits.mplot3d import Axes3D

from pca_helpers import extract_session_psth_worker
from session_class import Session

warnings.filterwarnings('ignore')


class MultiSessionPCA:
    """
    Multi-session PCA analysis for neural population dynamics.

    Workflow:
        1. Load data and validate sessions
        2. Extract PSTH matrices from valid sessions (parallel processing)
        3. Concatenate and prepare data for PCA
        4. Fit PCA on GO trials
        5. Project all conditions (GO and STOP) onto PC space
        6. Visualize and save results

    Example:
        >>> config = {'epok': [-50, 500], 'bin_size': 1, 'n_pca_components': 5}
        >>> analyzer = MultiSessionPCA(config)
        >>> analyzer.load_data('data/msn_fiona_cell_trial_data.pkl')
        >>> analyzer.validate_all_sessions()
        >>> analyzer.extract_all_sessions_parallel()
        >>> analyzer.fit_and_project()
        >>> analyzer.plot_3d_trajectory()
    """

    def __init__(self, config=None):
        """
        Initialize MultiSessionPCA with configuration parameters.

        Parameters:
        -----------
        config : dict, optional
            Configuration dictionary with analysis parameters.
            If None, uses default configuration.
        """
        # Set default configuration
        self.config = self._get_default_config()

        # Update with user-provided config
        if config:
            self.config.update(config)

        # Initialize data containers
        self.cell_df = None
        self.session_stats_df = None
        self.validation_df = None
        self.valid_sessions = []
        self.session_psth_data = []
        self.pca = None
        self.standard_scaler = False  # Flag: whether StandardScaler was used

        # PC projections (set after fitting)
        self.go_left_PCs = None
        self.go_right_PCs = None
        self.stop_left_PCs = None
        self.stop_right_PCs = None
        self.time = None

        # Combined matrices
        self.combined_go_left = None
        self.combined_go_right = None
        self.combined_stop_left = None
        self.combined_stop_right = None
        self.combined_avg_psth = None

        # Cell-to-session mapping
        self.cell_session_df = None

    def _get_default_config(self):
        """Return default configuration parameters."""
        return {
            'monkey': 'fiona',
            'excluded_sessions': ['fi210628', 'fi210629', 'fi210704'],

            # PCA parameters
            'epok': [-50, 500],
            'bin_size': 1,
            'alignment_point': 'go_cue',
            'smooth_ker_size': 25,
            'normalize': 'by_baseline_FR',
            'n_pca_components': 5,
            'subtract_average': True,
            'ssd_number': 1,

            # Validation criteria
            'min_cells_per_session': 10,
            'min_trials_per_condition': 5,
            'required_trial_types': ['GO', 'STOP', 'CONT'],
            'required_directions': [0, 180],

            # Visualization
            'figsize_3d': (12, 10),
            'figsize_2d_grid': (16, 14),
            'figsize_time': (16, 6),

            # Parallel processing
            'n_workers': max(1, mp.cpu_count() - 2)
        }

    # ========================================================================
    # SECTION 1: Data Loading
    # ========================================================================

    def load_data(self, pickle_path):
        """
        Load cell database from pickle file.

        Parameters:
        -----------
        pickle_path : str or Path
            Path to MSN cell trial data pickle file
        """
        pickle_path = Path(pickle_path)
        if not pickle_path.exists():
            raise FileNotFoundError(f"Data file not found: {pickle_path}")

        print(f"Loading data from: {pickle_path}")
        self.cell_df = pd.read_pickle(pickle_path)

        print(f"✓ Database loaded: {self.cell_df.shape[0]:,} cell-trial combinations")
        print(f"  Total sessions: {self.cell_df['trial_session'].nunique()}")
        print(f"  Total unique cells: {self.cell_df['cell_ID'].nunique()}")

        return self

    # ========================================================================
    # SECTION 2: Session Validation
    # ========================================================================

    def get_session_statistics(self):
        """Compute basic statistics for all sessions."""
        if self.cell_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        session_stats = []

        for session_id in self.cell_df['trial_session'].unique():
            session_data = self.cell_df[self.cell_df['trial_session'] == session_id]

            n_cells = session_data['cell_ID'].nunique()
            n_trials = len(session_data['trial_number'].unique())
            trial_types = sorted(session_data['type'].unique())
            directions = sorted(session_data['dir'].dropna().unique())

            is_excluded = any(excl in session_id for excl in self.config['excluded_sessions'])

            session_stats.append({
                'session_id': session_id,
                'n_cells': n_cells,
                'n_trials': n_trials,
                'trial_types': trial_types,
                'directions': directions,
                'is_excluded': is_excluded
            })

        self.session_stats_df = pd.DataFrame(session_stats).sort_values('n_cells', ascending=False)

        print(f"✓ Session statistics computed for {len(self.session_stats_df)} sessions")
        print(f"  Excluded sessions: {self.session_stats_df['is_excluded'].sum()}")
        print(f"  Candidate sessions: {(~self.session_stats_df['is_excluded']).sum()}")

        return self

    def _validate_single_session(self, session_data, session_id):
        """
        Validate if a single session meets criteria for PCA analysis.

        Workflow:
        1. Create Session instance
        2. Drop cells with incomplete data
        3. Check validation criteria using Session methods

        Returns:
        --------
        tuple : (is_valid, reason, n_cells_after_drop)
        """
        min_cells = self.config['min_cells_per_session']
        min_trials = self.config['min_trials_per_condition']
        required_types = self.config['required_trial_types']
        required_dirs = self.config['required_directions']

        # Step 1: Create Session
        try:
            session = Session(session_data, verbose=False)
        except Exception as e:
            return False, f"Session creation failed: {str(e)}", 0

        # Step 2: Drop incomplete cells
        session.drop_cells_with_missing_trial_type_or_dir_data()

        # Check 1: Sufficient cells after dropping
        if session.n_cells < min_cells:
            return False, f"Insufficient cells: {session.n_cells} < {min_cells}", session.n_cells

        # Check 2: Has all required trial types
        if not session.has_trial_types(required_types):
            missing = set(required_types) - set(session.trial_types)
            return False, f"Missing trial types: {missing}", session.n_cells

        # Check 3: Has all required directions
        if not session.has_directions(required_dirs):
            present_dirs = [d for d in session.directions if pd.notna(d)]
            missing = set(required_dirs) - set(present_dirs)
            return False, f"Missing directions: {missing}", session.n_cells

        # Check 4: Sufficient trials per condition
        is_valid, reason = session.validate_min_trials_per_condition(
            required_types, required_dirs, min_trials, success_only=True
        )
        if not is_valid:
            return False, reason, session.n_cells

        return True, "Valid", session.n_cells

    def validate_all_sessions(self, verbose=True):
        """
        Validate all sessions and filter for those meeting criteria.

        Parameters:
        -----------
        verbose : bool
            Print validation progress
        """
        if self.session_stats_df is None:
            self.get_session_statistics()

        if verbose:
            print("Validating sessions...")
            print("-" * 80)

        # Use pandas apply for better performance
        def _validate_row(row):
            """Validate a single row from session_stats_df."""
            session_id = row['session_id']

            # Handle excluded sessions
            if row['is_excluded']:
                return pd.Series({
                    'session_id': session_id,
                    'is_valid': False,
                    'reason': 'In exclusion list',
                    'n_cells': 0
                })

            # Get session data and validate
            session_data = self.cell_df[self.cell_df['trial_session'] == session_id]
            is_valid, reason, n_cells = self._validate_single_session(session_data, session_id)

            return pd.Series({
                'session_id': session_id,
                'is_valid': is_valid,
                'reason': reason,
                'n_cells': n_cells
            })

        # Apply validation to all rows
        self.validation_df = self.session_stats_df.apply(_validate_row, axis=1)

        # Print results if verbose
        if verbose:
            for _, row in self.validation_df.iterrows():
                session_id = row['session_id']
                if row['reason'] == 'In exclusion list':
                    print(f"⊗ {session_id}: EXCLUDED")
                elif row['is_valid']:
                    print(f"✓ {session_id}: Valid ({row['n_cells']} cells)")
                else:
                    print(f"✗ {session_id}: Invalid - {row['reason']}")

        # Extract valid sessions
        self.valid_sessions = self.validation_df[self.validation_df['is_valid']]['session_id'].tolist()

        if verbose:
            print("-" * 80)
            print(f"✓ Validation complete:")
            print(f"  Valid sessions: {len(self.valid_sessions)}")
            print(f"  Invalid sessions: {(~self.validation_df['is_valid']).sum()}")
            print(f"  Total cells in valid sessions: {self.validation_df[self.validation_df['is_valid']]['n_cells'].sum()}")

        return self

    # ========================================================================
    # SECTION 3: PSTH Extraction
    # ========================================================================

    def extract_all_sessions_parallel(self, use_preloaded_data=True, pickle_path=None, n_workers=None):
        """
        Extract PSTH matrices from all valid sessions using parallel processing.

        Parameters:
        -----------
        use_preloaded_data : bool, optional
            If True (default), pass session DataFrames to workers (efficient).
            If False, pass pickle_path and let workers load data (backward compatible).
        pickle_path : str or Path, optional
            Path to pickle file. Only used if use_preloaded_data=False.
        n_workers : int, optional
            Number of parallel workers (uses config default if not specified)

        Returns:
        --------
        list : List of session PSTH data dictionaries
            Also stored in self.session_psth_data for later use
        """
        if not self.valid_sessions:
            raise ValueError("No valid sessions. Call validate_all_sessions() first.")

        if n_workers is None:
            n_workers = self.config['n_workers']

        # Determine data passing strategy
        if use_preloaded_data:
            if self.cell_df is None:
                raise ValueError("No data loaded. Call load_data() first or use use_preloaded_data=False with pickle_path.")
            print(f"Extracting PSTHs from {len(self.valid_sessions)} sessions using {n_workers} workers...")
            print(f"  Using pre-loaded data (efficient mode)")
        else:
            # Backward compatibility: use pickle path
            if pickle_path is None:
                base_path = Path.cwd().parent / 'data' / 'unified_cell_trial_data'
                pickle_path = str(base_path / f"msn_{self.config['monkey']}_cell_trial_data.pkl")
            else:
                pickle_path = str(pickle_path)
            print(f"Extracting PSTHs from {len(self.valid_sessions)} sessions using {n_workers} workers...")
            print(f"  Using pickle_path (backward compatible mode)")

        print(f"  Subtract average: {self.config['subtract_average']}")
        print("-" * 80)

        futures = {}
        results = []

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all sessions
            for session_id in self.valid_sessions:
                # Prepare arguments based on mode
                if use_preloaded_data:
                    # Pass session DataFrame directly (efficient)
                    session_df = self.cell_df[self.cell_df['trial_session'] == session_id]
                    future = executor.submit(
                        extract_session_psth_worker,
                        session_id=session_id,
                        session_df=session_df,
                        epok=self.config['epok'],
                        bin_size=self.config['bin_size'],
                        alignment_point=self.config['alignment_point'],
                        smooth_ker_size=self.config['smooth_ker_size'],
                        normalize=self.config['normalize'],
                        compute_average=self.config['subtract_average'],
                        ssd_number=self.config['ssd_number']
                    )
                else:
                    # Pass pickle path (backward compatible)
                    future = executor.submit(
                        extract_session_psth_worker,
                        session_id=session_id,
                        pickle_path=pickle_path,
                        epok=self.config['epok'],
                        bin_size=self.config['bin_size'],
                        alignment_point=self.config['alignment_point'],
                        smooth_ker_size=self.config['smooth_ker_size'],
                        normalize=self.config['normalize'],
                        compute_average=self.config['subtract_average'],
                        ssd_number=self.config['ssd_number']
                    )
                futures[future] = session_id

            # Collect results
            completed = 0
            for future in tqdm(as_completed(futures), total=len(self.valid_sessions), desc="Processing"):
                completed += 1
                session_id = futures[future]

                try:
                    result = future.result()
                    counter = f'[{completed}/{len(self.valid_sessions)}]'

                    if 'error' in result:
                        print(f"{counter} ✗ {session_id}: ERROR - {result['error']}")
                    else:
                        results.append(result)
                        print(f"{counter} ✓ {session_id}: {result['n_cells']} cells")
                except Exception as e:
                    print(f"{counter} ✗ {session_id}: EXCEPTION - {str(e)}")
                    raise e

        print("-" * 80)
        print(f"✓ Extraction complete: {len(results)}/{len(self.valid_sessions)} successful")

        self.session_psth_data = results
        return results

    # ========================================================================
    # SECTION 4: Data Preparation
    # ========================================================================

    def concatenate_sessions(self):
        """Concatenate PSTH matrices from all sessions."""
        if not self.session_psth_data:
            raise ValueError("No session data. Call extract_all_sessions_parallel() first.")

        print("Concatenating PSTH matrices across sessions...")

        go_left_matrices = []
        go_right_matrices = []
        stop_left_matrices = []
        stop_right_matrices = []
        cell_session_mapping = []

        for session_data in self.session_psth_data:
            session_id = session_data['session_id']

            # Append matrices
            go_left_matrices.append(session_data['go_left'])
            go_right_matrices.append(session_data['go_right'])
            stop_left_matrices.append(session_data['stop_left'])
            stop_right_matrices.append(session_data['stop_right'])

            # Track cell-to-session mapping
            for cell_id in session_data['cell_ids']:
                cell_session_mapping.append({
                    'cell_id': cell_id,
                    'session_id': session_id
                })

        # Concatenate along cell dimension
        self.combined_go_left = np.concatenate(go_left_matrices, axis=0)
        self.combined_go_right = np.concatenate(go_right_matrices, axis=0)
        self.combined_stop_left = np.concatenate(stop_left_matrices, axis=0)
        self.combined_stop_right = np.concatenate(stop_right_matrices, axis=0)
        self.time = self.session_psth_data[0]['bin_centers']

        self.cell_session_df = pd.DataFrame(cell_session_mapping)

        print(f"✓ Concatenation complete:")
        print(f"  GO Left: {self.combined_go_left.shape}")
        print(f"  GO Right: {self.combined_go_right.shape}")
        print(f"  STOP Left: {self.combined_stop_left.shape}")
        print(f"  STOP Right: {self.combined_stop_right.shape}")

        return self

    def subtract_average(self):
        """Subtract average PSTH from all conditions."""
        if self.combined_go_left is None:
            raise ValueError("Data not concatenated. Call concatenate_sessions() first.")

        if not self.config['subtract_average']:
            print("✓ Skipping average subtraction (subtract_average=False)")
            return self

        print("Subtracting average PSTH from all conditions...")

        # Collect average PSTH from all sessions
        avg_psth_matrices = []
        for session_data in self.session_psth_data:
            if 'avg_psth' not in session_data:
                raise ValueError(f"Session {session_data['session_id']} missing avg_psth")
            avg_psth_matrices.append(session_data['avg_psth'])

        self.combined_avg_psth = np.concatenate(avg_psth_matrices, axis=0)

        print(f"  Average PSTH shape: {self.combined_avg_psth.shape}")
        print(f"  Mean: {self.combined_avg_psth.mean():.4f}, Std: {self.combined_avg_psth.std():.4f}")

        # Subtract from all conditions
        self.combined_go_left = self.combined_go_left - self.combined_avg_psth
        self.combined_go_right = self.combined_go_right - self.combined_avg_psth
        self.combined_stop_left = self.combined_stop_left - self.combined_avg_psth
        self.combined_stop_right = self.combined_stop_right - self.combined_avg_psth

        print("✓ Average PSTH subtracted")

        return self

    def prepare_pca_matrix(self):
        """Create combined matrix for PCA (GO left + GO right)."""
        if self.combined_go_left is None:
            raise ValueError("Data not prepared. Call concatenate_sessions() first.")

        combined_psth_matrix = np.concatenate(
            [self.combined_go_left, self.combined_go_right],
            axis=1
        )

        print(f"✓ PCA matrix prepared: {combined_psth_matrix.shape}")
        print(f"  (n_cells={combined_psth_matrix.shape[0]}, n_features={combined_psth_matrix.shape[1]})")

        # Data quality checks
        if np.isnan(combined_psth_matrix).any():
            print("  WARNING: Matrix contains NaN values!")
        if np.isinf(combined_psth_matrix).any():
            print("  WARNING: Matrix contains Inf values!")

        return combined_psth_matrix

    # ========================================================================
    # SECTION 5: PCA Fitting & Projection
    # ========================================================================

    def fit_pca(self, n_components=None, standard_scaler=False):
        """
        Fit PCA on GO trial data.

        Parameters:
        -----------
        n_components : int, optional
            Number of components (uses config default if not specified)
        standard_scaler : bool, optional
            If True, apply StandardScaler before PCA (standardizes features to mean=0, std=1).
            Default: False
        """
        if n_components is None:
            n_components = self.config['n_pca_components']

        combined_psth_matrix = self.prepare_pca_matrix()

        print(f"Fitting PCA with {n_components} components...")
        mat_for_pca = combined_psth_matrix.T

        # Apply StandardScaler if requested
        if standard_scaler:
            print("  Using StandardScaler preprocessing")
            scaler = StandardScaler()
            mat_for_pca = scaler.fit_transform(mat_for_pca)
            self.standard_scaler = True
        else:
            self.standard_scaler = False

        self.pca = PCA(n_components=n_components)
        self.pca.fit(mat_for_pca)

        print(f"✓ PCA fitted")
        for i, var in enumerate(self.pca.explained_variance_ratio_, 1):
            print(f"  PC{i}: {var*100:.2f}% variance")
        print(f"  Total: {self.pca.explained_variance_ratio_.sum()*100:.2f}%")

        return self

    def project_all_conditions(self):
        """Project GO and STOP data onto PC space."""
        if self.pca is None:
            raise ValueError("PCA not fitted. Call fit_pca() first.")

        print("Projecting all conditions onto PC space...")

        # Project GO data
        combined_go_matrix = np.concatenate(
            [self.combined_go_left, self.combined_go_right],
            axis=1
        )
        go_projection = self.pca.transform(combined_go_matrix.T).T

        cutoff = self.combined_go_left.shape[1]
        self.go_left_PCs = go_projection[:, :cutoff]
        self.go_right_PCs = go_projection[:, cutoff:]

        # Project STOP data
        combined_stop_matrix = np.concatenate(
            [self.combined_stop_left, self.combined_stop_right],
            axis=1
        )
        stop_projection = self.pca.transform(combined_stop_matrix.T).T

        self.stop_left_PCs = stop_projection[:, :cutoff]
        self.stop_right_PCs = stop_projection[:, cutoff:]

        print(f"✓ Projections complete:")
        print(f"  GO Left: {self.go_left_PCs.shape}")
        print(f"  GO Right: {self.go_right_PCs.shape}")
        print(f"  STOP Left: {self.stop_left_PCs.shape}")
        print(f"  STOP Right: {self.stop_right_PCs.shape}")

        return self

    def fit_and_project(self, n_components=None):
        """Convenience method: fit PCA and project all conditions."""
        self.fit_pca(n_components)
        self.project_all_conditions()
        return self

    # ========================================================================
    # SECTION 6: Visualization
    # ========================================================================

    def plot_3d_trajectory(self, save_path=None, show=True):
        """
        Plot 3D PC trajectories for GO and STOP trials.

        Parameters:
        -----------
        save_path : str or Path, optional
            Path to save figure
        show : bool
            Display figure
        """
        if self.go_left_PCs is None:
            raise ValueError("Data not projected. Call project_all_conditions() first.")

        fig = plt.figure(figsize=self.config['figsize_3d'])
        ax = fig.add_subplot(111, projection='3d')

        # GO trajectories
        ax.plot(self.go_left_PCs[0, :], self.go_left_PCs[1, :], self.go_left_PCs[2, :],
                color='blue', linewidth=2, label='GO Left (180°)', alpha=0.8)
        ax.plot(self.go_right_PCs[0, :], self.go_right_PCs[1, :], self.go_right_PCs[2, :],
                color='green', linewidth=2, label='GO Right (0°)', alpha=0.8)

        # STOP trajectories
        ax.plot(self.stop_left_PCs[0, :], self.stop_left_PCs[1, :], self.stop_left_PCs[2, :],
                color='red', linewidth=2, label='STOP Left (180°)', linestyle='dashed', alpha=0.8)
        ax.plot(self.stop_right_PCs[0, :], self.stop_right_PCs[1, :], self.stop_right_PCs[2, :],
                color='orange', linewidth=2, label='STOP Right (0°)', linestyle='dashed', alpha=0.8)

        # Start markers
        ax.scatter(self.go_left_PCs[0, 0], self.go_left_PCs[1, 0], self.go_left_PCs[2, 0],
                  marker='^', s=200, color='blue', edgecolors='black', linewidths=2, zorder=5)
        ax.scatter(self.go_right_PCs[0, 0], self.go_right_PCs[1, 0], self.go_right_PCs[2, 0],
                  marker='^', s=200, color='green', edgecolors='black', linewidths=2, zorder=5)
        ax.scatter(self.stop_left_PCs[0, 0], self.stop_left_PCs[1, 0], self.stop_left_PCs[2, 0],
                  marker='*', s=200, color='red', edgecolors='black', linewidths=2, zorder=5)
        ax.scatter(self.stop_right_PCs[0, 0], self.stop_right_PCs[1, 0], self.stop_right_PCs[2, 0],
                  marker='*', s=200, color='orange', edgecolors='black', linewidths=2, zorder=5)

        ax.set_xlabel('PC1', fontsize=14)
        ax.set_ylabel('PC2', fontsize=14)
        ax.set_zlabel('PC3', fontsize=14)
        ax.set_title(f'3D PC Trajectories (n={len(self.valid_sessions)} sessions, {self.n_cells} cells)',
                    fontsize=16)
        ax.legend(fontsize=10, loc='upper left')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig, ax

    def plot_2d_grid(self, save_path=None, show=True):
        """
        Plot 2D PC projections in a column layout.

        Displays three 2D projections:
        - PC1 vs PC2
        - PC2 vs PC3
        - PC3 vs PC1

        Parameters:
        -----------
        save_path : str or Path, optional
            Path to save figure
        show : bool
            Display figure
        """
        if self.go_left_PCs is None:
            raise ValueError("Data not projected. Call project_all_conditions() first.")

        def _plot_pc_projection(ax, i, j):
            """
            Helper function to plot PCi vs PCj trajectories.

            Parameters:
            -----------
            ax : matplotlib axis
                Axis to plot on
            i : int
                PC index for x-axis (0-indexed)
            j : int
                PC index for y-axis (0-indexed)
            """
            # Plot trajectories
            ax.plot(self.go_left_PCs[i, :], self.go_left_PCs[j, :], 'b-', lw=2, label='GO Left')
            ax.plot(self.go_right_PCs[i, :], self.go_right_PCs[j, :], 'g-', lw=2, label='GO Right')
            ax.plot(self.stop_left_PCs[i, :], self.stop_left_PCs[j, :], 'r--', lw=2, label='STOP Left')
            ax.plot(self.stop_right_PCs[i, :], self.stop_right_PCs[j, :], linestyle='--',
                    color='orange', lw=2, label='STOP Right')

            # GO start markers
            ax.scatter([self.go_left_PCs[i, 0], self.go_right_PCs[i, 0]],
                      [self.go_left_PCs[j, 0], self.go_right_PCs[j, 0]],
                      marker='^', s=150, c=['blue', 'green'], edgecolors='black', lw=2, zorder=5)

            # STOP start markers
            ax.scatter([self.stop_left_PCs[i, 0], self.stop_right_PCs[i, 0]],
                      [self.stop_left_PCs[j, 0], self.stop_right_PCs[j, 0]],
                      marker='*', s=200, c=['red', 'orange'], edgecolors='black', lw=2, zorder=5)

            # Labels and formatting
            ax.set_xlabel(f'PC{i+1}')
            ax.set_ylabel(f'PC{j+1}')
            ax.set_title(f'PC{i+1} vs PC{j+1}')
            ax.legend()
            ax.grid(True, alpha=0.3)

        fig, axes = plt.subplots(3, 1, figsize=(10, 12))

        # PC1 vs PC2
        _plot_pc_projection(axes[0], 0, 1)

        # PC2 vs PC3
        _plot_pc_projection(axes[1], 1, 2)

        # PC3 vs PC1
        _plot_pc_projection(axes[2], 2, 0)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig, axes

    def plot_pc_timeseries(self, pcs_to_plot=[0, 1, 2], save_path=None, show=True):
        """
        Plot PC time series.

        Parameters:
        -----------
        pcs_to_plot : list
            List of PC indices to plot (0-indexed). Default: [0, 1, 2] (PC1, PC2, PC3)
        save_path : str or Path, optional
            Path to save figure
        show : bool
            Display figure
        """
        if self.go_left_PCs is None:
            raise ValueError("Data not projected. Call project_all_conditions() first.")

        def _plot_pc_timeseries(ax, pc_idx):
            """
            Helper function to plot PC timeseries for a single PC.

            Parameters:
            -----------
            ax : matplotlib axis
                Axis to plot on
            pc_idx : int
                PC index (0-indexed)
            """
            # Plot trajectories over time
            ax.plot(self.time, self.go_left_PCs[pc_idx, :], 'b-', lw=2, label='GO Left')
            ax.plot(self.time, self.go_right_PCs[pc_idx, :], 'g-', lw=2, label='GO Right')
            ax.plot(self.time, self.stop_left_PCs[pc_idx, :], 'r--', lw=2, label='STOP Left')
            ax.plot(self.time, self.stop_right_PCs[pc_idx, :], linestyle='--',
                   color='orange', lw=2, label='STOP Right')

            # Mark t=0 (alignment point)
            ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)

            # Labels and formatting
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel(f'PC{pc_idx+1}')
            ax.set_title(f'PC{pc_idx+1} over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)

        n_pcs = len(pcs_to_plot)
        fig, axes = plt.subplots(n_pcs, 1, figsize=(10, 4*n_pcs))

        if n_pcs == 1:
            axes = [axes]

        # Plot each PC timeseries
        for i, pc_idx in enumerate(pcs_to_plot):
            _plot_pc_timeseries(axes[i], pc_idx)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig, axes

    def plot_session_contributions(self, save_path=None, show=True):
        """
        Plot session contributions (cell counts per session).

        Parameters:
        -----------
        save_path : str or Path, optional
            Path to save figure
        show : bool
            Display figure
        """
        if self.cell_session_df is None:
            raise ValueError("Cell session mapping not available.")

        session_contributions = self.cell_session_df['session_id'].value_counts().sort_values(ascending=False)

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        # Bar plot
        ax = axes[0]
        session_contributions.plot(kind='barh', ax=ax, color='steelblue')
        ax.set_xlabel('Number of Cells')
        ax.set_ylabel('Session ID')
        ax.set_title('Sessions by Cell Contribution')
        ax.invert_yaxis()

        # Histogram
        ax = axes[1]
        ax.hist(session_contributions.values, bins=10, color='steelblue', edgecolor='black')
        ax.set_xlabel('Cells per Session')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Session Contributions')
        ax.axvline(session_contributions.median(), color='red', linestyle='--', lw=2,
                  label=f'Median: {session_contributions.median():.0f}')
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig, axes

    # ========================================================================
    # SECTION 7: I/O and Utilities
    # ========================================================================

    def save_results(self, output_dir):
        """
        Save PC projections, metadata, and mappings to disk.

        Parameters:
        -----------
        output_dir : str or Path
            Directory to save results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save PC projections
        np.save(output_dir / 'go_left_PCs.npy', self.go_left_PCs)
        np.save(output_dir / 'go_right_PCs.npy', self.go_right_PCs)
        np.save(output_dir / 'stop_left_PCs.npy', self.stop_left_PCs)
        np.save(output_dir / 'stop_right_PCs.npy', self.stop_right_PCs)
        np.save(output_dir / 'time.npy', self.time)

        # Save metadata
        metadata = self.get_metadata_dict()
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save mappings
        if self.cell_session_df is not None:
            self.cell_session_df.to_csv(output_dir / 'cell_session_mapping.csv', index=False)

        if self.validation_df is not None:
            self.validation_df.to_csv(output_dir / 'session_validation.csv', index=False)

        print(f"✓ Results saved to: {output_dir}")

        return self

    def get_metadata_dict(self):
        """Return metadata as dictionary."""
        metadata = {
            'epok': self.config['epok'],
            'bin_size': self.config['bin_size'],
            'alignment_point': self.config['alignment_point'],
            'smooth_ker_size': self.config['smooth_ker_size'],
            'normalize': str(self.config['normalize']),
            'subtract_average': self.config['subtract_average'],
            'ssd_number': self.config['ssd_number'],
            'n_pca_components': self.config['n_pca_components'],
            'n_sessions': len(self.valid_sessions),
            'n_cells_total': self.n_cells,
            'valid_sessions': self.valid_sessions,
            'excluded_sessions': self.config['excluded_sessions'],
        }

        if self.pca is not None:
            metadata.update({
                'explained_variance_ratio': self.pca.explained_variance_ratio_.tolist(),
                'cumulative_variance': self.pca.explained_variance_ratio_.cumsum().tolist()
            })

        return metadata

    def get_summary(self):
        """Print analysis summary."""
        print("=" * 80)
        print("MULTI-SESSION PCA ANALYSIS SUMMARY")
        print("=" * 80)

        print(f"\nData Summary:")
        print(f"  Monkey: {self.config['monkey']}")
        print(f"  Sessions analyzed: {len(self.valid_sessions)}")
        print(f"  Total cells: {self.n_cells}")

        print(f"\nPCA Configuration:")
        print(f"  Epoch: {self.config['epok']} ms (relative to {self.config['alignment_point']})")
        print(f"  Bin size: {self.config['bin_size']} ms")
        print(f"  Smoothing: {self.config['smooth_ker_size']} ms")
        print(f"  Normalization: {self.config['normalize']}")
        print(f"  Subtract average: {self.config['subtract_average']}")
        print(f"  Components: {self.config['n_pca_components']}")

        if self.pca is not None:
            print(f"\nPCA Results:")
            for i, var in enumerate(self.pca.explained_variance_ratio_, 1):
                print(f"  PC{i}: {var*100:.2f}% variance")
            print(f"  Total: {self.pca.explained_variance_ratio_.sum()*100:.2f}%")

        print("=" * 80)

    # ========================================================================
    # SECTION 8: Properties
    # ========================================================================

    @property
    def n_sessions(self):
        """Number of valid sessions."""
        return len(self.valid_sessions)

    @property
    def n_cells(self):
        """Total number of cells across all sessions."""
        if self.combined_go_left is not None:
            return self.combined_go_left.shape[0]
        return 0

    @property
    def n_timepoints(self):
        """Number of time points."""
        if self.time is not None:
            return len(self.time)
        return 0

"""
Pani et al. (2022) PCA Implementation

This module provides a class to perform PCA based on the methodology
described in Pani et al. (2022), "Neuronal population dynamics during
motor plan cancellation in nonhuman primates".

The key idea is to identify functional subspaces (Holding-and-Planning Axis, HPA,
and Planning-and-Execution Axis, PEA) to analyze neural dynamics.

Author: Barak & Gemini
Date: November 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.linalg import svd
from tqdm import tqdm

from population_analysis.multi_session_pca import MultiSessionPCA
from population_analysis.session_class import Session


class PaniEtAlPCA(MultiSessionPCA):
    """
    Implements the PCA methodology from Pani et al. (2022).

    This class inherits from MultiSessionPCA to reuse data loading, validation,
    and session management, but implements a distinct PCA workflow.

    Workflow:
    1.  Load and validate data using parent methods.
    2.  Extract PSTHs for finely-grouped trials (by RT for no-stop, by SSD for correct-stop).
    3.  Perform an initial PCA on the concatenated data from all groups for a single session.
    4.  Identify the 'holding plane' using SVD on correct-stop trials.
    5.  Define the HPA (Holding-and-Planning Axis) and PEA (Planning-and-Execution Axis).
    6.  Project trajectories onto these functional axes.
    7.  Visualize the results.
    """

    def __init__(self, config=None):
        """Initialize the PaniEtAlPCA analyzer."""
        super().__init__(config)
        self.hpa = None
        self.pea = None
        self.holding_plane_vectors = None
        self.initial_pca = None
        self.grouped_psth_data = {}
        self.psth_config = {}
        self.normalization_mean = None
        self.normalization_std = None

    def _get_default_config(self):
        """Override default config to add Pani-specific parameters."""
        config = super()._get_default_config()
        config.update({
            'n_rt_groups': 5,
            'n_initial_pca_components': 3,
            'hpa_fit_time_window': [0, 300],
        })
        return config

    def extract_grouped_psths(self, verbose=False):
        """
        Extracts PSTHs for all valid sessions, with trials grouped by RT and SSD.
        This is the core data preparation step for the Pani et al. method.
        It uses a workaround by creating temporary Session objects for each trial group
        to avoid modifying the existing Session class.
        """
        if not self.valid_sessions:
            raise ValueError("No valid sessions. Call validate_all_sessions() first.")

        print(f"Extracting grouped PSTHs from {len(self.valid_sessions)} sessions...")
        
        self.psth_config = {
            'epok': self.config['epok'],
            'bin_size': self.config['bin_size'],
            'alignment_point': self.config['alignment_point'],
            'smooth': True,
            'smooth_ker_size': self.config['smooth_ker_size'],
            'normalize': self.config['normalize'],
        }
        
        psth_config_for_group = {
            'epok': self.psth_config['epok'],
            'bin_size': self.psth_config['bin_size'],
            'alignment_point': self.psth_config['alignment_point'],
            'smooth': self.psth_config['smooth'],
            'smooth_ker_size': self.psth_config['smooth_ker_size'],
            'normalize': self.psth_config['normalize'],
            'trial_type': None,
            'direction': None,
            'ssd_number': None,
            'success_only': False, # Filtering is done before creating the temp session
        }

        for session_id in tqdm(self.valid_sessions, desc="Processing Sessions"):
            session_data = self.cell_df[self.cell_df['trial_session'] == session_id]
            session_groups = {}

            # Group GO trials by Reaction Time
            go_trials_full_session = session_data[(session_data['type'] == 'GO') & (session_data['trial_failed'] == False)].copy()
            if not go_trials_full_session.empty and len(go_trials_full_session['trial_number'].unique()) >= self.config['n_rt_groups']:
                try:
                    rt_labels, rt_bins = pd.qcut(go_trials_full_session['reaction_time'], self.config['n_rt_groups'], labels=False, retbins=True, duplicates='drop')
                    go_trials_full_session['rt_group'] = rt_labels
                    
                    for i in range(go_trials_full_session['rt_group'].max() + 1):
                        group_trial_data = go_trials_full_session[go_trials_full_session['rt_group'] == i]
                        if not group_trial_data.empty:
                            group_session = Session(group_trial_data, verbose=False)
                            if group_session.n_cells > 0:
                                psth_data = group_session.get_population_PSTH_single_condition(**psth_config_for_group)
                                if psth_data and psth_data['psth_matrix'] is not None:
                                    session_groups[f'go_rt_group_{i}'] = psth_data
                except Exception as e:
                    if verbose:
                        print(f"Could not bin RTs for session {session_id}: {e}")

            # Group successful STOP trials by SSD
            stop_trials_full_session = session_data[(session_data['type'] == 'STOP') & (session_data['trial_failed'] == False)]
            if not stop_trials_full_session.empty:
                for ssd in stop_trials_full_session['ssd_number'].unique():
                    if pd.notna(ssd):
                        group_trial_data = stop_trials_full_session[stop_trials_full_session['ssd_number'] == ssd]
                        if not group_trial_data.empty:
                            group_session = Session(group_trial_data, verbose=False)
                            if group_session.n_cells > 0:
                                psth_data = group_session.get_population_PSTH_single_condition(**psth_config_for_group)
                                if psth_data and psth_data['psth_matrix'] is not None:
                                    session_groups[f'stop_ssd_{int(ssd)}'] = psth_data
            
            if session_groups:
                self.grouped_psth_data[session_id] = session_groups

        print("✓ Grouped PSTH extraction complete.")
        return self

    def fit_pani_pca(self, session_id=None):
        """
        Fits the full Pani et al. PCA model for a single session.
        """
        if not self.grouped_psth_data:
            self.extract_grouped_psths()

        if not self.grouped_psth_data:
            raise ValueError("PSTH extraction yielded no data.")

        if session_id is None:
            session_id = self.valid_sessions[0]
        
        if session_id not in self.grouped_psth_data:
            raise ValueError(f"Session {session_id} has no grouped PSTH data. Try another session.")

        print(f"Running Pani et al. PCA on session: {session_id}")
        
        session_groups = self.grouped_psth_data[session_id]
        if not session_groups:
            raise ValueError(f"Session {session_id} has no valid trial groups.")

        # Align cell order across all groups
        all_cell_ids = [set(d['cell_ids']) for d in session_groups.values()]
        common_cell_ids = sorted(list(set.intersection(*all_cell_ids)))
        
        if not common_cell_ids:
            raise ValueError(f"No common cells across all trial groups in session {session_id}.")

        aligned_matrices = {}
        for name, data in session_groups.items():
            cell_to_idx = {cell_id: i for i, cell_id in enumerate(data['cell_ids'])}
            indices = [cell_to_idx[cid] for cid in common_cell_ids]
            aligned_matrices[name] = data['psth_matrix'][indices, :]

        concatenated_matrix = np.concatenate(list(aligned_matrices.values()), axis=1)
        
        self.normalization_mean = concatenated_matrix.mean(axis=1, keepdims=True)
        self.normalization_std = concatenated_matrix.std(axis=1, keepdims=True)
        self.normalization_std[self.normalization_std == 0] = 1
        normalized_matrix = (concatenated_matrix - self.normalization_mean) / self.normalization_std
        
        print(f"Performing initial PCA with {self.config['n_initial_pca_components']} components...")
        self.initial_pca = PCA(n_components=self.config['n_initial_pca_components'])
        self.initial_pca.fit(normalized_matrix.T)
        
        projected_groups = {}
        for name, psth_matrix in aligned_matrices.items():
            norm_psth = (psth_matrix - self.normalization_mean) / self.normalization_std
            projected_groups[name] = self.initial_pca.transform(norm_psth.T).T

        print("Identifying holding plane from correct-stop trials...")
        stop_trajectories = [proj for name, proj in projected_groups.items() if 'stop' in name]
        if not stop_trajectories:
            raise ValueError("No correct-stop trials found to define holding plane.")
            
        concatenated_stops = np.concatenate(stop_trajectories, axis=1)
        
        U, s, Vh = svd(concatenated_stops.T, full_matrices=False)
        self.holding_plane_vectors = Vh[:2, :]
        self.pea = Vh[2, :]
        
        print("Defining HPA and PEA...")
        time_vec = self.grouped_psth_data[session_id][list(session_groups.keys())[0]]['bin_centers']
        hpa_fit_mask = (time_vec >= self.config['hpa_fit_time_window'][0]) & (time_vec <= self.config['hpa_fit_time_window'][1])

        all_trajectories_on_plane = []
        for proj in projected_groups.values():
            data_on_plane = self.holding_plane_vectors @ proj
            all_trajectories_on_plane.append(data_on_plane[:, hpa_fit_mask])
            
        concatenated_on_plane = np.concatenate(all_trajectories_on_plane, axis=1)
        
        pca_hpa = PCA(n_components=1)
        pca_hpa.fit(concatenated_on_plane.T)
        
        hpa_in_2d = pca_hpa.components_[0]
        self.hpa = hpa_in_2d @ self.holding_plane_vectors

        print("✓ Pani et al. PCA model fitted.")
        return self

    def project_on_axes(self, psth_matrix):
        if self.hpa is None or self.pea is None:
            raise ValueError("HPA and PEA not defined. Run fit_pani_pca() first.")
        
        if self.normalization_mean is None or self.normalization_std is None:
            raise ValueError("Normalization parameters not available. Fit the model first.")

        norm_psth = (psth_matrix - self.normalization_mean) / self.normalization_std
        projected_data = self.initial_pca.transform(norm_psth.T).T

        hpa_projection = self.hpa @ projected_data
        pea_projection = self.pea @ projected_data
        
        return hpa_projection, pea_projection

    def plot_hpa_pea_projections(self, session_id=None, show=True):
        if self.hpa is None:
            print("Model not fitted. Run fit_pani_pca() first.")
            return

        if session_id is None:
            session_id = self.valid_sessions[0]

        print(f"Plotting HPA/PEA projections for session {session_id}...")
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        session_groups = self.grouped_psth_data[session_id]
        time_axis = session_groups[list(session_groups.keys())[0]]['bin_centers']

        # Re-align matrices before projecting
        all_cell_ids = [set(d['cell_ids']) for d in session_groups.values()]
        common_cell_ids = sorted(list(set.intersection(*all_cell_ids)))

        for name, data in session_groups.items():
            cell_to_idx = {cell_id: i for i, cell_id in enumerate(data['cell_ids'])}
            indices = [cell_to_idx[cid] for cid in common_cell_ids]
            aligned_matrix = data['psth_matrix'][indices, :]

            hpa_proj, pea_proj = self.project_on_axes(aligned_matrix)
            
            color = 'green' if 'go' in name else 'red'
            linestyle = '-'
            label = name
            
            axes[0].plot(time_axis, pea_proj, color=color, linestyle=linestyle, alpha=0.6, label=label)
            axes[1].plot(time_axis, hpa_proj, color=color, linestyle=linestyle, alpha=0.6)

        axes[0].set_title(f"PEA Projections - Session {session_id}")
        axes[0].set_ylabel("PEA Activity (a.u.)")
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        axes[1].set_title(f"HPA Projections - Session {session_id}")
        axes[1].set_xlabel("Time from Go Cue (ms)")
        axes[1].set_ylabel("HPA Activity (a.u.)")

        for ax in axes:
            ax.grid(True, alpha=0.3)
            ax.axvline(0, color='k', linestyle='--')
            
        plt.tight_layout()
        if show:
            plt.show()
        
        return fig, axes

    def plot_3d_trajectories_with_axes(self, session_id=None, show=True):
        if self.hpa is None:
            print("Model not fitted. Run fit_pani_pca() first.")
            return

        if session_id is None:
            session_id = self.valid_sessions[0]

        print(f"Plotting 3D trajectories for session {session_id}...")
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        session_groups = self.grouped_psth_data[session_id]
        
        # Re-align matrices before projecting
        all_cell_ids = [set(d['cell_ids']) for d in session_groups.values()]
        common_cell_ids = sorted(list(set.intersection(*all_cell_ids)))

        max_abs_val = 0
        for name, data in session_groups.items():
            cell_to_idx = {cell_id: i for i, cell_id in enumerate(data['cell_ids'])}
            indices = [cell_to_idx[cid] for cid in common_cell_ids]
            aligned_matrix = data['psth_matrix'][indices, :]

            norm_psth = (aligned_matrix - self.normalization_mean) / self.normalization_std
            proj = self.initial_pca.transform(norm_psth.T).T
            max_abs_val = max(max_abs_val, np.max(np.abs(proj)))
            
            color = 'green' if 'go' in name else 'red'
            ax.plot(proj[0], proj[1], proj[2], color=color, alpha=0.6, label=name)

        origin = [0, 0, 0]
        ax.quiver(*origin, *self.hpa, color='purple', length=max_abs_val, linewidth=3, label='HPA')
        ax.quiver(*origin, *self.pea, color='black', length=max_abs_val, linewidth=3, label='PEA')

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title(f"3D Trajectories for {session_id}")
        ax.legend(bbox_to_anchor=(1.1, 1))
        
        if show:
            plt.show()
            
        return fig, ax

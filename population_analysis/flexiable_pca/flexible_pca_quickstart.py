"""
Quick start script for FlexiblePCA

Demonstrates:
1. Fit PCA on GO trials
2. Project STOP/CONT trials onto GO-defined PCs
3. Visualize results using plotting utilities

Author: Claude & Barak
Date: December 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import TruncatedSVD

from flexible_pca import FlexiblePCA, TrialSpec
from flexible_pca_plots import create_comparison_figure


def main():
    """Run FlexiblePCA quick start demo."""

    print("="*80)
    print("FLEXIBLE PCA QUICK START")
    print("="*80)

    # 1. Load data
    print("\n1. Loading data...")
    data_path = Path('../../data/unified_cell_trial_data/msn_fiona_cell_trial_data.pkl')
    cell_df = pd.read_pickle(data_path)

    # Filter
    excluded_sessions = ['fi210628', 'fi210629', 'fi210704']
    cell_df = cell_df[~cell_df['trial_session'].isin(excluded_sessions)]
    cell_df = cell_df[cell_df['trial_failed'] == False]

    # Valid sessions
    min_cells = 15
    session_cell_counts = cell_df.groupby('trial_session')['cell_ID'].nunique()
    valid_sessions = session_cell_counts[session_cell_counts >= min_cells].index.tolist()
    cell_df = cell_df[cell_df['trial_session'].isin(valid_sessions)]

    print(f"   Loaded {len(cell_df):,} cell-trial combinations")
    print(f"   {cell_df['cell_ID'].nunique()} unique cells")
    print(f"   {len(valid_sessions)} valid sessions")

    # 2. Create FlexiblePCA instance
    print("\n2. Creating FlexiblePCA instance...")
    fpca = FlexiblePCA(
        cell_df,
        bin_size=1,
        smooth_ker_size=25,
        n_components=5,
        pca_function=TruncatedSVD,
        verbose=True
    )

    # 3. Fit on GO trials only
    print("\n3. Fitting PCA on GO trials (both directions)...")
    fit_specs = [
        TrialSpec('GO', direction=0, epoch=[-50, 300], alignment='go_cue'),
        TrialSpec('GO', direction=180, epoch=[-50, 300], alignment='go_cue'),
    ]

    fpca.fit(fit_specs)

    # 4. Get fit trajectories
    print("\n4. Getting fit trajectories...")
    fit_trajectories = fpca.get_fit_trajectories()

    print("   Fit trajectories:")
    for label, traj in fit_trajectories.items():
        print(f"     {label}: shape {traj.shape}")

    # 5. Project STOP and CONT trials
    print("\n5. Projecting STOP and CONT trials onto GO-defined PCs...")
    proj_specs = [
        TrialSpec('STOP', direction=0, epoch=[-50, 300], alignment='go_cue', ssd_number=2),
        TrialSpec('STOP', direction=180, epoch=[-50, 300], alignment='go_cue', ssd_number=2),
        TrialSpec('CONT', direction=0, epoch=[-50, 300], alignment='go_cue', ssd_number=2),
        TrialSpec('CONT', direction=180, epoch=[-50, 300], alignment='go_cue', ssd_number=2),
    ]

    projections = fpca.project(proj_specs)

    print("   Projection trajectories:")
    for label, traj in projections.items():
        print(f"     {label}: shape {traj.shape}")

    # 6. Visualize using plotting utilities
    print("\n6. Creating visualizations...")

    output_dir = Path('../../data/flexible_pca_results')
    time_axis = fpca.get_time_axis(fit_specs[0])

    # Create all comparison figures
    create_comparison_figure(
        fit_trajectories=fit_trajectories,
        proj_trajectories=projections,
        time_axis=time_axis,
        pca_model=fpca.pca_model,
        output_dir=output_dir,
        prefix='quickstart'
    )

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    var_ratios = fpca.pca_model.explained_variance_ratio_
    print(f"PCA fitted on: GO trials (both directions)")
    print(f"Projected: STOP and CONT trials")
    print(f"Total variance explained: {var_ratios.sum()*100:.2f}%")
    print(f"Top 3 PCs: {var_ratios[:3].sum()*100:.2f}%")
    print(f"\nFigures saved:")
    print(f"  - {output_dir}/quickstart_3d_trajectory.png")
    print(f"  - {output_dir}/quickstart_2d_projections.png")
    print(f"  - {output_dir}/quickstart_timeseries.png")
    print(f"  - {output_dir}/quickstart_variance.png")
    print("="*80)
    print("\nNext steps:")
    print("  1. Check the saved figures in data/flexible_pca_results/")
    print("  2. Try flexible_pca_demo.ipynb for more examples")
    print("  3. Read README_FLEXIBLE_PCA.md for detailed documentation")
    print("="*80)


if __name__ == '__main__':
    main()

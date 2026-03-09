"""
Visualization utilities for FlexiblePCA

Provides ready-to-use plotting functions for PCA results.

Author: Claude & Barak
Date: December 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, Optional, List, Tuple
import seaborn as sns


# Default color scheme
DEFAULT_COLORS = {
    'GO_R': 'cyan',
    'GO_L': 'blue',
    'STOP_R': 'orange',
    'STOP_L': 'red',
    'CONT_R': 'lime',
    'CONT_L': 'green',
}

DEFAULT_LINESTYLES = {
    '_R': '-',   # Solid for right
    '_L': '--',  # Dashed for left
}


def get_linestyle(label: str) -> str:
    """Get linestyle based on label."""
    for key, style in DEFAULT_LINESTYLES.items():
        if key in label:
            return style
    return '-'


def plot_3d_trajectories(trajectories: Dict[str, np.ndarray],
                         pca_model,
                         colors: Optional[Dict[str, str]] = None,
                         title: str = 'Neural Trajectories in PC Space',
                         figsize: Tuple[int, int] = (16, 12),
                         show_start_end: bool = True,
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot 3D trajectories in PC space.

    Parameters:
    -----------
    trajectories : dict
        Dictionary mapping labels to trajectory arrays (n_components, n_time_bins)
    pca_model : sklearn PCA model
        Fitted PCA model (for variance ratios)
    colors : dict or None
        Color mapping for trajectories (defaults to DEFAULT_COLORS)
    title : str
        Plot title
    figsize : tuple
        Figure size
    show_start_end : bool
        Show start (circle) and end (square) markers
    save_path : str or None
        Path to save figure (None = don't save)

    Returns:
    --------
    fig : matplotlib Figure
        The created figure
    """
    if colors is None:
        colors = DEFAULT_COLORS

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Plot each trajectory
    for label, traj in trajectories.items():
        color = colors.get(label, 'gray')
        linestyle = get_linestyle(label)

        ax.plot(traj[0, :], traj[1, :], traj[2, :],
                label=label, color=color, linewidth=2, alpha=0.7, linestyle=linestyle)

        if show_start_end:
            # Start marker (circle)
            ax.scatter(traj[0, 0], traj[1, 0], traj[2, 0],
                      marker='o', s=80, color=color, edgecolors='black',
                      linewidths=1.5, alpha=0.8, zorder=5)

            # End marker (square)
            ax.scatter(traj[0, -1], traj[1, -1], traj[2, -1],
                      marker='s', s=80, color=color, edgecolors='black',
                      linewidths=1.5, alpha=0.8, zorder=5)

    # Labels
    var_ratios = pca_model.explained_variance_ratio_
    ax.set_xlabel(f'PC1 ({var_ratios[0]*100:.1f}%)', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'PC2 ({var_ratios[1]*100:.1f}%)', fontsize=12, fontweight='bold')
    ax.set_zlabel(f'PC3 ({var_ratios[2]*100:.1f}%)', fontsize=12, fontweight='bold')

    # Title
    if show_start_end:
        title += '\nCircle=start, Square=end | Dashed=Left, Solid=Right'
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Legend
    ax.legend(fontsize=10, loc='upper left')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")

    return fig


def plot_2d_projections(trajectories: Dict[str, np.ndarray],
                        pca_model,
                        n_pcs: int = 3,
                        colors: Optional[Dict[str, str]] = None,
                        figsize: Tuple[int, int] = (16, 16),
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot all pairwise 2D projections of PCs.

    Parameters:
    -----------
    trajectories : dict
        Dictionary mapping labels to trajectory arrays
    pca_model : sklearn PCA model
        Fitted PCA model
    n_pcs : int
        Number of PCs to include in grid
    colors : dict or None
        Color mapping for trajectories
    figsize : tuple
        Figure size
    save_path : str or None
        Path to save figure

    Returns:
    --------
    fig : matplotlib Figure
        The created figure
    """
    if colors is None:
        colors = DEFAULT_COLORS

    fig, axes = plt.subplots(n_pcs, n_pcs, figsize=figsize)

    var_ratios = pca_model.explained_variance_ratio_

    for i in range(n_pcs):
        for j in range(n_pcs):
            ax = axes[i, j]

            if i == j:
                # Diagonal: show PC info
                ax.text(0.5, 0.5, f'PC{i+1}\n{var_ratios[i]*100:.1f}%',
                       ha='center', va='center', fontsize=14, fontweight='bold')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
            else:
                # Off-diagonal: plot trajectories
                for label, traj in trajectories.items():
                    color = colors.get(label, 'gray')
                    linestyle = get_linestyle(label)

                    ax.plot(traj[j, :], traj[i, :],
                           label=label, color=color, linewidth=1.2,
                           alpha=0.6, linestyle=linestyle)

                    # Start markers
                    ax.scatter(traj[j, 0], traj[i, 0],
                              marker='o', s=30, color=color, edgecolors='black',
                              linewidths=0.5, zorder=5, alpha=0.7)

                ax.grid(alpha=0.3)

                # Labels only on edges
                if i == n_pcs - 1:
                    ax.set_xlabel(f'PC{j+1}', fontsize=10)
                if j == 0:
                    ax.set_ylabel(f'PC{i+1}', fontsize=10)

                # Legend only on top-right
                if i == 0 and j == n_pcs - 1:
                    ax.legend(fontsize=7, loc='upper right', framealpha=0.8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")

    return fig


def plot_pc_timeseries(trajectories: Dict[str, np.ndarray],
                       time_axis: np.ndarray,
                       pca_model,
                       n_pcs: int = 3,
                       colors: Optional[Dict[str, str]] = None,
                       title: str = 'PC Trajectories Over Time',
                       xlabel: str = 'Time (ms)',
                       alignment_time: float = 0.0,
                       figsize: Tuple[int, int] = (16, 12),
                       save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot PC time series.

    Parameters:
    -----------
    trajectories : dict
        Dictionary mapping labels to trajectory arrays
    time_axis : ndarray
        Time values in ms
    pca_model : sklearn PCA model
        Fitted PCA model
    n_pcs : int
        Number of PCs to plot
    colors : dict or None
        Color mapping for trajectories
    title : str
        Plot title
    xlabel : str
        X-axis label
    alignment_time : float
        Time value for alignment marker
    figsize : tuple
        Figure size
    save_path : str or None
        Path to save figure

    Returns:
    --------
    fig : matplotlib Figure
        The created figure
    """
    if colors is None:
        colors = DEFAULT_COLORS

    fig, axes = plt.subplots(n_pcs, 1, figsize=figsize)
    if n_pcs == 1:
        axes = [axes]

    var_ratios = pca_model.explained_variance_ratio_

    for i in range(n_pcs):
        ax = axes[i]

        # Plot trajectories
        for label, traj in trajectories.items():
            color = colors.get(label, 'gray')
            linestyle = get_linestyle(label)

            ax.plot(time_axis, traj[i, :], label=label,
                   color=color, linewidth=2, alpha=0.7, linestyle=linestyle)

        # Formatting
        ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(alignment_time, color='black', linestyle=':', alpha=0.5,
                  linewidth=1.5, label='Alignment' if i == 0 else '')
        ax.set_ylabel(f'PC{i+1}\n({var_ratios[i]*100:.1f}%)',
                     fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8, ncol=2, framealpha=0.9)
        ax.grid(alpha=0.3)

        if i == 0:
            ax.set_title(title, fontsize=14, fontweight='bold')
        if i == n_pcs - 1:
            ax.set_xlabel(xlabel, fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")

    return fig


def plot_variance_explained(pca_model,
                           figsize: Tuple[int, int] = (14, 5),
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot explained variance (scree plot and cumulative).

    Parameters:
    -----------
    pca_model : sklearn PCA model
        Fitted PCA model
    figsize : tuple
        Figure size
    save_path : str or None
        Path to save figure

    Returns:
    --------
    fig : matplotlib Figure
        The created figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    n_components = len(pca_model.explained_variance_ratio_)

    # Scree plot
    axes[0].bar(range(1, n_components + 1),
               pca_model.explained_variance_ratio_ * 100,
               alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Explained Variance (%)')
    axes[0].set_title('Scree Plot')
    axes[0].grid(axis='y', alpha=0.3)

    # Cumulative variance
    cumvar = np.cumsum(pca_model.explained_variance_ratio_) * 100
    axes[1].plot(range(1, n_components + 1), cumvar,
                marker='o', linewidth=2, markersize=8, color='darkred')
    axes[1].axhline(90, color='gray', linestyle='--', alpha=0.5, label='90%')
    axes[1].set_xlabel('Principal Component')
    axes[1].set_ylabel('Cumulative Explained Variance (%)')
    axes[1].set_title('Cumulative Variance Explained')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")

    return fig


def plot_single_pc_comparison(trajectories: Dict[str, np.ndarray],
                              time_axis: np.ndarray,
                              pc_idx: int,
                              colors: Optional[Dict[str, str]] = None,
                              title: Optional[str] = None,
                              ylabel: Optional[str] = None,
                              xlabel: str = 'Time (ms)',
                              alignment_time: float = 0.0,
                              figsize: Tuple[int, int] = (12, 6),
                              save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot a single PC across multiple conditions.

    Parameters:
    -----------
    trajectories : dict
        Dictionary mapping labels to trajectory arrays
    time_axis : ndarray
        Time values in ms
    pc_idx : int
        PC index to plot (0-indexed)
    colors : dict or None
        Color mapping for trajectories
    title : str or None
        Plot title (auto-generated if None)
    ylabel : str or None
        Y-axis label (auto-generated if None)
    xlabel : str
        X-axis label
    alignment_time : float
        Time value for alignment marker
    figsize : tuple
        Figure size
    save_path : str or None
        Path to save figure

    Returns:
    --------
    fig : matplotlib Figure
        The created figure
    """
    if colors is None:
        colors = DEFAULT_COLORS

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot trajectories
    for label, traj in trajectories.items():
        color = colors.get(label, 'gray')
        linestyle = get_linestyle(label)

        ax.plot(time_axis, traj[pc_idx, :], label=label,
               color=color, linewidth=2.5, alpha=0.8, linestyle=linestyle)

    # Formatting
    ax.axhline(0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    ax.axvline(alignment_time, color='black', linestyle=':', alpha=0.5,
              linewidth=2, label='Alignment')

    if ylabel is None:
        ylabel = f'PC{pc_idx+1} Activity'
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')

    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')

    if title is None:
        title = f'PC{pc_idx+1} Across Conditions'
    ax.set_title(title, fontsize=14, fontweight='bold')

    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")

    return fig


def create_comparison_figure(fit_trajectories: Dict[str, np.ndarray],
                            proj_trajectories: Dict[str, np.ndarray],
                            time_axis: np.ndarray,
                            pca_model,
                            output_dir: str,
                            prefix: str = 'pca_analysis'):
    """
    Create a complete set of comparison figures.

    Generates:
    1. 3D trajectory plot
    2. 2D projection grid
    3. PC time series
    4. Variance explained plots

    Parameters:
    -----------
    fit_trajectories : dict
        Trajectories from fit data
    proj_trajectories : dict
        Trajectories from projected data
    time_axis : ndarray
        Time values
    pca_model : sklearn PCA model
        Fitted PCA model
    output_dir : str
        Directory to save figures
    prefix : str
        Prefix for filenames

    Returns:
    --------
    None (saves figures to disk)
    """
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Combine trajectories
    all_traj = {**fit_trajectories, **proj_trajectories}

    print(f"\nCreating comparison figures in: {output_dir}")

    # 1. 3D trajectory
    print("  1. 3D trajectory plot...")
    plot_3d_trajectories(
        all_traj,
        pca_model,
        title='Neural Trajectories in PC Space',
        save_path=output_dir / f'{prefix}_3d_trajectory.png'
    )
    plt.close()

    # 2. 2D projections
    print("  2. 2D projection grid...")
    plot_2d_projections(
        all_traj,
        pca_model,
        save_path=output_dir / f'{prefix}_2d_projections.png'
    )
    plt.close()

    # 3. Time series
    print("  3. PC time series...")
    plot_pc_timeseries(
        all_traj,
        time_axis,
        pca_model,
        save_path=output_dir / f'{prefix}_timeseries.png'
    )
    plt.close()

    # 4. Variance explained
    print("  4. Variance explained...")
    plot_variance_explained(
        pca_model,
        save_path=output_dir / f'{prefix}_variance.png'
    )
    plt.close()

    print(f"\n✓ All figures saved to: {output_dir}")


if __name__ == '__main__':
    print("FlexiblePCA plotting utilities loaded successfully!")
    print("\nAvailable functions:")
    print("  - plot_3d_trajectories()")
    print("  - plot_2d_projections()")
    print("  - plot_pc_timeseries()")
    print("  - plot_variance_explained()")
    print("  - plot_single_pc_comparison()")
    print("  - create_comparison_figure()")

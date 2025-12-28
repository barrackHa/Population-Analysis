"""
Test script to verify custom labels work correctly in FlexiblePCA.

This tests the fix for the KeyError issue with custom labels like 'GO_R_early'.
"""

import numpy as np
import pandas as pd
from flexible_pca import FlexiblePCA, TrialSpec

print("="*80)
print("Testing Custom Labels in FlexiblePCA")
print("="*80)

# Load a small subset of data for quick testing
print("\n1. Loading data...")
data_path = '../../data/unified_cell_trial_data/msn_fiona_cell_trial_data.pkl'
cell_df = pd.read_pickle(data_path)

# Filter to one session for speed
test_session = 'fi211110a'
cell_df = cell_df[cell_df['trial_session'] == test_session]
cell_df = cell_df[cell_df['trial_failed'] == False]

print(f"   Using session: {test_session}")
print(f"   Cells: {cell_df['cell_ID'].nunique()}")
print(f"   Trials: {len(cell_df)}")

# Create FlexiblePCA
print("\n2. Creating FlexiblePCA instance...")
fpca = FlexiblePCA(
    cell_df,
    bin_size=1,  # 1ms bins
    smooth_ker_size=25,
    n_components=3,
    verbose=False  # Suppress detailed output
)

# Test 1: Custom labels in fit
print("\n3. Testing custom labels in fit()...")
fit_specs = [
    TrialSpec('GO', direction=0, epoch=[-50, 100], alignment='go_cue', label='GO_R_early'),
    TrialSpec('GO', direction=180, epoch=[-50, 100], alignment='go_cue', label='GO_L_early'),
]

print("   Fit specs:")
for spec in fit_specs:
    print(f"     - {spec.label}")

fpca.fit(fit_specs)

# Get trajectories - should have custom labels
fit_traj = fpca.get_fit_trajectories()
print("\n   ✓ Fit trajectories:")
for label, traj in fit_traj.items():
    print(f"     - {label}: shape {traj.shape}")

# Verify custom labels are present
assert 'GO_R_early' in fit_traj, "Missing custom label 'GO_R_early'"
assert 'GO_L_early' in fit_traj, "Missing custom label 'GO_L_early'"
print("\n   ✓ Custom labels preserved correctly!")

# Test 2: Custom labels in project
print("\n4. Testing custom labels in project()...")
proj_specs = [
    TrialSpec('GO', direction=0, epoch=[100, 200], alignment='go_cue', label='GO_R_late'),
    TrialSpec('GO', direction=180, epoch=[100, 200], alignment='go_cue', label='GO_L_late'),
]

print("   Project specs:")
for spec in proj_specs:
    print(f"     - {spec.label}")

projections = fpca.project(proj_specs)
print("\n   ✓ Projection trajectories:")
for label, traj in projections.items():
    print(f"     - {label}: shape {traj.shape}")

# Verify custom labels in projections
assert 'GO_R_late' in projections, "Missing custom label 'GO_R_late'"
assert 'GO_L_late' in projections, "Missing custom label 'GO_L_late'"
print("\n   ✓ Custom labels in projections work correctly!")

# Test 3: Mix of custom and auto labels
print("\n5. Testing mix of custom and auto-generated labels...")
mixed_specs = [
    TrialSpec('GO', direction=0, epoch=[-50, 100], alignment='go_cue'),  # Auto: GO_R
    TrialSpec('STOP', direction=0, epoch=[-50, 100], alignment='go_cue', ssd_number=2, label='STOP_R_custom'),
]

fpca2 = FlexiblePCA(cell_df, bin_size=1, smooth_ker_size=25, n_components=3, verbose=False)
fpca2.fit(mixed_specs)

mixed_traj = fpca2.get_fit_trajectories()
print("   ✓ Mixed labels:")
for label in mixed_traj.keys():
    print(f"     - {label}")

assert 'GO_R' in mixed_traj, "Missing auto-generated label 'GO_R'"
assert 'STOP_R_custom' in mixed_traj, "Missing custom label 'STOP_R_custom'"
print("\n   ✓ Mix of custom and auto labels works correctly!")

# Summary
print("\n" + "="*80)
print("ALL TESTS PASSED!")
print("="*80)
print("\nCustom labels now work correctly:")
print("  ✓ Fit with custom labels")
print("  ✓ Project with custom labels")
print("  ✓ get_fit_trajectories() returns correct labels")
print("  ✓ Mix of custom and auto-generated labels")
print("\nThe demo notebook should now work without KeyError!")
print("="*80)

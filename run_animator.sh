#!/bin/bash
# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Run the dashboard using conda environment's python
.conda/bin/python population_analysis/xie_decoder/xie_trial_animator.py

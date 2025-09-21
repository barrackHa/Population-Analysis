# Behavioral EDA Class

This directory contains a Python class (`BehavioralEDA`) that organizes the exploratory data analysis functionality from the `behavioral_eda.ipynb` notebook into a reusable, well-structured class.

## Files

- `behavioral_eda_class.py` - Main class file with all EDA functionality
- `test_behavioral_eda.py` - Test script demonstrating class usage
- `behavioral_eda.ipynb` - Original notebook (unchanged)

## Class Overview

The `BehavioralEDA` class provides a clean separation between:
1. **Data Processing Methods** - Generate DataFrames for analysis
2. **Plotting Methods** - Create visualizations
3. **Utility Methods** - Helper functions for data processing

## Usage

### Basic Setup

```python
from behavioral_eda_class import BehavioralEDA

# Initialize with path to pickle file
filepath = "path/to/your/behavioral_data.pkl"
eda = BehavioralEDA(filepath)

# Print comprehensive summary
eda.print_summary_stats()
```

### Data Processing Methods

```python
# Basic dataset overview
summary = eda.get_basic_summary()

# Trial counts by type and outcome
trial_summary = eda.get_trial_summary_data()

# Success rates by trial type
success_rates = eda.get_success_rates_data()

# Performance by signal delay (returns stop_perf, cont_perf)
stop_performance, cont_performance = eda.get_signal_delay_performance_data()

# RT scatter data for session analysis
rt_scatter_data = eda.get_rt_scatter_data()

# RT distribution data (returns cont_df, stop_df)
cont_dist, stop_dist = eda.get_rt_distribution_data()
```

### Plotting Methods

```python
# Trial distribution (stacked bar chart)
plot1 = eda.plot_trial_distribution()

# Success rates as percentages
plot2 = eda.plot_success_rates_percentage()

# Direction analysis
plot3 = eda.plot_direction_analysis()

# Trial length distributions
plot4 = eda.plot_trial_length_distribution()

# Go cue timing
plot5 = eda.plot_go_cue_timing()

# Signal delay performance (Figure 1b replication)
plot6 = eda.plot_signal_delay_performance()

# RT scatter plots
plot7 = eda.plot_rt_scatter()

# RT distributions (Figure 1d replication)
plot8 = eda.plot_rt_distributions()

# Display plots (in Jupyter)
plot1.show()  # or just 'plot1'
```

## Key Features

### Automatic Data Processing
- Loads behavioral data from pickle files
- Automatically extracts monkey name from filename
- Processes reaction times on-demand
- Handles various saccade data formats
- Validates data integrity

### Comprehensive Analysis
- **Trial Type Analysis**: GO, STOP, CONT trial performance
- **Direction Effects**: Left/right saccade direction analysis
- **Signal Delay Effects**: Stop Signal Delay (SSD) and Continue Signal Delay (CSD) analysis
- **Reaction Time Analysis**: RT extraction, classification, and distribution analysis
- **Session-by-Session Analysis**: Cross-session RT comparisons

### Clean Architecture
- Separates data processing from visualization
- Caches processed data to avoid recomputation
- Consistent error handling and validation
- Comprehensive documentation

## Dependencies

- pandas
- numpy
- scipy
- holoviews
- hvplot
- panel
- matplotlib
- pathlib

## Method Categories

### Data Processing Methods
- `get_basic_summary()` - Dataset overview statistics
- `get_trial_summary_data()` - Trial counts by type/outcome
- `get_success_rates_data()` - Success rates by trial type
- `get_trial_percentage_data()` - Trial outcomes as percentages
- `get_direction_summary_data()` - Direction-based analysis
- `get_direction_success_rates()` - Success rates by direction
- `get_signal_delay_performance_data()` - Stop/Continue performance by delay
- `get_rt_scatter_data()` - RT data for scatter analysis
- `get_rt_distribution_data()` - RT distribution data

### Plotting Methods
- `plot_trial_distribution()` - Stacked bar chart of trial outcomes
- `plot_success_rates_percentage()` - Success rates as percentages
- `plot_direction_analysis()` - Success by trial type and direction
- `plot_trial_length_distribution()` - Trial length histograms
- `plot_go_cue_timing()` - Go cue timing distributions
- `plot_signal_delay_performance()` - Figure 1b: Stop/Continue performance
- `plot_rt_scatter()` - Session mean RT scatter plots
- `plot_rt_distributions()` - Figure 1d: RT distributions by delay

### Utility Methods
- `print_summary_stats()` - Comprehensive statistical summary
- `_extract_saccade_start()` - Extract saccade timing from arrays
- `_check_flag_consistency()` - Validate trial success flags
- `_process_reaction_times()` - Process RT data from saccades

## Example Workflow

```python
# 1. Initialize
eda = BehavioralEDA("data/fiona_behavioral_data.pkl")

# 2. Get overview
eda.print_summary_stats()

# 3. Process specific data
success_rates = eda.get_success_rates_data()
print(success_rates)

# 4. Create visualizations
trial_dist_plot = eda.plot_trial_distribution()
performance_plot = eda.plot_signal_delay_performance()
rt_dist_plot = eda.plot_rt_distributions()

# 5. Display or save plots
trial_dist_plot.show()
```

## Testing

Run the test script to verify everything works:

```bash
python test_behavioral_eda.py
```

This will test all data processing and plotting methods with your data.

## Notes

- The class automatically detects monkey name from filename ('fiona' or 'yasmin')
- Reaction time processing is performed on-demand and cached
- All plots use holoviews/hvplot for interactive visualization
- The class maintains the same statistical analyses as the original notebook
- Error handling is included for missing data and invalid formats
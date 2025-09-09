# Neural Population Analysis - Non-homogeneous Poisson Process

An interactive Python Panel application that demonstrates neural population analysis using Non-homogeneous Poisson Processes. This app simulates neural spike trains with time-varying firing rates to model how neurons respond to stimuli.

## Features

- **Interactive simulation** of neural spike trains using Non-homogeneous Poisson Processes
- **Customizable parameters** for baseline and enhanced firing rates
- **Real-time visualization** including:
  - Time-varying rate function λ(t)
  - Spike raster plots across multiple trials
  - Peri-Stimulus Time Histogram (PSTH)
- **Statistical analysis** of firing patterns
- **User-friendly web interface** built with Panel

## Installation

1. Clone this repository:
```bash
git clone https://github.com/barrackHa/Population-Analysis-.git
cd Population-Analysis-
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

#### Option 1: Using the runner script
```bash
python run_app.py
```

#### Option 2: Direct Panel command
```bash
panel serve neural_analysis_app.py --show
```

The app will start a local web server. Open your browser and navigate to `http://localhost:5007` to use the application.

### Application Interface

The app provides an interactive interface with:

1. **Parameter Controls**:
   - `lambda_1`: Baseline firing rate (spikes/ms)
   - `lambda_2`: Enhanced firing rate during stimulus (spikes/ms)  
   - `n_trials`: Number of spike trains to simulate

2. **Visualizations**:
   - **Rate Function Plot**: Shows λ(t) across three time periods
   - **Spike Raster Plot**: Individual spike trains across trials
   - **PSTH**: Population firing rate histogram

3. **Statistics Panel**: Real-time analysis of firing rates and patterns

## Neural Model

The application simulates a neuron that exhibits three distinct firing periods:

- **Baseline Period** (-200 to 0 ms): Fires at rate λ₁
- **Enhanced Period** (0 to 200 ms): Increased firing at rate λ₂ (stimulus response)
- **Return Period** (200 to 500 ms): Returns to baseline rate λ₁

### Non-homogeneous Poisson Process

The spike generation follows the theory described in [Random Services - Non-homogeneous Poisson Processes](https://www.randomservices.org/random/poisson/Nonhomogeneous.html).

Key implementation details:
- Uses the **thinning algorithm** for generating spikes with time-varying rates
- Rate function λ(t) defines instantaneous firing probability
- Poisson statistics ensure realistic neural variability

## Files

- `neural_analysis_app.py`: Main application with the Panel interface and Poisson process implementation
- `run_app.py`: Convenience script for starting the application
- `requirements.txt`: Python dependencies
- `README.md`: This documentation

## Dependencies

- Panel ≥ 1.3.0: Interactive web applications
- NumPy ≥ 1.24.0: Numerical computations  
- SciPy ≥ 1.10.0: Statistical functions
- Matplotlib ≥ 3.7.0: Plotting backend
- Bokeh ≥ 3.0.0: Interactive visualization
- Param ≥ 2.0.0: Parameter declarations

## Example Output

The application generates:
1. A rate function plot showing the three distinct time periods
2. Spike raster plots displaying individual trials
3. PSTH showing population-level firing patterns
4. Real-time statistics about observed vs. expected firing rates

## Educational Value

This application is ideal for:
- **Neuroscience education**: Understanding neural firing patterns
- **Statistics learning**: Poisson processes and time-varying rates
- **Data visualization**: Interactive scientific plotting
- **Computational neuroscience**: Spike train analysis methods

## License

MIT License - see the repository for details.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.
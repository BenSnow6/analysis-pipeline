# Thesis Analysis Pipeline

This folder contains all the tools and documentation needed to analyze hovercraft experiment data for thesis Chapter 4 (Validation).

## Structure

```
thesis_analysis/
├── README.md           # This file
├── QUICKSTART.md       # Quick start guide
├── requirements.txt    # Python dependencies
├── plotting/           # Plotting modules
│   ├── __init__.py
│   └── experiment_plots.py
├── scripts/            # Main analysis scripts
│   ├── analyze_experiments.py    # Main analysis pipeline
│   └── simulator_validation.py   # Simulator comparison tools
└── docs/               # Documentation and templates
    ├── experiment_catalog.md     # List of all experiments
    └── thesis_results_template.md # Chapter 4 template
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the analysis:**
   ```bash
   python scripts/analyze_experiments.py
   ```

3. **View results in:**
   - `../analysis_results/` - Summary reports and statistics
   - Individual experiment folders - Plots and processed data

## Features

- **Automated Plot Generation**: Creates standardized plots for all 26 experiments
- **Statistical Analysis**: Calculates key metrics for each experiment
- **Simulator Validation**: Framework for comparing real vs simulated data
- **Thesis Templates**: Pre-formatted content for Chapter 4

## Usage

See `QUICKSTART.md` for detailed instructions.

## Key Scripts

### `scripts/analyze_experiments.py`
Main analysis pipeline that:
- Processes all experiments
- Generates plots
- Creates summary statistics
- Prepares data for simulator comparison

### `scripts/simulator_validation.py`
Validation framework that:
- Compares real vs simulated trajectories
- Calculates error metrics
- Generates validation plots
- Evaluates against criteria

### `plotting/experiment_plots.py`
Plotting module that creates:
- GPS track visualizations
- Speed and heading analysis
- Turn performance metrics
- IMU sensor comparisons
- Interactive HTML maps

## Output

All results are saved to `../analysis_results/` including:
- Summary reports (Markdown and JSON)
- Processed data for simulator comparison
- Validation results (when simulator data is available)

## Documentation

- `docs/experiment_catalog.md` - Complete list of experiments with descriptions
- `docs/thesis_results_template.md` - Template for thesis Chapter 4
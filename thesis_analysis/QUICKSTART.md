# Quick Start Guide for Hovercraft Analysis Pipeline

## Overview

This guide will help you quickly start analyzing your hovercraft experiment data and prepare results for your thesis.

## Prerequisites

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure your data is organized in the expected structure:
```
02_Evaluation_Experiments/
├── 1a_1_Minimum_Radius_Turn/
│   ├── morning/
│   └── afternoon/
├── 1a_2_Rate_of_Turn_vs_Nosewheel_Steering_Angle/
└── ... (other categories)
```

## Step 1: Generate All Plots

Run the main analysis script to generate plots for all experiments:

```bash
cd thesis_analysis
python scripts/analyze_experiments.py
```

This will:
- Process all 26 experiments
- Generate standardized plots for each
- Create summary statistics
- Save results to `analysis_results/`

## Step 2: View Individual Experiment Plots

For a specific experiment, you can generate plots directly:

```python
from plotting.experiment_plots import ExperimentPlotter

# Example: Plot the fast starboard turn experiment
plotter = ExperimentPlotter(
    base_path="02_Evaluation_Experiments",
    experiment_name="1a_1_Minimum_Radius_Turn/afternoon/007_Fast_stbd_turn_1"
)
plotter.generate_all_plots()
```

## Step 3: Prepare Data for Simulator Comparison

Export standardized data format for simulator validation:

```bash
cd thesis_analysis
python scripts/analyze_experiments.py --simulator-prep
```

This creates files in `analysis_results/simulator_comparison/` with:
- Standardized CSV files
- Metadata JSON files
- Ready for simulator comparison

## Step 4: Run Simulator Validation

Once you have simulator output data:

```python
import sys
sys.path.append('thesis_analysis')
from scripts.simulator_validation import validate_experiment

# Compare real vs simulated data
validate_experiment(
    real_data_path="path/to/real/experiment",
    sim_data_path="path/to/simulator/output",
    output_path="validation_results/experiment_name"
)
```

## Step 5: Generate Thesis Content

### View Analysis Summary

Check `analysis_results/` for:
- `analysis_summary_YYYYMMDD_HHMMSS.md` - Markdown summary
- `experiment_summary_YYYYMMDD_HHMMSS.json` - Detailed statistics

### Use the Thesis Template

Copy content from `docs/thesis_results_template.md` and fill in with your results.

## Common Tasks

### Process Only Specific Experiments

```python
import sys
sys.path.append('thesis_analysis')
from scripts.analyze_experiments import ExperimentAnalyzer

analyzer = ExperimentAnalyzer(".")
stats = analyzer.analyze_experiment(
    category="1a_1_Minimum_Radius_Turn",
    time_slot="afternoon",
    experiment="007_Fast_stbd_turn_1"
)
print(stats)
```

### Generate Plots Without Analysis

```bash
cd thesis_analysis
python scripts/analyze_experiments.py --no-plots
```

### View Interactive Maps

Open the HTML files in experiment `plots/` directories:
- `interactive_map.html` - GPS track with controls

### Export Data for External Analysis

All processed data is saved as CSV in the plots directories for use in other tools.

## Troubleshooting

### Missing Data Files
- Check file naming conventions match expected patterns
- Ensure `time_from_sync` column exists in all data files

### Memory Issues with Large Datasets
- Process experiments individually rather than all at once
- Reduce plot resolution in `plotting/experiment_plots.py`

### Validation Errors
- Ensure simulator output matches expected format
- Check time synchronization between datasets

## Next Steps

1. Review generated plots and identify any data quality issues
2. Run simulator experiments matching real-world conditions
3. Perform validation comparisons
4. Document findings using the thesis template
5. Iterate on simulator parameters based on validation results

For detailed documentation, see individual module docstrings and comments.
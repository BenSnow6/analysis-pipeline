# Debug Interpolation Script Guide

## Overview

The `debug_interpolation.py` script is a diagnostic tool for testing and visualizing the RPM interpolation functionality used in the multi-sensor fusion pipeline (WP-4). It helps identify issues with gap filling and validates interpolation quality.

## Location

- **Script**: `/src/scripts/debug_interpolation.py`
- **Module**: Part of the RPM estimation analysis pipeline

## Purpose

1. **Test interpolation algorithms** on synthetic data with known gaps
2. **Visualize interpolation results** to understand behavior
3. **Analyze interpolation quality** with various metrics
4. **Debug real experimental data** to identify interpolation issues
5. **Optimize interpolation parameters** like maximum gap size

## Features

### Synthetic Data Testing
- Creates RPM profiles with controlled gaps
- Tests different gap sizes and interpolation limits
- Validates interpolation accuracy

### Real Data Analysis
- Loads WP-2 (Welch PSD) results
- Applies interpolation with configurable parameters
- Generates quality metrics and visualizations

### Visualization
- Multi-panel plots showing:
  - Original vs interpolated RPM values
  - SNR levels and validity thresholds
  - Availability percentage over time
- Highlights interpolated regions

### Quality Metrics
- Availability gain
- Number of gaps filled
- Interpolation fraction
- Gap length statistics
- Smoothness analysis

## Usage

### Basic Synthetic Test
```bash
python src/scripts/debug_interpolation.py --test-synthetic
```

### Real Data Test
```bash
python src/scripts/debug_interpolation.py \
    --experiment 026_Engine_rpm_sweep \
    --session afternoon \
    --sensor Sensor_1
```

### Batch Processing
```bash
# Test all sensors for an experiment
python src/scripts/debug_interpolation.py \
    --experiment 016_Straight_cruise_1 \
    --session morning
```

### Interactive Plotting
```bash
python src/scripts/debug_interpolation.py \
    --test-synthetic \
    --plot
```

## Command Line Arguments

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--test-synthetic` | flag | Run synthetic data test | - |
| `--experiment` | str | Experiment name for real data | - |
| `--session` | str | Session (morning/afternoon) | - |
| `--sensor` | str | Specific sensor ID | All sensors |
| `--plot` | flag | Show plots interactively | Save only |
| `--max-gap` | float | Maximum gap to interpolate (s) | 5.0 |

## Output

### Console Output
- Interpolation statistics
- Quality metrics
- Processing progress

### Generated Files
- `debug_interpolation_synthetic_gap{X}.png` - Synthetic test results
- `debug_interpolation_{experiment}_{sensor}.png` - Real data results

## Synthetic Test Details

The synthetic test creates a 60-second RPM profile with:
1. **Ramp up** (0-20s): 1000 → 2000 RPM
2. **Steady state** (20-42s): 2000 RPM with slight variation
3. **Ramp down** (42-54s): 2000 → 1000 RPM
4. **Idle** (54-60s): 1000 RPM

Configured gaps:
- 2-second gap at t=10-12s
- 5-second gap at t=25-30s
- 8-second gap at t=40-48s
- 1-second gap at t=55-56s

## Quality Metrics Explained

### Availability
Percentage of valid RPM estimates in the time series.

### Availability Gain
Improvement in availability after interpolation.

### Gap Statistics
- **num_gaps_original**: Number of discontinuous gaps
- **num_gaps_filled**: Gaps successfully interpolated
- **max_gap_length**: Longest gap duration
- **mean_gap_length**: Average gap duration

### Interpolation Fraction
Percentage of frames that were interpolated vs original valid data.

## Interpolation Algorithm

The script uses the `interpolate_missing_frames` function from the fusion module:

1. **Linear interpolation** between valid points
2. **Gap size limiting** - only interpolate gaps ≤ max_gap_s
3. **Boundary handling** - no extrapolation beyond data bounds
4. **SNR assignment** - interpolated frames get SNR=10dB (minimum valid)
5. **Confidence scoring** - interpolated frames marked with confidence=0.5

## Common Use Cases

### 1. Validate Interpolation Parameters
Test different `max_gap` values to find optimal settings:
```bash
for gap in 3.0 5.0 7.0 10.0; do
    python src/scripts/debug_interpolation.py \
        --experiment 026_Engine_rpm_sweep \
        --session afternoon \
        --max-gap $gap
done
```

### 2. Debug Specific Sensor Issues
Investigate why a sensor has poor availability:
```bash
python src/scripts/debug_interpolation.py \
    --experiment 021_Quarter_turn_port \
    --session morning \
    --sensor Sensor_3 \
    --plot
```

### 3. Compare Interpolation Strategies
Run synthetic tests with known ground truth to evaluate accuracy.

## Integration with WP-4

This script helps debug the interpolation step in the WP-4 fusion pipeline:

1. **Pre-fusion**: Interpolate individual sensor streams
2. **Fusion**: Combine interpolated streams using SNR weighting
3. **Post-fusion**: Apply final smoothing and validation

## Troubleshooting

### No WP-2 Results Found
- Ensure WP-2 processing has been run for the experiment
- Check file paths in `/data/processed/rpm/wp2/`

### Memory Issues with Large Datasets
- Process sensors individually rather than all at once
- Reduce plot resolution or skip interactive plotting

### Interpolation Not Working
- Check if gaps exceed `max_gap` parameter
- Verify SNR thresholds in the original data
- Ensure monotonic timestamps

## Related Documentation

- [WP-4 Fusion README](/docs/results/rpm_estimation/wp4_fusion/WP4_README.md)
- [RPM Estimation Overview](/docs/results/rpm_estimation/README.md)
- [Fusion Module API](/src/analysis/rpm/fusion.py)
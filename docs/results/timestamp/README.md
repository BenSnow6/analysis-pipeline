# Timestamp Analysis Tool

A Python tool for analyzing timestamp consistency and detecting timing issues in multi-sensor hovercraft data.

## Quick Start

### Installation

Ensure you have Python 3.7+ and the required dependencies:

```bash
pip install numpy pandas matplotlib seaborn pyyaml
```

### Basic Usage

Analyze a single experiment:
```bash
python -m hovercraft_data_analysis.timestamp_analysis \
    --experiment "1a_1_Minimum_Radius_Turn/afternoon/007_Fast_stbd_turn_1"
```

Analyze all experiments:
```bash
python -m hovercraft_data_analysis.timestamp_analysis --all
```

## Features

- **Multi-sensor Support**: Handles GPS (1Hz) and IMU sensors (100-200Hz)
- **Configurable Thresholds**: Sensor-specific timing requirements via YAML
- **Comprehensive Analysis**: Jitter, gaps, sampling rate validation
- **Rich Visualizations**: Time series, histograms, alignment plots
- **Multiple Output Formats**: HTML reports, CSV summaries, JSON data

## Configuration

Edit `config/sensor_specs.yaml` to adjust sensor-specific parameters:

```yaml
sensors:
  sensor_3:
    expected_rate_hz: 200
    jitter_threshold_ms: 20
    gap_threshold_factor: 10.0
```

## Output

The tool generates:
- **HTML Report**: Comprehensive analysis with embedded plots
- **CSV Summary**: Key metrics for spreadsheet analysis
- **PNG Plots**: Individual sensor and summary visualizations
- **JSON Results**: Complete analysis data for programmatic access

## Command Line Options

```
--experiment, -e    Path to specific experiment
--all, -a          Analyze all experiments
--spec, -s         Custom sensor specifications file
--output, -o       Output directory (default: timestamp_analysis_output)
--plot, -p         Generate plots (default: True)
--verbose, -v      Detailed console output
--update-spec      Auto-detect sampling rates
```

## Example Output

```
Analyzing: 007_Fast_stbd_turn_1
=====================================
GPS (1Hz): Mean jitter 15.2ms (✓), Max 19.8ms (✓)
IMU Sensor_3 (200Hz): Mean jitter 0.5ms (✓), Max 2.1ms (✓)
WARNING: 2 gaps detected > 100ms in GPS data
```

## See Also

- [Technical Documentation](TECHNICAL_DOCUMENTATION.md) - Detailed implementation and theory
- [Sensor Specifications](config/sensor_specs.yaml) - Default sensor configurations
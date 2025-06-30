# Orientation Analysis Module

This module validates sensor orientations and estimates biases for hovercraft IMU data.

## Overview

The orientation analysis performs three key validations:
1. **Static Validation**: Uses gravity measurements during static periods to verify rotation matrices
2. **Dynamic Validation**: Uses known maneuver patterns to confirm sensor orientations
3. **Bias Estimation**: Calculates accelerometer and gyroscope biases from static data

## Key Features

- Validates rotation matrices without assuming they are correct
- Compares measured gravity direction with expected sensor axes
- Uses physical intuition from known maneuvers (e.g., forward acceleration)
- Estimates and corrects sensor biases
- Generates comprehensive visualizations and reports

## Directory Structure

```
orientation_analysis/
├── orientation_config.yaml    # Configuration file with sensor positions and parameters
├── orientation_check.py       # Main validation orchestrator
├── rotation_validator.py      # Validates rotation matrices using gravity
├── static_detector.py         # Detects stationary periods in data
├── dynamic_validator.py       # Validates using dynamic maneuvers
├── bias_estimator.py         # Estimates sensor biases
├── plot_orientation.py       # Visualization utilities
├── run_orientation.py        # CLI interface
├── test_orientation.py       # Unit tests
└── README.md                 # This file
```

## Usage

### Basic Usage

Process the three key experiments:

```bash
python run_orientation.py
```

### Custom Experiments

```bash
python run_orientation.py -e 007_Fast_stbd_turn_1 016_Straight_cruise_1 -o results/
```

### Generate Plots Only

```bash
python run_orientation.py --plot-only
```

## Configuration

The `orientation_config.yaml` file contains:

- **Craft specifications**: Dimensions and reference frame definition
- **Sensor positions**: Exact 3D coordinates relative to craft origin
- **Expected orientations**: How each sensor is mounted
- **Validation tolerances**: Acceptable error thresholds
- **Static detection parameters**: Thresholds for identifying stationary periods

## Validation Process

### 1. Static Validation

- Detects periods where angular velocity < 0.05 rad/s
- Extracts gravity vector from accelerometer data
- Compares with expected sensor orientation
- Validates rotation matrix orthonormality

### 2. Dynamic Validation

For each experiment type:
- **007 (Fast turn)**: Expects forward acceleration + gravity
- **016 (Straight cruise)**: Expects mainly gravity, minimal lateral acceleration
- **021 (Quarter turn)**: Expects centripetal acceleration during turn

### 3. Bias Estimation

- Uses first 30 seconds of static data
- Removes outliers (> 3σ)
- Calculates mean offset from expected values
- Provides bias corrections for both accelerometer and gyroscope

## Output

### Per Experiment
- `validation_results.yaml`: Detailed numerical results
- `VALIDATION_REPORT.md`: Human-readable summary
- Individual sensor plots:
  - Gravity alignment visualization
  - Transformation comparison
  - Bias estimation plots
  - Dynamic validation timeseries

### Overall Summary
- `all_validation_results.yaml`: Combined results
- `ORIENTATION_ANALYSIS_FINAL_REPORT.md`: Executive summary
- Summary visualizations:
  - Sensor coordinate systems
  - Validation status heatmap
  - Cross-sensor consistency

## Tolerances

- **Primary sensors** (3, 4, 5): ≤ 3° rotation error
- **Secondary sensors** (wb, wnb): ≤ 5° rotation error
- **Orthonormality**: ||R·R^T - I|| < 0.001
- **Cross-sensor consistency**: < 2° relative error

## Key Algorithms

### Gravity Direction Extraction
```python
# Average static accelerometer readings
mean_accel = np.mean(static_accel_data, axis=0)
# Normalize to get direction
gravity_direction = mean_accel / np.linalg.norm(mean_accel)
```

### Rotation Matrix Validation
```python
# Check orthonormality
identity_error = np.linalg.norm(R @ R.T - np.eye(3))
# Check determinant (should be +1)
is_valid = identity_error < tolerance and abs(det(R) - 1) < tolerance
```

### Bias Estimation
```python
# Transform to body frame
accel_body = R_bs @ accel_sensor
# Expected: gravity only
expected = [0, 0, 9.80665]
# Bias is the difference
bias_body = mean(accel_body) - expected
```

## Troubleshooting

### No Static Segments Found
- Check if data contains sufficient stationary periods
- Adjust `gyro_threshold_rad_s` in config if needed
- Verify sensor data is properly loaded

### High Rotation Errors
- Verify sensor mounting matches configuration
- Check for sensor damage or miscalibration
- Review gravity vector plots for anomalies

### Bias Estimation Failures
- Ensure at least 30 seconds of static data available
- Check for excessive sensor noise
- Verify rotation matrix is correct first

## Dependencies

- numpy
- pandas
- matplotlib
- pyyaml
- h5py
- tqdm

## Testing

Run unit tests:
```bash
python test_orientation.py
```

## Integration with Week 1 Pipeline

This module is designed to work with aligned data from the alignment_analysis module:

1. Run alignment first: `python ../alignment_analysis/run_alignment.py`
2. Run orientation: `python run_orientation.py`
3. Results feed into Week 2 Kalman filtering

## Notes

- Sensor_wnb may show poor results due to known timing issues
- GPS orientation is included but not validated (no IMU data)
- Magnetometer data is not used due to engine interference
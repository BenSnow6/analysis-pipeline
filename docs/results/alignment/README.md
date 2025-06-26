# Data Alignment Module

This module provides functionality to align multi-rate sensor data from hovercraft experiments to a common time base.

## Overview

The alignment module handles:
- 200 Hz IMU sensors (sensor_3, sensor_4, sensor_5)
- 100 Hz IMU sensor (sensor_wb) with 2:1 downsampling
- GPS data at 1 Hz (future implementation)
- Excludes sensor_wnb due to timing issues

## Key Features

- **Vectorized Processing**: Uses NumPy's `searchsorted` for efficient nearest-neighbor matching
- **Multi-rate Support**: Handles different sampling rates with appropriate tolerances
- **Cross-sensor Validation**: Ensures all sensors are aligned within 1ms tolerance
- **Performance**: Processes 5-minute datasets in under 1 second
- **HDF5 Output**: Saves aligned data in queryable pandas HDFStore format

## Installation

```bash
pip install numpy pandas pyyaml tqdm pytest
```

## Usage

### Command Line Interface

Process a single experiment:
```bash
python run_alignment.py -e 007_Fast_stbd_turn_1
```

Process multiple experiments:
```bash
python run_alignment.py -e 007_Fast_stbd_turn_1 016_Straight_cruise_1 021_Quarter_turn_port
```

Dry run to check performance:
```bash
python run_alignment.py -e 016_Straight_cruise_1 --dry-run
```

Specify output directory:
```bash
python run_alignment.py -e 007_Fast_stbd_turn_1 -o /path/to/output/
```

### Python API

```python
from align import DataAligner

# Initialize aligner
aligner = DataAligner()

# Load experiment data
sensor_data = aligner.load_experiment_data('007_Fast_stbd_turn_1', base_path)

# Align sensors
aligned_data = aligner.align_all_sensors(sensor_data)

# Save to HDF5
store = aligner.save_aligned_data(output_path)
```

## Configuration

The alignment parameters are defined in `alignment_config.yaml`:

- **reference_sensor**: `sensor_3` (200 Hz with zero jitter)
- **target_rate**: 200 Hz
- **tolerances**:
  - 200 Hz sensors: 2.5 ms
  - 100 Hz sensor: 5.0 ms
  - GPS (1 Hz): 20.0 ms (future)
- **max_cross_sensor_offset_ms**: 1.0 ms

## Output Format

Aligned data is saved as HDF5 files with the following structure:

```
experiment_aligned.h5
├── sensor_3       # Reference sensor data
├── sensor_4       # Aligned sensor data with time_diff_ms column
├── sensor_5       # Aligned sensor data with time_diff_ms column
├── sensor_wb      # Aligned 100Hz data (downsampled 2:1)
└── metadata       # Alignment parameters and timestamp
```

Each aligned sensor DataFrame includes:
- Original sensor columns (time_from_sync, x, y, z, etc.)
- `aligned_time`: The reference timestamp this sample was aligned to
- `time_diff_ms`: Time difference between original and aligned timestamp

## Quality Checks

### Visual Verification

Use the debug notebook to verify alignment quality:
```bash
jupyter notebook debug_align.ipynb
```

The notebook provides:
- Time difference histograms for each sensor
- Alignment consistency over time
- Effective sample rate verification

### Unit Tests

Run the test suite:
```bash
pytest test_align.py -v
```

Tests include:
- Nearest-neighbor matching accuracy
- 2:1 downsampling for 100Hz sensor
- Cross-sensor drift detection
- Performance benchmarks
- Edge case handling

## Performance

Target performance metrics:
- < 1 second for 5-minute dataset (300 seconds × 200 Hz = 60,000 samples)
- < 2.5 ms alignment error for 200 Hz sensors
- < 1 ms cross-sensor offset

## Limitations

1. **sensor_wnb** is excluded due to 25% rate error and large jitter
2. GPS alignment is deferred to Phase 2 implementation
3. Gap handling and interpolation not yet implemented

## Troubleshooting

### "Reference sensor not found"
Ensure sensor_3 data exists in the experiment directory with a valid `time_from_sync` column.

### "Cross-sensor offset exceeds limit"
Check for timing drift between sensors. This may indicate hardware synchronization issues.

### Performance warnings
If alignment takes > 1 second, check:
- Data is on local disk (not network drive)
- No other intensive processes running
- Dataset size is as expected

## Future Enhancements

- [ ] GPS alignment with interpolation
- [ ] Gap detection and repair
- [ ] Support for variable sampling rates
- [ ] Real-time alignment mode
- [ ] Parallel processing for multiple experiments
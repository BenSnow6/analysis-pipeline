# RPM Estimation from IMU Vibration Data

This module implements engine RPM estimation from hovercraft IMU vibration data using spectral analysis techniques.

## Overview

The RPM estimation pipeline extracts engine speed from accelerometer vibration signatures using:
- **Welch PSD**: For steady-state RPM estimation with high frequency resolution (WP-2 ✅)
- **STFT**: For transient analysis during RPM sweeps with 4 Hz update rate (WP-3 ✅)
- **Multi-sensor fusion**: SNR-based sensor selection and confidence gating (WP-4 🚧)

## Current Status

- ✅ **WP-0**: Repository scaffold complete
- ✅ **WP-1**: Raw data audit & orientation complete
- ✅ **WP-2**: Welch PSD core implementation complete
- ✅ **WP-3**: STFT + order tracking complete (2025-06-20)
- 🚧 **WP-4**: Multi-sensor fusion in progress (2025-06-20)
- ⏳ **WP-5**: Validation & blind test
- ⏳ **WP-6**: Batch processing

## Quick Start

```bash
# WP-2: Welch PSD for steady-state analysis
python -m rpm_estimation.cli --wp 2 --exp 016_Straight_cruise_1 --session afternoon

# WP-3: STFT for transient analysis
python -m rpm_estimation.cli --wp 3 --exp 026_Engine_rpm_sweep --session afternoon --plot

# WP-4: Multi-sensor fusion (NEW!)
python -m rpm_estimation.cli --wp 4 --exp 026_Engine_rpm_sweep --session afternoon --plot

# Process all experiments with fusion
python -m rpm_estimation.cli --wp 4 --all --session afternoon
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests to verify installation
pytest tests/
```

## Configuration

See `rpm_config.yaml` for all tunable parameters:
- Sampling rate: 200 Hz
- High-pass filter: 5 Hz cutoff
- Welch window: 6 seconds with 50% overlap
- SNR threshold: 10 dB

## Data Format

Expects aligned CSV data from the orientation analysis pipeline with columns:
- `t`: timestamp
- `x`, `y`, `z`: accelerations in m/s²
- `gyro_x`, `gyro_y`, `gyro_z`: angular velocities in rad/s

## Module Structure

- `io.py`: Data loading and file I/O operations
- `preprocess.py`: Filtering and signal conditioning
- `spectral.py`: Welch PSD and STFT implementations
- `tracking.py`: RPM tracking data structures
- `fusion.py`: Multi-sensor fusion algorithms
- `cli.py`: Command-line interface

## Development Status

See `DEVELOPMENT_CHECKLIST.md` for current progress on all work packages.

## Testing

Run the test suite:
```bash
pytest tests/ -v           # Run all tests
pytest tests/ -v --cov=.   # With coverage report
```

## Contributing

1. Follow the existing code style
2. Add tests for new functionality
3. Update DEVELOPMENT_CHECKLIST.md
4. Ensure all tests pass before committing

## References

Based on the expert roadmap for RPM estimation from vibration data, incorporating:
- Welch PSD for steady-state analysis
- STFT for transient tracking
- SNR-based confidence metrics
- Multi-sensor fusion strategies
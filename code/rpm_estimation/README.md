# RPM Estimation from IMU Vibration Data

This module implements engine RPM estimation from hovercraft IMU vibration data using spectral analysis techniques.

## Overview

The RPM estimation pipeline extracts engine speed from accelerometer vibration signatures using:
- **Welch PSD**: For steady-state RPM estimation with high frequency resolution
- **STFT**: For transient analysis during RPM sweeps
- **Multi-sensor fusion**: SNR-based sensor selection and confidence gating

## Quick Start

```bash
# Estimate RPM for engine sweep experiment
python -m rpm_estimation.cli --exp 026_Engine_rpm_sweep --session afternoon --method welch

# Run with custom config
python -m rpm_estimation.cli --exp 007_Fast_stbd_turn_1 --session afternoon --config my_config.yaml
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
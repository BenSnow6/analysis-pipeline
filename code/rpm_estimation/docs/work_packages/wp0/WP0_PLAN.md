# WP-0 Implementation Plan: Repository & Config Scaffold for RPM Estimation

This document captures the complete implementation plan for Work Package 0 of the RPM estimation project.

## Overview

WP-0 establishes the foundational repository structure, configuration system, and testing framework for the RPM estimation module. This work package ensures all subsequent development has a solid, well-tested base.

## Directory Structure

```
/code/rpm_estimation/
├── __init__.py
├── io.py           # Data loading and file I/O operations
├── preprocess.py   # Pre-processing operations (filtering, detrending)
├── spectral.py     # Spectral analysis (Welch PSD, STFT)
├── tracking.py     # RPM tracking and data structures
├── fusion.py       # Multi-sensor fusion logic
├── cli.py          # Command-line interface with argparse skeleton
├── rpm_config.yaml # Configuration file with exact schema
├── requirements.txt # Package dependencies
├── pytest.ini      # Test configuration
├── README.md       # Module overview and usage guide
├── DEVELOPMENT_CHECKLIST.md  # Progress tracking for all WPs
├── WP0_PLAN.md     # This implementation plan for reference
├── .github/
│   └── workflows/
│       └── test.yml # CI workflow for automated testing
└── tests/
    ├── __init__.py
    ├── test_dataclass.py
    ├── test_config.py
    ├── test_io.py
    └── test_imports.py  # Smoke test for module imports
```

## Key Components

### 1. Configuration Schema (rpm_config.yaml)

```yaml
# RPM Estimation Configuration
fs: 200  # Sampling frequency in Hz

# High-pass filter parameters
hp_cutoff: 5  # Hz - remove quasi-static motion

# Welch PSD parameters
welch:
  win_sec: 6      # Window length in seconds
  overlap: 0.5    # Overlap fraction (0-1)
  
# STFT parameters  
stft:
  win_sec: 1.0    # Window length in seconds
  hop_sec: 0.25   # Hop size in seconds
  
# SNR threshold for confidence gating
snr_thresh_db: 10  # dB - threshold for valid estimates

# Anti-aliasing filter parameters
anti_alias:
  cutoff_hz: 85     # Hz - low-pass cutoff
  order: 4          # Filter order
  type: "butterworth"
```

### 2. RPMFrame Dataclass

The core data structure for storing RPM estimates with metadata:

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class RPMFrame:
    time: float
    rpm: float
    snr_db: float
    sensor_id: str
    method: Literal['welch', 'stft', 'order_tracking']
    
    def is_valid(self, snr_threshold: float = 10.0) -> bool:
        """Check if estimate meets confidence threshold"""
        return self.snr_db >= snr_threshold
```

### 3. CLI Interface

Command-line interface with full argument parsing:

```bash
# Basic usage
python -m rpm_estimation.cli --exp 026_Engine_rpm_sweep --session afternoon --method welch

# With custom config
python -m rpm_estimation.cli --exp 007_Fast_stbd_turn_1 --session afternoon --config my_config.yaml

# Debug mode
python -m rpm_estimation.cli --exp 016_Straight_cruise_1 --session afternoon --debug
```

### 4. Test Suite

Four comprehensive unit tests ensure robustness:

1. **test_config.py**: Configuration loading and round-trip persistence
2. **test_dataclass.py**: RPMFrame instantiation and validation
3. **test_io.py**: File I/O operations and path handling
4. **test_imports.py**: Smoke test for all module imports

### 5. CI/CD Pipeline

GitHub Actions workflow for automated testing on every commit:
- Python 3.9 environment
- Dependency installation
- Full test suite execution with coverage
- CLI smoke test

## Integration Points

### Data Sources
- Aligned CSV data from `/hovercraft_data_analysis/alignment_analysis/aligned_data/`
- Validated rotation matrices from `orientation_config.yaml`
- Sensor data format: time, x/y/z accelerations (m/s²), gyro data (rad/s)

### Key Considerations
1. **Sampling Rate**: 200 Hz confirmed from existing data
2. **Sensor Selection**: Focus on Sensors 3, 4, and wb (validated with <3.5° error)
3. **Vibration Environment**: High vibrations (2-11 rad/s) perfect for RPM extraction
4. **File Structure**: Follows existing project patterns for consistency

## Done Criteria

✓ Repository structure created with all modules  
✓ rpm_config.yaml with exact schema keys  
✓ RPMFrame dataclass defined with is_valid() method  
✓ CLI entry point with complete argument parser  
✓ Four unit tests (config, dataclass, io, imports)  
✓ CI workflow for automated testing  
✓ README.md with usage guide  
✓ DEVELOPMENT_CHECKLIST.md tracking all WPs  
✓ WP0_PLAN.md documenting this plan  
✓ All modules import without errors  
✓ `python -m rpm_estimation.cli --help` runs successfully  

## Next Steps

Upon completion of WP-0, the foundation is ready for:
- WP-1: Raw data audit and orientation
- WP-2: Welch PSD implementation
- WP-3: STFT and order tracking
- WP-4: Multi-sensor fusion
- WP-5: Validation framework
- WP-6: Batch processing

This modular approach ensures each work package builds on a solid, tested foundation.
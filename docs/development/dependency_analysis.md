# Python Dependency Analysis Report

## Summary

- Total Python files: 107
- Files with internal imports: 76
- Unique external packages: 39
- Unique internal modules: 55
- Files with file path references: 90
- Files with sys.path manipulations: 32
- Files with hardcoded paths: 4

## Directory Structure

### code/
- 38 files

### hovercraft_data_analysis/
- 51 files

### root/
- 11 files

### src/
- 3 files

### thesis_analysis/
- 4 files

## External Dependencies

### Core Python Libraries
- __future__
- argparse
- collections
- dataclasses
- datetime
- glob
- json
- logging
- os
- pathlib
- re
- shutil
- subprocess
- sys
- time
- typing
- unittest
- warnings

### Third-party Libraries
- dash
- h5py
- matplotlib.gridspec
- matplotlib.image
- matplotlib.patches
- matplotlib.pyplot
- numpy
- pandas
- plotly.express
- plotly.graph_objects
- pyarrow
- pyarrow.parquet
- pytest
- scipy
- scipy.fft
- scipy.interpolate
- scipy.ndimage
- scipy.signal
- scipy.spatial.distance
- seaborn
- yaml

## Internal Module Structure

### code
- code.rpm_estimation.cli

### concurrent
- concurrent.futures

### hovercraft_data_analysis
- hovercraft_data_analysis.timestamp_analysis
- hovercraft_data_analysis.timestamp_analysis.main

### mpl_toolkits
- mpl_toolkits.mplot3d

### plotting
- plotting.experiment_plots

### rpm_estimation
- rpm_estimation.cli
- rpm_estimation.fusion
- rpm_estimation.io
- rpm_estimation.preprocess
- rpm_estimation.quality
- rpm_estimation.schema
- rpm_estimation.spectral
- rpm_estimation.tracking

### Root-level Modules
- align
- align_additional_data
- ast
- base64
- bias_estimator
- callbacks
- classes
- cli
- config
- csv
- dash_bootstrap_components
- data_loader
- data_utils
- dynamic_validator
- folium
- frame_definitions
- fusion
- io
- layout
- logging_config
- main
- math
- orientation_check
- plot_orientation
- preprocess
- quality
- rotation_validator
- rpm_estimation
- schema
- spectral
- static_detector
- statistics
- tempfile
- timestamp_analyzer
- tqdm
- traceback
- tracking
- visualizer
- wp2_process
- wp3_process
- wp4_process

## Critical Issues for Repository Reorganization

### Files with Hardcoded Paths

**hovercraft_data_analysis/dashboard_app/config.py**
- `../../02_Evaluation_Experiments`

**hovercraft_data_analysis/orientation_analysis/analyze_static_gyro_simple.py**
- `../../all_expts/afternoon/Experiments/010_Waiting_for_static_turns/IMU/Sensor_3/gyro_010_Waiting_for_static_turns.csv`
- `../../all_expts/afternoon/Experiments/011_Static_stbd_1/IMU/Sensor_3/gyro_011_Static_stbd_1.csv`
- `../../all_expts/afternoon/Experiments/012_Static_port_1/IMU/Sensor_3/gyro_012_Static_port_1.csv`

**hovercraft_data_analysis/orientation_analysis/check_gyro_units.py**
- `../../all_expts/afternoon/Experiments/010_Waiting_for_static_turns/IMU/Sensor_3/gyro_010_Waiting_for_static_turns.csv`
- `../../all_expts/afternoon/Experiments/011_Static_stbd_1/IMU/Sensor_3/gyro_011_Static_stbd_1.csv`
- `../../all_expts/afternoon/Experiments/012_Static_port_1/IMU/Sensor_3/gyro_012_Static_port_1.csv`

**thesis_analysis/scripts/simulator_validation.py**
- `/path/to/real/experiment/data`
- `/path/to/simulator/output/data`
- `/path/to/validation/results`

### Files with sys.path Manipulations
These files modify Python's import path and may break if moved:

- **code/rpm_estimation/cli.py**: 3 manipulation(s)
- **code/rpm_estimation/generate_wp3_test_plots.py**: 1 manipulation(s)
- **code/rpm_estimation/results/wp1/check_parquet.py**: 1 manipulation(s)
- **code/rpm_estimation/results/wp1/run_wp1.py**: 1 manipulation(s)
- **code/rpm_estimation/run_wp2_tests.py**: 1 manipulation(s)
- **code/rpm_estimation/test_wp3_run.py**: 1 manipulation(s)
- **code/rpm_estimation/test_wp4_integration.py**: 1 manipulation(s)
- **code/rpm_estimation/tests/test_cli.py**: 1 manipulation(s)
- **code/rpm_estimation/tests/test_fusion.py**: 1 manipulation(s)
- **code/rpm_estimation/tests/test_preprocessing.py**: 1 manipulation(s)
- **code/rpm_estimation/tests/test_quality.py**: 1 manipulation(s)
- **code/rpm_estimation/tests/test_schema.py**: 1 manipulation(s)
- **code/rpm_estimation/tests/test_stft.py**: 1 manipulation(s)
- **code/rpm_estimation/validate_wp2.py**: 1 manipulation(s)
- **code/rpm_estimation/visualize_unit_tests.py**: 1 manipulation(s)
- **code/rpm_estimation/wp2_process.py**: 1 manipulation(s)
- **hovercraft_data_analysis/alignment_analysis/align_additional_all.py**: 1 manipulation(s)
- **hovercraft_data_analysis/orientation_analysis/bias_estimator.py**: 1 manipulation(s)
- **hovercraft_data_analysis/orientation_analysis/check_sensor3_orientation.py**: 1 manipulation(s)
- **hovercraft_data_analysis/orientation_analysis/check_sensor5_orientation.py**: 1 manipulation(s)
- **hovercraft_data_analysis/orientation_analysis/debug_rotation_validation.py**: 1 manipulation(s)
- **hovercraft_data_analysis/orientation_analysis/deduce_sensor_orientation.py**: 1 manipulation(s)
- **hovercraft_data_analysis/orientation_analysis/dynamic_validator.py**: 1 manipulation(s)
- **hovercraft_data_analysis/orientation_analysis/orientation_check.py**: 1 manipulation(s)
- **hovercraft_data_analysis/orientation_analysis/rotation_validator.py**: 1 manipulation(s)
- **hovercraft_data_analysis/orientation_analysis/test_orientation_simple.py**: 1 manipulation(s)
- **hovercraft_data_analysis/orientation_analysis/test_unit_conversion.py**: 1 manipulation(s)
- **hovercraft_data_analysis/process_all_experiments.py**: 1 manipulation(s)
- **hovercraft_data_analysis/run_static_orientation_analysis.py**: 1 manipulation(s)
- **run_timestamp_analysis_standalone.py**: 1 manipulation(s)
- **test_timestamp_analysis.py**: 1 manipulation(s)
- **thesis_analysis/scripts/analyze_experiments.py**: 1 manipulation(s)

## Key Import Relationships

### Most Imported Internal Modules
- `tracking`: imported by 12 file(s)
- `orientation_check`: imported by 10 file(s)
- `csv`: imported by 9 file(s)
- `logging_config`: imported by 8 file(s)
- `spectral`: imported by 8 file(s)
- `frame_definitions`: imported by 7 file(s)
- `io`: imported by 6 file(s)
- `traceback`: imported by 6 file(s)
- `static_detector`: imported by 6 file(s)
- `tempfile`: imported by 5 file(s)

### Potential Circular Dependencies
None detected.

## Recommendations for Safe Reorganization

1. **Fix hardcoded paths**: Convert absolute paths to relative paths or use configuration files
2. **Update sys.path manipulations**: Use proper package structure with __init__.py files
3. **Create proper packages**: Add __init__.py files to directories that are imported as modules
4. **Use relative imports**: Within packages, use relative imports (e.g., `from . import module`)
5. **Configuration files**: Move file paths to configuration files (JSON/YAML) for easy updates

## Suggested File Organization

Based on the analysis, here's a suggested organization:

```
analysis-pipeline/
├── setup.py
├── requirements.txt
├── config/
│   └── paths.yaml  # Centralized path configuration
├── src/
│   ├── __init__.py
│   ├── core/  # Core utilities
│   │   ├── __init__.py
│   │   ├── data_utils.py
│   │   ├── frame_definitions.py
│   │   └── classes.py
│   ├── rpm_estimation/
│   │   ├── __init__.py
│   │   └── ... (existing structure)
│   ├── hovercraft_analysis/
│   │   ├── __init__.py
│   │   ├── alignment/
│   │   ├── orientation/
│   │   └── timestamp/
│   └── visualization/
│       ├── __init__.py
│       ├── dashboard/
│       └── plotting/
├── tests/
├── scripts/  # Standalone scripts
└── notebooks/  # Jupyter notebooks
```

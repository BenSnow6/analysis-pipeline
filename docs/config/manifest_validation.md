# Experiment Manifest Validation Framework

This document describes the validation framework for the `experiment_manifest.yaml` file, which serves as the central source of truth for all experimental data in the project.

## Overview

The validation framework provides comprehensive checking of:
1. **YAML Structure** - Validates the internal consistency of the manifest file
2. **Data Integrity** - Ensures required fields, valid values, and no duplicates
3. **Path Consistency** - Verifies path fields are consistent
4. **Cross-References** - Validates references between sections
5. **Filesystem Integrity** - Checks that referenced directories and data files actually exist

## Components

### Core Modules

- `src/core/experiment_manifest.py` - Helper functions for loading and querying the manifest
- `src/core/validation_report.py` - Comprehensive reporting system for validation results
- `src/scripts/validate_manifest.py` - Command-line tool for running validation

### Test Suite

- `tests/config/test_experiment_manifest_unit.py` - Unit tests (no filesystem access)
- `tests/config/test_experiment_manifest_integration.py` - Integration tests (filesystem validation)
- `tests/config/test_experiment_manifest_helpers.py` - Tests for helper functions

## Usage

### Command-Line Validation

Run the validation script from the project root:

```bash
# Basic validation with markdown output
python -m src.scripts.validate_manifest

# Save report to file
python -m src.scripts.validate_manifest --output validation_report.md

# Generate JSON report
python -m src.scripts.validate_manifest --format json --output report.json

# Skip filesystem checks (faster, for CI)
python -m src.scripts.validate_manifest --no-filesystem

# Validate a different manifest
python -m src.scripts.validate_manifest --manifest path/to/other/manifest.yaml
```

### Running Tests

```bash
# Run all tests
pytest tests/config -v

# Run only unit tests (fast, no filesystem access)
pytest tests/config -m unit

# Run only integration tests
pytest tests/config -m integration

# Run with coverage
pytest tests/config --cov=src.core.experiment_manifest --cov=src.core.validation_report
```

### Programmatic Usage

```python
from pathlib import Path
from src.core.experiment_manifest import load_manifest, validate_manifest_structure
from src.core.validation_report import validate_manifest_comprehensive

# Load and validate manifest
manifest_path = Path("config/experiments/experiment_manifest.yaml")
manifest = load_manifest(manifest_path)

# Run structural validation only
errors = validate_manifest_structure(manifest)
for error in errors:
    print(f"{error.severity}: {error.message}")

# Run comprehensive validation
report = validate_manifest_comprehensive(manifest_path)
print(report.to_markdown())

# Query experiments
all_experiments = manifest.get_all_experiments()
morning_exp = manifest.get_experiment_by_name("006_Departure", session="morning")
category_exps = manifest.get_experiments_by_category("1b_4_Normal_Take_off")
```

## Validation Checks

### 1. Structure Validation

- **Top-level keys**: Ensures required sections exist
- **Session subsections**: Checks for morning/afternoon divisions
- **Required fields**: Validates each experiment has required fields

### 2. Data Integrity

- **Unique names**: No duplicate experiment names within sections
- **Valid types**: Experiment type must be 'static' or 'dynamic'
- **Valid data types**: Sensor names must be from allowed list
- **Boolean fields**: Ensures boolean fields are actually boolean

### 3. Path Consistency

- **Path matching**: Validates consistency between `path` and `paths.full_path`
- **Name in path**: Checks if experiment name appears in paths
- **Relative path resolution**: Ensures relative paths can be resolved

### 4. Cross-Reference Validation

- **Analysis config**: Validates experiments referenced in `analysis_config` exist
- **Duplicate consistency**: Ensures duplicated entries have identical data

### 5. Filesystem Validation

- **Directory existence**: All experiment directories must exist
- **Data folders**: Expected GPS/IMU folders must exist
- **Non-empty data**: Warns about empty data directories
- **Orphan detection**: Finds directories not listed in manifest

## Error Severity Levels

- **ERROR**: Critical issues that must be fixed (e.g., missing directories)
- **WARNING**: Important issues to review (e.g., missing data folders)
- **INFO**: Informational messages (e.g., naming conventions)

## Example Validation Report

```markdown
# Experiment Manifest Validation Report

**Date:** 2025-06-24 10:30:00
**Manifest:** `config/experiments/experiment_manifest.yaml`

## Summary
- Total experiments defined: 45
- Experiments checked: 45
- **Errors:** 2
- **Warnings:** 5
- **Info:** 10

### Filesystem Check
- Experiment directories found: 43
- Experiment directories missing: 2
- Data folders found: 256
- Data folders missing: 5
- Orphan directories: 3

## Critical Errors
These issues must be fixed:

### Experiment directory not found: 016_Plough_in
- **experiment:** 016_Plough_in
- **session:** morning
- **expected_path:** /data/raw/morning/Experiments/016_Plough_in

## Warnings
These issues should be reviewed:

### Filesystem Issues
- Missing data folder for 006_Departure
- Missing data folder for 007_Fast_stbd_turn_1
...

## Orphan Directories
These directories exist on filesystem but are not in the manifest:

### Morning Session
- 099_Test_Obsolete
- 100_Debug_Run
```

## Best Practices

1. **Regular Validation**: Run validation after any manifest changes
2. **CI Integration**: Include unit tests in CI pipeline
3. **Pre-commit Hook**: Validate manifest before committing changes
4. **Documentation**: Keep experiment descriptions up to date
5. **Cleanup**: Remove or document orphan directories

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure package is installed: `pip install -e .`
   - Run from project root directory

2. **Filesystem Tests Fail**
   - Check data directory paths are correct
   - Ensure you have read access to data directories
   - Use `--no-filesystem` flag if data is not available

3. **Slow Performance**
   - Use unit tests for quick validation
   - Run filesystem checks separately
   - Consider caching for large datasets

## Current Validation Findings (2025-06-24)

Running the validation framework on the current repository reveals several critical issues that need to be addressed:

### 1. Inconsistent Data Directory Structure

The most significant finding is that the experimental data is stored inconsistently:

**Morning Experiments Structure:**
```
morning/Experiments/006_Departure/
├── GPS/
├── Sensor_3/      # Sensors directly under experiment
├── Sensor_4/
├── Sensor_5/
├── Sensor_wb/
└── Sensor_wnb/
```

**Afternoon Experiments Structure:**
```
afternoon/Experiments/007_Fast_stbd_turn_1/
├── GPS/
└── IMU/           # Sensors under IMU subdirectory
    ├── Sensor_3/
    ├── Sensor_4/
    ├── Sensor_5/
    ├── Sensor_wb/
    └── Sensor_wnb/
```

**Expected Structure (per tests):**
All experiments should follow the afternoon pattern with sensors under an IMU subdirectory for consistency.

### 2. Manifest Data Quality Issues

The `experiment_manifest.yaml` has several data quality problems:

#### Invalid Experiment Types
- Using `'mixed'`, `'static_turn'`, `'unknown'` instead of the valid types: `'static'` or `'dynamic'`
- Missing `type` field entirely in some experiments

#### Examples of Invalid Types:
```yaml
- name: 014_Floating_on_sea_and_takeoff
  type: mixed  # Should be 'dynamic'
  
- name: 011_Static_stbd_1
  type: static_turn  # Should be 'static'
  
- name: 001_Synchronisation
  type: unknown  # Should be 'static'
```

#### Path Issues
- Manifest uses absolute paths (`/data/raw/...`) instead of relative paths
- This causes portability issues between different environments

### 3. Missing Sensor Data

The manifest claims all experiments have all 5 sensors (Sensor_3, 4, 5, wb, wnb), but filesystem validation shows:
- Many experiments are missing some or all sensor data
- 95 missing sensor directories across 31 experiments
- This indicates either:
  - The manifest is incorrect about which sensors were used
  - Data was lost or never collected
  - Data needs to be reorganized

### 4. Orphan Directories

16 directories exist on the filesystem but are not documented in the manifest:
- These may be test runs, calibration data, or obsolete experiments
- They should be either added to the manifest or removed

## Recommended Actions

### 1. Standardize Directory Structure
Choose one consistent structure for all experiments. The recommended structure is:
```
experiment_name/
├── GPS/
│   └── *.csv files
└── IMU/
    ├── Sensor_3/
    │   └── *.csv files
    ├── Sensor_4/
    │   └── *.csv files
    ├── Sensor_5/
    │   └── *.csv files
    ├── Sensor_wb/
    │   └── *.csv files
    └── Sensor_wnb/
        └── *.csv files
```

### 2. Fix Manifest Data Quality
1. Update all experiment types to use only 'static' or 'dynamic'
2. Add missing 'type' fields
3. Update 'data_types' to accurately reflect which sensors actually have data
4. Convert absolute paths to relative paths

### 3. Data Migration Script
Create a script to:
1. Reorganize morning experiments to match the IMU subdirectory structure
2. Verify no data is lost during migration
3. Update manifest to reflect actual data availability
4. Document any missing data

### 4. Document Orphan Directories
Either:
- Add orphan directories to the manifest with proper metadata
- Move them to an archive location
- Delete if confirmed obsolete

## Running Validation

To see all current issues:
```bash
# Full validation with filesystem checks
python -m src.scripts.validate_manifest

# Save detailed report
python -m src.scripts.validate_manifest --output current_issues.md

# Check only manifest structure (fast)
python -m src.scripts.validate_manifest --no-filesystem
```

## Future Enhancements

- [ ] Validate CSV file contents and schemas
- [ ] Check for required metadata files
- [ ] Validate timestamp consistency across sensors
- [ ] Generate data availability matrix
- [ ] Auto-fix common issues (with confirmation)
- [ ] Create data migration script for standardization
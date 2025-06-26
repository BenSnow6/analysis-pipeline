# Analysis Pipeline Scripts Documentation

This directory contains documentation for the various utility and analysis scripts in the `/src/scripts/` directory.

## Available Scripts

### Analysis Scripts

#### RPM Estimation
- **[debug_interpolation.py](debug_interpolation_guide.md)** - Debug and visualize RPM interpolation for sensor fusion
- **preprocess_data.py** - Preprocess raw sensor data for RPM analysis
- **process_all_experiments.py** - Batch process multiple experiments

#### Timestamp Analysis
- **basic_timestamp_analysis.py** - Basic timestamp consistency checks
- **run_timestamp_analysis_standalone.py** - Standalone timestamp analysis runner

#### Orientation Analysis
- **run_static_orientation_analysis.py** - Analyze sensor orientations from static experiments
- **export_static_experiments_to_csv.py** - Export static experiment data to CSV format

#### Data Alignment
- **run_week1_complete.py** - Run complete Week 1 alignment workflow

### Utility Scripts

#### Data Management
- **data_sync.py** - Synchronize data between locations
- **data_utils.py** - Common data processing utilities
- **clean_raw_data_plots.py** - Clean up generated plot files
- **generate_data_plots.py** - Generate standard data visualizations

#### Repository Management
- **repo_tree.py** - Generate repository structure documentation
- **validate_structure.py** - Validate repository structure
- **update_old_paths.py** - Update old path references
- **validate_manifest.py** - Validate experiment manifest
- **update_experiment_manifest.py** - Update experiment manifest file

#### Migration Scripts
- **migrate_all_docs.py** - Migrate all documentation
- **migrate_configs.py** - Migrate configuration files
- **migrate_rpm_docs.py** - Migrate RPM-specific documentation

### Application Scripts

- **dashboard_app.py** - Launch the analysis dashboard web application
- **frame_definitions.py** - Define coordinate frames for analysis

## Script Categories

### 1. Analysis Scripts
These scripts perform specific analysis tasks on experimental data:
- Process sensor data
- Compute metrics
- Generate results

### 2. Utility Scripts
Helper scripts for data management and repository maintenance:
- File organization
- Path updates
- Validation checks

### 3. Migration Scripts
One-time scripts used during repository reorganization:
- Move files to new locations
- Update import statements
- Maintain compatibility

### 4. Application Scripts
Interactive tools and applications:
- Web dashboards
- Visualization tools
- Configuration interfaces

## Running Scripts

All scripts can be run from the repository root:

```bash
# Using Python directly
python src/scripts/script_name.py [arguments]

# Using module syntax (preferred)
python -m src.scripts.script_name [arguments]

# Some scripts are available as console commands after installation
pip install -e .
hovercraft-dashboard
```

## Common Patterns

### Configuration Loading
Most scripts load configuration from `/config/pipeline.yaml`:
```python
from src.core import get_config
config = get_config()
```

### Path Management
Scripts use centralized path definitions:
```python
from src.core.paths import DATA_DIR, get_experiment_path
exp_path = get_experiment_path("experiment_name", "session")
```

### Logging
Scripts use structured logging:
```python
import logging
logger = logging.getLogger(__name__)
```

## Adding New Scripts

When adding a new script:
1. Place it in `/src/scripts/`
2. Add appropriate documentation
3. Update this README
4. Add unit tests if applicable
5. Consider adding as console script in `pyproject.toml`

## Script Documentation Template

For each script, document:
- **Purpose**: What the script does
- **Usage**: Command line examples
- **Arguments**: Available options
- **Output**: What files/results are generated
- **Dependencies**: Required data or prior processing
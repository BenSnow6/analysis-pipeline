# Development Guidelines

This document contains critical information about working with this codebase. Follow these guidelines precisely.

## Repository Structure

After final consolidation (2025-06-24), the repository follows this structure:
```
analysis-pipeline/
├── /config            # Master configuration directory
│   ├── pipeline.yaml  # Master configuration file
│   ├── /experiments   # Experiment mappings and metadata
│   ├── /sensors       # Sensor specifications and orientations
│   └── /processing    # Module-specific processing configs
├── /src               # All source code (consolidated)
│   ├── /core          # Core utilities and config management
│   ├── /analysis      # Analysis modules
│   │   ├── /alignment # Time alignment
│   │   ├── /orientation # Sensor orientation
│   │   ├── /rpm       # RPM estimation
│   │   └── /timestamp # Timestamp analysis
│   ├── /apps          # Applications (dashboard)
│   ├── /scripts       # Command-line scripts
│   ├── /notebooks     # Jupyter notebooks
│   └── /plans         # Development plans
├── /data              # All experimental and processed data
│   ├── /raw           # Raw experimental data
│   │   ├── /morning/Experiments/
│   │   └── /afternoon/Experiments/
│   ├── /processed     # All processed outputs
│   │   ├── /aligned   # Time-aligned sensor data
│   │   ├── /orientation # Orientation analysis results
│   │   ├── /rpm       # RPM estimation results
│   │   └── /timestamp # Timestamp analysis results
│   └── /cache         # Temporary files and cache
├── /docs              # Documentation
│   ├── /config        # Config documentation only
│   ├── /results       # Analysis results by thesis WPs
│   │   ├── /raw_data_analysis
│   │   ├── /timestamp_analysis
│   │   ├── /alignment
│   │   ├── /orientation
│   │   ├── /attitude_estimation
│   │   ├── /steering
│   │   ├── /rpm_estimation
│   │   │   ├── /wp0_exploration
│   │   │   ├── /wp1_preprocessing
│   │   │   ├── /wp2_peak_detection
│   │   │   ├── /wp3_stft
│   │   │   ├── /wp4_fusion
│   │   │   └── /wp5-7_future
│   │   └── /validation
│   ├── /development   # Development docs
│   ├── /experimental_setup # Experiment documentation
│   └── /migration     # Migration history
├── /notes             # Thesis notes
└── /tests             # Centralized test suite
```

## Import Conventions

After final consolidation (2025-06-24), use these import patterns:
```python
# CORRECT (consolidated structure):
from src.analysis.alignment import align_data
from src.core import DATA_DIR, get_experiment_path
from src.analysis.orientation import analyze_gravity
from src.apps.dashboard import app
from src.analysis.rpm import RPMFrame

# INCORRECT (old patterns):
from hovercraft_analysis.xxx import ...  # NO: old package structure
from code.xxx import ...  # NO: code directory removed
sys.path.append('..')  # NO: use proper imports
```

## Package Installation

Install the package in development mode:
```bash
pip install -e .
```

This allows imports to work from anywhere without path manipulation.

## Configuration Management (Phase 3 - 2025-06-23)

The project now uses a unified configuration system with a master configuration file:

### Master Configuration
The master config file is located at `/config/pipeline.yaml` and provides:
- Environment variable substitution (`${VAR}` or `${VAR:-default}`)
- Internal reference resolution (`${paths.data_root}`)
- Environment-specific overrides (development/production/testing)
- Centralized path management
- Feature flags

### Using Configuration
```python
from hovercraft_analysis.core import get_config

# Get configuration manager
config = get_config()

# Access configuration values
project_name = config.get('project.name')
data_root = config.get('paths.data_root')

# Get paths as Path objects
data_path = config.get_path('paths.data_root')

# Load sub-configurations
rpm_config = config.load_sub_config('processing.rpm')
```

### Environment Variables
- `HOVERCRAFT_ENV`: Set to 'development', 'production', or 'testing'
- `PROJECT_ROOT`: Automatically set if not defined

### Legacy Compatibility
The new config system maintains backward compatibility:
```python
# These still work for legacy code
config.experiment_mapping  # Loads from new location
config.sensor_orientations
config.orientation_config
```

### Configuration Files
- `/config/pipeline.yaml` - Master configuration
- `/config/experiments/` - Experiment mappings and categories
- `/config/sensors/` - Sensor specs and orientations  
- `/config/processing/` - Module-specific configs (alignment, rpm, etc.)

## Path Management

Always use the centralized path configuration:
```python
from src.core.paths import (
    RAW_DATA_DIR,
    DATA_DIR, 
    get_experiment_path,
    MORNING_DATA_DIR,
    AFTERNOON_DATA_DIR,
    PROCESSED_DATA_DIR,
    ALIGNED_DATA_DIR,
    CACHE_DIR
)

# Get experiment path (automatically uses /data/raw structure)
exp_path = get_experiment_path("007_Fast_stbd_turn_1", "afternoon")

# Get aligned data path (uses /data/processed/aligned structure)
aligned_path = get_aligned_data_path("007_Fast_stbd_turn_1", "afternoon")

# Access processed data
orientation_results = PROCESSED_DATA_DIR / "orientation" / "validation_results"
rpm_results = PROCESSED_DATA_DIR / "rpm" / "wp1"
timestamp_results = PROCESSED_DATA_DIR / "timestamp" / "timestamp_analysis_results"

# Access documentation and reports
docs_results = DOCS_DIR / "results"  # Organized by thesis work packages

# Never hardcode paths like:
# "../../02_Evaluation_Experiments/..."  # This folder no longer exists
# "../code/alignment_analysis/aligned_data/"  # Old location - now in ALIGNED_DATA_DIR
# "data/morning/Experiments"  # Old structure - now "data/raw/morning/Experiments"
```

## Experiment Mapping

To understand experiment categorization, consult `experiment_mapping.json`:
```python
import json
with open('experiment_mapping.json', 'r') as f:
    mapping = json.load(f)
# Shows which experiments belong to which evaluation categories
```

## Core Development Rules

1. **Data Organization**
   - Raw data is in `/data/raw/morning/Experiments/` and `/data/raw/afternoon/Experiments/`
   - Aligned data outputs go to `/data/processed/aligned/`
   - Results and plots go to `/docs/results/`

2. **Code Quality**
   - Type hints required for all code
   - Public APIs must have docstrings
   - Functions must be focused and small
   - Follow existing patterns in each module
   - Use verbose imports for AI readability

3. **Testing Requirements**
   - Test critical workflows after any changes
   - Key components to test:
     - Dashboard app: `python3 code/scripts/dashboard_app.py`
     - Alignment analysis scripts
     - RPM estimation workflows
   - Fix import errors before functionality testing

## Common Tasks

### Running Analysis Scripts

After installing the package with `pip install -e .`, you can use command-line tools:
```bash
# Using installed scripts
hovercraft-align --experiment 007_Fast_stbd_turn_1
hovercraft-dashboard
hovercraft-timestamp --experiment 016_Straight_cruise_1
hovercraft-orientation --experiment 011_Static_stbd_1

# Or run directly as modules
python -m src.scripts.run_alignment
python -m src.scripts.dashboard_app
```

### Accessing Experiment Data
```python
from src.core.paths import get_experiment_path
from src.core.io import load_experiment_data
import pandas as pd

# Load all data for an experiment
data = load_experiment_data("016_Straight_cruise_1", "afternoon")
gps_data = data['gps']
accel_data = data['Sensor_3_accel']

# Or load individual files
exp_path = get_experiment_path("016_Straight_cruise_1", "afternoon")
gps_data = pd.read_csv(exp_path / "GPS" / "GPS_016_Straight_cruise_1.csv")
```

## Migration Status

**Config Cleanup Completed: 2025-06-26**
- ✅ Removed duplicate config files from `/src/core/` 
- ✅ Config files now centralized in `/config/` only
- ✅ Updated all imports to use centralized config paths
- ✅ Moved timestamp_analysis_results from `/docs/` to `/data/processed/timestamp/`
- ✅ Reorganized `/docs/results/` by thesis work packages
- ✅ Removed empty sensor location files
- ✅ Removed auto-generated `/docs/api/` folder

**Final Consolidation Completed: 2025-06-24**
- ✅ Flattened `/src/hovercraft_analysis/` to `/src/`
- ✅ Updated all imports from `hovercraft_analysis.` to `src.`
- ✅ Migrated all unique content from `/code/` to `/src/`
- ✅ Updated all imports from `code.` to `src.`
- ✅ Updated pyproject.toml for new structure
- ✅ Removed `/code/` directory entirely
- ✅ All code now consolidated under `/src/`

**Previous Migration Phases:**
- Phase 1-4 (2025-06-23): Initial reorganization and package structure
- Phase 5 (2025-06-24): Final consolidation to single `/src/` directory

The codebase now has a clean, consolidated structure:
- All source code is under `/src/`
- No more `hovercraft_analysis` package level
- No more `/code/` directory
- All imports use `from src.xxx import yyy` pattern
- Configuration remains in `/config/`
- Data remains in `/data/`

### Test Results (Post-Migration)
Tests run with venv environment activated:
- **Alignment tests**: 8/9 pass (1 minor assertion issue)
- **Orientation tests**: 9/11 pass (2 edge case failures)
- **Import verification**: All migration changes work correctly
  - `code.config.paths` imports ✓
  - `code.dashboard_app.config` imports ✓
  - `code.alignment_analysis.align` imports ✓
  - `code.scripts.frame_definitions` imports ✓

Note: All tests should now work with the consolidated structure using `pytest` from the repository root.

## Git Commit Guidelines

- For commits fixing bugs or adding features based on user reports add:
  ```bash
  git commit --trailer "Reported-by:<name>"
  ```

- For commits related to a Github issue, add:
  ```bash
  git commit --trailer "Github-Issue:#<number>"
  ```

- NEVER mention `co-authored-by` or the tool used to create the commit

## Testing Checklist

Before committing changes:
1. ✓ All imports updated to new structure
2. ✓ No hardcoded paths (use `code.config.paths`)
3. ✓ Dashboard app loads without import errors
4. ✓ Key analysis scripts run successfully
5. ✓ No references to deleted directories (`02_Evaluation_Experiments`, `src`, old `hovercraft_data_analysis`)

## Key Files Reference

- **Path Configuration**: `/code/config/paths.py`
- **Experiment Mapping**: `/experiment_mapping.json`
- **Sensor Configurations**: `/code/config/sensor_orientations.json`
- **Dashboard Entry**: `/code/scripts/dashboard_app.py`
- **Main Analysis Modules**: 
  - `/code/alignment_analysis/`
  - `/code/orientation_analysis/`
  - `/code/rpm_estimation/`
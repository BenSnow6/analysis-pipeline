# Repository Reorganization Summary

## Date: 2025-06-23

## What Was Done

### 1. Created Experiment Mapping
- Created `experiment_mapping.json` that documents which experiments belong to which evaluation categories
- Preserves the knowledge of how experiments were organized in `02_Evaluation_Experiments`

### 2. New Directory Structure
```
analysis-pipeline/
├── /code              # All Python code
│   ├── /alignment_analysis
│   ├── /dashboard_app
│   ├── /orientation_analysis
│   ├── /timestamp_analysis
│   ├── /rpm_estimation
│   ├── /plans
│   ├── /data_repository
│   ├── /scripts       # Standalone scripts
│   ├── /notebooks     # Jupyter notebooks
│   └── /config        # Configuration files
├── /data              # All experimental data
│   ├── /morning
│   └── /afternoon
├── /docs              # Documentation
│   ├── /results
│   ├── /timestamp_analysis_results
│   ├── /codebase_analysis
│   └── /Experimental setup
├── /notes             # Thesis notes (existing)
└── /venv              # Virtual environment (existing)
```

### 3. Files Moved
- All folders from `hovercraft_data_analysis/` → `/code/`
- Python files from `src/` → `/code/`
- All data from `all_expts/` → `/data/`
- All documentation → `/docs/`
- All notebooks → `/code/notebooks/`
- All root Python scripts → `/code/scripts/`
- All config files → `/code/config/`

### 4. Created Path Configuration
- Created `/code/config/paths.py` for centralized path management
- Provides helper functions for accessing experiments

### 5. Deleted
- `02_Evaluation_Experiments/` folder (data preserved in `/data/`)
- Empty directories: `all_expts/`, `src/`, `hovercraft_data_analysis/`, `notebooks/`

## Next Steps

### Import Updates Required
Many files still use old import paths. You'll need to update:
- Replace `from src.` with `from code.`
- Replace `from hovercraft_data_analysis.` with `from code.`
- Remove `sys.path.append()` calls
- Update hardcoded paths to use `from code.config.paths import ...`

### Testing Required
1. Dashboard app needs import updates
2. Alignment analysis scripts need testing
3. RPM estimation workflows need verification

### Example Import Updates

**Before:**
```python
from src.data_processing import process_data
from hovercraft_data_analysis.alignment import align
sys.path.append('..')
```

**After:**
```python
from code.data_processing import process_data
from code.alignment_analysis.align import align
from code.config.paths import DATA_DIR, get_experiment_path
```

## Benefits
- Clear separation of code, data, and documentation
- Consistent structure following Python best practices
- Easier to understand and navigate
- Path configuration centralizes all directory references
- Experiment mapping preserves categorization knowledge
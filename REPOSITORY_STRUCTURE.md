# Repository Structure Guide

## Quick Reference
- **Source Code**: `/src/` - All Python modules and scripts
- **Data**: `/data/` - Raw and processed experimental data  
- **Config**: `/config/` - All configuration files
- **Documentation**: `/docs/` - All documentation and results
- **Tests**: `/tests/` - Test suite
- **Notes**: `/notes/` - Thesis notes and planning

## Directory Structure

```
analysis-pipeline/
├── /config                    # Master configuration directory
│   ├── pipeline.yaml         # Master configuration file
│   ├── /experiments          # Experiment mappings and metadata
│   │   ├── experiment_categories.yaml
│   │   ├── experiment_manifest.yaml
│   │   └── experiment_mapping.json
│   ├── /sensors             # Sensor specifications and orientations
│   │   ├── sensor_orientations.json
│   │   └── sensor_specs.yaml
│   └── /processing          # Module-specific processing configs
│       ├── alignment_config.yaml
│       ├── orientation_config.yaml
│       ├── rpm_config.yaml
│       └── timestamp_config.yaml
│
├── /src                      # All source code (consolidated)
│   ├── /core                # Core utilities and config management
│   │   ├── __init__.py
│   │   ├── config.py        # Configuration management
│   │   ├── io.py           # Data I/O utilities
│   │   ├── paths.py        # Path constants and utilities
│   │   └── utils.py        # General utilities
│   ├── /analysis            # Analysis modules
│   │   ├── /alignment       # Time alignment
│   │   │   ├── __init__.py
│   │   │   ├── align.py
│   │   │   └── export.py
│   │   ├── /orientation     # Sensor orientation
│   │   │   ├── __init__.py
│   │   │   ├── core.py
│   │   │   ├── plotting.py
│   │   │   └── validation.py
│   │   ├── /rpm            # RPM estimation
│   │   │   ├── __init__.py
│   │   │   ├── fusion.py
│   │   │   ├── models.py
│   │   │   ├── peak_detection.py
│   │   │   ├── preprocess.py
│   │   │   ├── spectral.py
│   │   │   └── visualize.py
│   │   └── /timestamp      # Timestamp analysis
│   │       ├── __init__.py
│   │       └── analyze.py
│   ├── /apps               # Applications
│   │   └── /dashboard      # Dashboard app modules
│   ├── /scripts            # Command-line scripts
│   │   ├── dashboard_app.py
│   │   ├── run_alignment.py
│   │   ├── run_orientation.py
│   │   └── run_timestamp_analysis.py
│   ├── /notebooks          # Jupyter notebooks
│   └── /plans              # Development plans
│
├── /data                     # All experimental and processed data
│   ├── /raw                 # Raw experimental data
│   │   ├── /morning/Experiments/  # Morning session experiments
│   │   └── /afternoon/Experiments/ # Afternoon session experiments
│   ├── /processed           # All processed outputs
│   │   ├── /aligned        # Time-aligned sensor data
│   │   │   ├── /morning    # Aligned morning experiments
│   │   │   ├── /afternoon  # Aligned afternoon experiments
│   │   │   └── /static     # Static experiment alignments
│   │   ├── /orientation    # Orientation analysis results
│   │   │   └── /validation_results
│   │   ├── /rpm           # RPM estimation results
│   │   │   ├── /wp1       # Preprocessing outputs
│   │   │   ├── /wp2       # Peak detection results
│   │   │   ├── /wp3       # STFT analysis results
│   │   │   └── /wp4       # Fusion results
│   │   └── /timestamp     # Timestamp analysis results
│   │       └── /timestamp_analysis_results
│   └── /cache              # Temporary files and cache
│
├── /docs                     # Documentation
│   ├── /config              # Config documentation only
│   ├── /results             # Analysis results by thesis WPs
│   │   ├── /raw_data_analysis
│   │   ├── /timestamp_analysis
│   │   ├── /alignment
│   │   ├── /orientation
│   │   ├── /attitude_estimation
│   │   ├── /steering
│   │   ├── /rpm_estimation      # All RPM documentation
│   │   │   ├── README.md       # Main RPM methodology
│   │   │   ├── /wp0_exploration
│   │   │   ├── /wp1_preprocessing
│   │   │   ├── /wp2_peak_detection
│   │   │   ├── /wp3_stft
│   │   │   ├── /wp4_fusion
│   │   │   └── /wp5-7_future
│   │   └── /validation
│   ├── /development         # Development docs
│   │   ├── /architecture
│   │   ├── /coding_standards
│   │   └── /rpm            # RPM-specific development guides
│   ├── /experimental_setup  # Experiment documentation
│   └── /migration          # Migration history
│
├── /notes                    # Thesis notes
├── /tests                    # Centralized test suite
│   ├── /test_alignment
│   ├── /test_orientation
│   ├── /test_rpm
│   └── /test_timestamp
│
├── .github/workflows         # CI/CD configuration
├── pyproject.toml           # Python package configuration
├── requirements.txt         # Python dependencies
├── Makefile                # Build automation
├── README.md               # Project overview
├── SETUP.md                # Setup instructions
├── CLAUDE.md              # Development guidelines
└── repository_tree.txt     # Repository structure listing
```

## Where Things Go

### When Writing Code:
- Core utilities → `/src/core/`
- Analysis modules → `/src/analysis/{module}/`
- Command-line scripts → `/src/scripts/`
- Jupyter notebooks → `/src/notebooks/`

### ⚠️ CRITICAL RULE:
**NO documentation or images in /src/!**
- ❌ NO .md files in /src/
- ❌ NO .png, .jpg, .jpeg, .gif files in /src/
- ❌ NO PDFs or other docs in /src/
- ✅ ONLY .py files (and config .yaml/.json) in /src/
- ✅ ALL documentation goes to /docs/

### When Documenting:
- Analysis methodology → `/docs/results/{analysis}/README.md`
- Work package docs → `/docs/results/{analysis}/wp*/`
- Development guides → `/docs/development/{module}/`
- Results and findings → `/docs/results/{analysis}/`
- Configuration docs → `/docs/config/`

### When Processing Data:
- Raw experimental data → `/data/raw/{morning,afternoon}/Experiments/`
- Aligned sensor data → `/data/processed/aligned/`
- Analysis outputs → `/data/processed/{analysis}/`
- Temporary files → `/data/cache/`

### When Configuring:
- Master config → `/config/pipeline.yaml`
- Experiment definitions → `/config/experiments/`
- Sensor specs → `/config/sensors/`
- Processing parameters → `/config/processing/`

## Import Patterns

### Correct Imports:
```python
# Core imports
from src.core import io, DATA_DIR, get_experiment_path
from src.core.paths import PROCESSED_DATA_DIR, ALIGNED_DATA_DIR

# Analysis imports
from src.analysis.rpm import preprocess, spectral, fusion
from src.analysis.alignment import align_data
from src.analysis.orientation import analyze_gravity

# App imports
from src.apps.dashboard import app
```

### Never Use:
```python
# DON'T use relative imports
import sys
sys.path.append('..')

# DON'T use old package names
from hovercraft_analysis.xxx import ...  # Old structure
from code.xxx import ...                  # Old structure

# DON'T hardcode paths
path = "../../02_Evaluation_Experiments/..."  # Use get_experiment_path()
path = "../data/aligned/..."                  # Use ALIGNED_DATA_DIR
```

## Key Files Reference

- **Master Config**: `/config/pipeline.yaml`
- **Path Management**: `/src/core/paths.py`
- **Dashboard App**: `/src/scripts/dashboard_app.py`
- **RPM Documentation**: `/docs/results/rpm_estimation/README.md`
- **Migration History**: `/docs/migration/`

## Development Guidelines

1. **Code in src, docs in docs**: Never mix documentation with source code
2. **Use path helpers**: Always use `get_experiment_path()` and constants from `src.core.paths`
3. **Follow import patterns**: Use absolute imports from `src.`
4. **Test after changes**: Run `pytest` from repository root
5. **Document in the right place**: Analysis docs go to `/docs/results/`, dev guides to `/docs/development/`

## Common Commands

```bash
# Install package in development mode
pip install -e .

# Run tests
pytest

# Run specific analysis
python -m src.scripts.run_alignment --experiment 007_Fast_stbd_turn_1

# Start dashboard
python src/scripts/dashboard_app.py

# Validate structure
python src/scripts/validate_structure.py
```
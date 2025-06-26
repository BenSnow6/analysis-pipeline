# Developer Setup Guide

This guide will help you set up the hovercraft analysis pipeline development environment.

## Prerequisites

- Python 3.8 or higher
- Git
- Access to the experimental data (in `/data` directory)

## Environment Setup

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd analysis-pipeline
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Install the package in development mode** (coming in Phase 2):
   ```bash
   # This will be available after Phase 2 implementation
   # pip install -e .
   ```

## Project Structure

```
analysis-pipeline/
├── code/              # All Python source code
│   ├── alignment_analysis/
│   ├── dashboard_app/
│   ├── orientation_analysis/
│   ├── timestamp_analysis/
│   ├── rpm_estimation/
│   ├── scripts/       # Standalone scripts
│   ├── notebooks/     # Jupyter notebooks
│   └── config/        # Configuration files
├── data/              # Experimental data
│   ├── morning/
│   └── afternoon/
├── docs/              # Documentation
├── tests/             # All test files (centralized)
│   ├── alignment/
│   ├── orientation/
│   ├── rpm_estimation/
│   ├── scripts/
│   └── timestamp/
└── notes/             # Thesis notes
```

## Running Tests

Run all tests:
```bash
python -m pytest tests/
```

Run tests for a specific module:
```bash
python -m pytest tests/alignment/
python -m pytest tests/orientation/
python -m pytest tests/rpm_estimation/
```

Run with coverage:
```bash
python -m pytest tests/ --cov=code --cov-report=html
```

## Import Convention

All imports should use the `code.` prefix:
```python
from code.alignment_analysis.align import align_data
from code.config.paths import DATA_DIR, get_experiment_path
from code.orientation_analysis import analyze_gravity
```

Never use:
- `sys.path.append()` or `sys.path.insert()`
- Relative imports across modules
- Hardcoded file paths

## Configuration

The main configuration files are located in `code/config/`:
- `paths.py` - Centralized path management
- `experiment_mapping.json` - Maps experiments to evaluation categories
- `sensor_orientations.json` - Sensor configuration data
- Various YAML files for module-specific settings

## Common Tasks

### Running the Dashboard
```bash
python code/scripts/dashboard_app.py
```

### Running Alignment Analysis
```bash
python code/scripts/run_alignment.py
```

### Processing Timestamp Analysis
```bash
python code/scripts/run_timestamp_analysis_standalone.py
```

## Data Access

Always use the centralized path helpers:
```python
from code.config.paths import get_experiment_path

# Get path to an experiment
exp_path = get_experiment_path("016_Straight_cruise_1", "afternoon")

# Access data files
gps_data = pd.read_csv(exp_path / "GPS" / "GPS_016_Straight_cruise_1.csv")
```

## Troubleshooting

### Import Errors
- Ensure you're using the `code.` prefix for all imports
- Check that you're in the project root directory when running scripts
- Verify all dependencies are installed: `pip install -r requirements.txt`

### Data Not Found
- Check that the data directory exists and contains the experimental data
- Use `get_experiment_path()` helper function instead of hardcoded paths
- Verify the experiment name and time_of_day parameters

### Test Failures
- Run tests from the project root directory
- Ensure test data files are available
- Check that all module dependencies are installed

## Code Style

- Use type hints for function parameters and return values
- Add docstrings to all public functions and classes
- Follow PEP 8 style guidelines
- Keep functions focused and small

## Getting Help

- Check the CLAUDE.md file for detailed development guidelines
- Review module-specific README files in each subdirectory
- Consult the documentation in the `/docs` directory

## Next Steps

After Phase 2 implementation, you'll be able to:
- Install the package with `pip install -e .`
- Use command-line tools like `hovercraft-align` and `hovercraft-dashboard`
- Run linting and formatting with `make lint` and `make format`
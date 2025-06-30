# Developer Setup Guide

This guide will help you set up the hovercraft analysis pipeline development environment.

## Prerequisites

- Python 3.12 or higher
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

3. **Install the package in development mode**:
   ```bash
   pip install --upgrade pip
   pip install -e ".[dev]"
   ```

## Project Structure

```
analysis-pipeline/
├── /config                    # Master configuration directory
│   ├── pipeline.yaml         # Master configuration file
│   ├── /experiments          # Experiment mappings and metadata
│   ├── /sensors             # Sensor specifications and orientations
│   └── /processing          # Module-specific processing configs
├── /src                      # All source code (consolidated)
│   ├── /core                # Core utilities and config management
│   ├── /analysis            # Analysis modules
│   │   ├── /alignment       # Time alignment
│   │   ├── /orientation     # Sensor orientation
│   │   ├── /rpm            # RPM estimation
│   │   └── /timestamp      # Timestamp analysis
│   ├── /apps               # Applications
│   │   └── /dashboard      # Dashboard app modules
│   ├── /scripts            # Command-line scripts
│   ├── /notebooks          # Jupyter notebooks
│   └── /plans              # Development plans
├── /data                     # All experimental and processed data
│   ├── /raw                 # Raw experimental data
│   │   ├── /morning/Experiments/  # Morning session experiments
│   │   └── /afternoon/Experiments/ # Afternoon session experiments
│   ├── /processed           # All processed outputs
│   │   ├── /aligned        # Time-aligned sensor data
│   │   ├── /orientation    # Orientation analysis results
│   │   ├── /rpm           # RPM estimation results
│   │   └── /timestamp     # Timestamp analysis results
│   └── /cache              # Temporary files and cache
├── /docs                     # Documentation
│   ├── /config              # Config documentation only
│   ├── /results             # Analysis results by thesis WPs
│   ├── /development         # Development docs
│   └── /experimental_setup  # Experiment documentation
├── /tests                    # Centralized test suite
├── .github/workflows         # CI/CD configuration
├── pyproject.toml           # Python package configuration
├── Makefile                # Build automation
└── README.md               # Project overview
```

## Running Tests

Run all tests:
```bash
make test-fast
# or
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
make test-cov
# or
python -m pytest tests/ --cov=src --cov-report=html
```

## Import Convention

All imports should use the `src.` prefix:
```python
from src.analysis.alignment import align_data
from src.core.paths import DATA_DIR, get_experiment_path
from src.analysis.orientation import analyze_gravity
from src.apps.dashboard import app
from src.analysis.rpm import RPMFrame
```

Never use:
- `sys.path.append()` or `sys.path.insert()`
- Old package names (`hovercraft_analysis.`, `code.`)
- Relative imports across modules
- Hardcoded file paths

## Configuration

The master configuration system is centralized in `/config/`:
- `pipeline.yaml` - Master configuration file with environment support
- `/experiments/` - Experiment mappings and categories
- `/sensors/` - Sensor specifications and orientations
- `/processing/` - Module-specific processing configurations

### Using Configuration
```python
from src.core import get_config

# Get configuration manager
config = get_config()

# Access configuration values
project_name = config.get('project.name')
data_root = config.get('paths.data_root')

# Get paths as Path objects
data_path = config.get_path('paths.data_root')
```

## Common Tasks

### Running the Dashboard
```bash
# Using installed entry point
hovercraft-dashboard

# Or run as module
python -m src.scripts.dashboard_app
```

### Running Alignment Analysis
```bash
# Using installed entry point
hovercraft-align --experiment 007_Fast_stbd_turn_1

# Or run as module
python -m src.scripts.run_alignment
```

### Processing Timestamp Analysis
```bash
# Using installed entry point
hovercraft-timestamp --experiment 016_Straight_cruise_1

# Or run as module
python -m src.scripts.run_timestamp_analysis_standalone
```

## Data Access

Always use the centralized path helpers:
```python
from src.core.paths import get_experiment_path
from src.core.io import load_experiment_data

# Load all data for an experiment
data = load_experiment_data("016_Straight_cruise_1", "afternoon")
gps_data = data['gps']
accel_data = data['Sensor_3_accel']

# Or get path to an experiment
exp_path = get_experiment_path("016_Straight_cruise_1", "afternoon")
gps_data = pd.read_csv(exp_path / "GPS" / "GPS_016_Straight_cruise_1.csv")
```

## Troubleshooting

### Import Errors
- Ensure you're using the `src.` prefix for all imports
- Install the package with `pip install -e .` to enable imports from anywhere
- Verify all dependencies are installed: `pip install -e ".[dev]"`

### Data Not Found
- Check that the data directory exists at `/data/raw/`
- Use `get_experiment_path()` helper function instead of hardcoded paths
- Verify the experiment name and time_of_day parameters

### Test Failures
- Run tests from the project root directory
- Ensure virtual environment is activated
- Check that all development dependencies are installed

## Code Style

- Use type hints for function parameters and return values
- Add docstrings to all public functions and classes
- Follow PEP 8 style guidelines
- Keep functions focused and small
- Run linting and formatting:
  ```bash
  make lint        # Check code quality
  make format      # Auto-format code
  ```

## Getting Help

- Check the CLAUDE.md file for detailed development guidelines
- Review the REPOSITORY_STRUCTURE.md for detailed structure information
- Consult the documentation in the `/docs` directory
- Report issues at the project's GitHub repository

## Development Workflow

1. **Before making changes**:
   ```bash
   make test-fast   # Ensure tests pass
   make lint        # Check code quality
   ```

2. **After making changes**:
   ```bash
   make test-fast   # Verify tests still pass
   make lint        # Ensure code quality
   ```

3. **Before committing**:
   ```bash
   make check-quality   # Run all quality checks
   ```

## Available Make Commands

- `make test` - Run all tests
- `make test-fast` - Run tests quickly (no coverage)
- `make test-cov` - Run tests with coverage report
- `make lint` - Run all linters
- `make format` - Auto-format code
- `make check-quality` - Run all quality checks
- `make clean` - Clean temporary files
# Getting Started with Hovercraft Analysis Pipeline

Welcome to the Hovercraft Analysis Pipeline! This guide will help you get up and running quickly.

## Prerequisites

- Python 3.8 or higher
- Git
- Make (optional but recommended)

## Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd analysis-pipeline
```

### 2. Set Up Development Environment

Using Make (recommended):
```bash
make dev
```

Or manually:
```bash
pip install -e ".[dev,notebook]"
```

### 3. Verify Installation

```bash
# Test that the package is installed
python -c "from src.core import get_experiment_path; print('✓ Package installed successfully')"

# Run tests
make test-fast
```

### 4. Launch the Dashboard

```bash
make run-dashboard
# or
hovercraft-dashboard
```

The dashboard will be available at http://localhost:8050

## Project Structure

```
analysis-pipeline/
├── config/           # Configuration files
├── src/              # Source code
│   ├── core/         # Core utilities
│   ├── analysis/     # Analysis modules
│   ├── apps/         # Applications (dashboard)
│   └── scripts/      # CLI scripts
├── data/             # Experimental data
│   ├── raw/          # Raw sensor data
│   └── processed/    # Processed outputs
├── tests/            # Test suite
└── docs/             # Documentation
```

## Available Commands

### Using Make

```bash
make help         # Show all available commands
make test         # Run tests with coverage
make lint         # Run code quality checks
make format       # Auto-format code
make docs         # Build documentation
```

### Using CLI Tools

After installation, these commands are available:

```bash
hovercraft-align --experiment <name>        # Run alignment analysis
hovercraft-orientation --experiment <name>  # Run orientation analysis
hovercraft-timestamp --experiment <name>    # Run timestamp analysis
hovercraft-dashboard                        # Launch analysis dashboard
```

## Working with Experiments

### Finding Experiments

List available experiments:
```python
from src.core import get_available_experiments

experiments = get_available_experiments()
for exp in experiments:
    print(f"{exp['name']} - {exp['category']} ({exp['time_of_day']})")
```

### Loading Experiment Data

```python
from src.core import load_experiment_data

# Load all data for an experiment
data = load_experiment_data("007_Fast_stbd_turn_1", "afternoon")

# Access specific sensor data
gps_data = data['gps']
accel_data = data['Sensor_3_accel']
```

### Running Analysis

```python
from src.analysis.alignment import align_experiment_data

# Align sensor data
aligned_data = align_experiment_data("007_Fast_stbd_turn_1", "afternoon")
```

## Configuration

The project uses a unified configuration system. Main configuration file: `/config/pipeline.yaml`

### Environment Variables

- `HOVERCRAFT_ENV`: Set to 'development', 'production', or 'testing'
- `PROJECT_ROOT`: Automatically detected, can be overridden

### Accessing Configuration

```python
from src.core import get_config

config = get_config()
data_root = config.get('paths.data_root')
```

## Development Workflow

1. Create a feature branch
2. Make changes
3. Format code: `make format`
4. Run tests: `make test`
5. Check quality: `make lint`
6. Commit changes
7. Push and create PR

## Troubleshooting

### Import Errors

If you encounter import errors:
```bash
# Reinstall in development mode
pip install -e .
```

### Missing Dependencies

```bash
# Install all dependencies
make dev
```

### Data Not Found

Check that data is in the correct location:
- Raw data: `/data/raw/morning|afternoon/Experiments/`
- Processed data: `/data/processed/`

## Next Steps

- Read the [Architecture Guide](architecture.md) to understand the system design
- Explore the [API Documentation](api/index.html) for detailed module references
- Check experiment documentation in `/docs/Experimental setup/`

## Getting Help

- Check existing documentation
- Look at test files for usage examples
- Review the codebase analysis in `/docs/codebase_analysis/`
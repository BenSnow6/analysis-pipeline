# Hovercraft Analysis Pipeline

A comprehensive Python package for processing and analyzing sensor data from hovercraft experiments, including IMU, GPS, and other sensor types.

## 🚀 Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd analysis-pipeline

# Install in development mode
make dev

# Run tests
make test

# Launch the dashboard
make run-dashboard
```

## 📁 Repository Structure

```
analysis-pipeline/
├── src/                      # Source code
│   ├── core/                # Core utilities
│   ├── analysis/            # Analysis modules
│   ├── apps/               # Applications (dashboard)
│   └── scripts/            # CLI entry points
├── config/                  # Configuration files
│   ├── pipeline.yaml       # Master configuration
│   ├── experiments/        # Experiment metadata
│   ├── sensors/           # Sensor configurations
│   └── processing/        # Processing configs
├── data/                   # All data (gitignored)
│   ├── raw/               # Raw experimental data
│   ├── processed/         # Analysis outputs
│   └── cache/             # Temporary files
├── tests/                  # Test suite
├── docs/                   # Documentation
│   ├── getting_started.md  # Quick start guide
│   ├── architecture.md     # System design
│   ├── migration/         # Migration history
│   └── development/       # Dev resources
├── scripts/               # Utility scripts
├── Makefile              # Development commands
└── pyproject.toml        # Package configuration
```

## 🛠️ Installation

### Prerequisites

- Python 3.12 or higher
- pip
- Make (optional but recommended)

### Development Installation

```bash
# Using Make (recommended)
make dev

# Or manually
pip install -e ".[dev,notebook]"
```

### Basic Installation

```bash
# For users who just want to run the tools
make install

# Or manually
pip install -e .
```

## 📊 Features

- **Data Processing**: Automatic alignment and synchronization of multi-sensor data
- **Analysis Modules**: 
  - Time alignment and synchronization
  - Orientation analysis
  - Timestamp validation
  - RPM estimation
- **Visualization**: Interactive web dashboard for data exploration
- **CLI Tools**: Command-line interfaces for all major functions
- **Quality Assurance**: Type hints, tests, and automated formatting

## 🎯 Usage

### Command-Line Tools

After installation, these commands are available:

```bash
# Align sensor data for an experiment
hovercraft-align --experiment 007_Fast_stbd_turn_1

# Launch the analysis dashboard
hovercraft-dashboard

# Run timestamp analysis
hovercraft-timestamp --experiment 016_Straight_cruise_1

# Analyze sensor orientations
hovercraft-orientation --experiment 011_Static_stbd_1
```

### Python API

```python
from src.core import load_experiment_data
from src.analysis.alignment import align_experiment_data

# Load experiment data
data = load_experiment_data("007_Fast_stbd_turn_1", "afternoon")

# Run alignment
aligned_data = align_experiment_data("007_Fast_stbd_turn_1", "afternoon")
```

## 🧑‍💻 Development

### Available Make Commands

```bash
make help         # Show all available commands
make test         # Run tests with coverage
make lint         # Run code quality checks
make format       # Auto-format code
make docs         # Build documentation
make clean        # Remove build artifacts
```

### Code Style

This project uses:
- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

Run `make format` before committing to ensure consistent style.

### Testing

```bash
# Run all tests with coverage
make test

# Run tests quickly (no coverage)
make test-fast

# Run specific test file
pytest tests/test_alignment.py -v
```

## 📚 Documentation

- [Getting Started Guide](docs/getting_started.md)
- [Architecture Overview](docs/architecture.md)
- [API Documentation](docs/api/index.html) (build with `make docs`)
- [Migration History](docs/migration/) - Details about the codebase reorganization

## 🔧 Configuration

The project uses a unified configuration system. The master configuration file is located at `config/pipeline.yaml`.

### Environment Variables

- `HOVERCRAFT_ENV`: Set to 'development', 'production', or 'testing'
- `PROJECT_ROOT`: Automatically detected, can be overridden

### Configuration Access

```python
from src.core import get_config

config = get_config()
data_root = config.get('paths.data_root')
```

## 🤝 Contributing

1. Create a feature branch
2. Make your changes
3. Run `make format` to format code
4. Run `make test` to ensure tests pass
5. Run `make lint` to check code quality
6. Submit a pull request

## 📄 License

[Your license here]

## 🙏 Acknowledgments

This project was developed as part of an EngD thesis on hovercraft sensor data analysis.

## 📞 Support

For questions or issues:
- Check the [documentation](docs/)
- Review [existing issues](https://github.com/your-org/hovercraft-analysis/issues)
- Create a new issue with a detailed description

---

**Note**: This codebase has undergone a major reorganization to follow Python best practices. For historical context, see the [migration documentation](docs/migration/).
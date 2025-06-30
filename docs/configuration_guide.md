# Configuration Guide

This guide explains how to use and modify the configuration system for the hovercraft analysis pipeline.

## Overview

The project uses a unified configuration system centered around a master configuration file at `/config/pipeline.yaml`. This system provides:

- Centralized configuration management
- Environment variable substitution
- Environment-specific overrides
- Path resolution and validation
- Backward compatibility with legacy code

## Configuration Structure

```
/config/
├── pipeline.yaml              # Master configuration file
├── experiments/
│   ├── experiment_mapping.json    # Maps experiments to categories
│   ├── experiment_manifest.yaml   # Experiment metadata
│   └── experiment_categories.yaml # Category descriptions
├── sensors/
│   ├── sensor_orientations.json   # Sensor mounting configurations
│   ├── sensor_specs.yaml          # Sensor specifications
│   ├── morning_sensor_locations.json
│   └── afternoon_sensor_locations.json
└── processing/
    ├── alignment_config.yaml      # Alignment processing settings
    ├── orientation_config.yaml    # Orientation analysis settings
    ├── rpm_config.yaml           # RPM estimation settings
    └── timestamp_config.yaml     # Timestamp analysis settings
```

## Using Configuration in Code

### Basic Usage

```python
from src.core import get_config

# Get the global configuration manager
config = get_config()

# Access configuration values using dot notation
project_name = config.get('project.name')
data_root = config.get('paths.data_root')
use_legacy = config.get('features.use_legacy_paths', False)  # With default

# Get paths as Path objects
data_path = config.get_path('paths.data_root')
aligned_dir = config.get_path('paths.processed_data.aligned')
```

### Loading Sub-Configurations

```python
# Load a referenced configuration file
rpm_config = config.load_sub_config('processing.rpm')
sensor_specs = config.load_sub_config('sensors.specs')

# Access nested values
win_sec = rpm_config['welch']['win_sec']
sensor_rate = sensor_specs['sensors']['sensor_3']['expected_rate_hz']
```

### Legacy Compatibility

The system maintains compatibility with legacy code:

```python
# These properties still work
mapping = config.experiment_mapping      # Loads experiments/experiment_mapping.json
orientations = config.sensor_orientations # Loads sensors/sensor_orientations.json

# Legacy helper methods
category = config.get_experiment_category("007_Fast_stbd_turn_1")
sensor_config = config.get_sensor_orientation("Sensor_3")
```

## Environment Variables

### Supported Variables

- `HOVERCRAFT_ENV`: Set the environment (development/production/testing)
- `PROJECT_ROOT`: Override the project root directory
- Any custom variables used in your configuration

### Variable Substitution Syntax

```yaml
# Simple substitution
data_root: "${PROJECT_ROOT}/data"

# With default value
cache_dir: "${CACHE_DIR:-/tmp/hovercraft_cache}"

# Nested references (resolved first)
aligned_data: "${paths.processed_data.root}/aligned"
```

## Environment-Specific Configuration

Define environment-specific overrides in `pipeline.yaml`:

```yaml
environments:
  development:
    logging.level: "DEBUG"
    processing.verbose: true
    
  production:
    logging.level: "WARNING"
    processing.verbose: false
    
  testing:
    paths.data_root: "${project_root}/tests/data"
    processing.n_workers: 1
```

Set the environment:
```bash
export HOVERCRAFT_ENV=production
python your_script.py
```

## Adding New Configuration

### 1. Add to Master Config

Edit `/config/pipeline.yaml`:

```yaml
# Add new module configuration reference
configs:
  processing:
    my_new_module: "${config_root}/processing/my_module_config.yaml"

# Add module defaults
modules:
  my_new_module:
    enabled: true
    some_default: 42
```

### 2. Create Module Config File

Create `/config/processing/my_module_config.yaml`:

```yaml
# My Module Configuration
processing_params:
  window_size: 100
  threshold: 0.5
  
output:
  format: "csv"
  save_plots: true
```

### 3. Use in Code

```python
from src.core import get_config

config = get_config()
module_config = config.load_sub_config('processing.my_new_module')

window_size = module_config['processing_params']['window_size']
```

## Best Practices

1. **Use the configuration system** - Don't hardcode paths or parameters
2. **Document your configs** - Add comments explaining parameters
3. **Validate early** - Check required config values at startup
4. **Use defaults** - Provide sensible defaults for optional parameters
5. **Keep secrets out** - Never commit sensitive data to config files

## Troubleshooting

### Config Not Found

If the master config isn't found, the system falls back to legacy mode:

```python
# Check if using master config
config = get_config()
if config._master_config_path:
    print(f"Using config from: {config._master_config_path}")
else:
    print("Running in legacy mode")
```

### Path Resolution Issues

Debug path resolution:

```python
# Check resolved paths
config = get_config()
print(f"Data root: {config.get('paths.data_root')}")
print(f"Exists: {config.get_path('paths.data_root').exists()}")

# Enable auto-creation of directories
if config.get('features.auto_create_dirs'):
    config.create_missing_dirs()
```

### Reload Configuration

If you modify config files during development:

```python
config = get_config()
config.reload()  # Reloads all configuration files
```

## Migration from Old System

If you have code using old hardcoded paths:

```python
# Old way (don't do this)
data_path = "../../02_Evaluation_Experiments/afternoon/007_Fast_stbd_turn_1"

# New way
from src.core import get_experiment_path
data_path = get_experiment_path("007_Fast_stbd_turn_1", "afternoon")
```

For old config imports:

```python
# Old way
with open("../config/sensor_orientations.json") as f:
    orientations = json.load(f)

# New way  
config = get_config()
orientations = config.sensor_orientations
```
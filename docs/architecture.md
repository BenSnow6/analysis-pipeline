# Hovercraft Analysis Pipeline Architecture

## Overview

The Hovercraft Analysis Pipeline is a comprehensive system for processing and analyzing sensor data from hovercraft experiments. It handles multiple IMU sensors, GPS data, and various experimental configurations.

## System Architecture

### High-Level Components

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Raw Data      │────▶│  Analysis Core   │────▶│  Visualization  │
│  (CSV Files)    │     │  (Processing)    │     │  (Dashboard)    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                       │                         │
         ▼                       ▼                         ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Experiment      │     │  Configuration   │     │   Results &     │
│   Metadata      │     │   Management     │     │   Reports       │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

### Package Structure

```
hovercraft_analysis/
├── core/              # Core utilities and infrastructure
│   ├── config.py      # Configuration management
│   ├── paths.py       # Path definitions
│   ├── io.py          # Data I/O operations
│   └── classes.py     # Data structures
│
├── analysis/          # Analysis modules
│   ├── alignment/     # Time synchronization
│   ├── orientation/   # Orientation analysis
│   ├── timestamp/     # Timestamp validation
│   └── rpm/           # RPM estimation
│
├── apps/              # Applications
│   └── dashboard/     # Web-based visualization
│
└── scripts/           # CLI entry points
```

## Data Flow

### 1. Data Ingestion

Raw sensor data flows through the pipeline:

```
Raw CSV Files → Data Loader → Validation → DataFrame
```

Key components:
- **Data Loader**: `core.io.load_experiment_data()`
- **Validation**: Checks for required columns, data types
- **Output**: Pandas DataFrames with standardized structure

### 2. Alignment Processing

Synchronizes data from multiple sensors:

```
Multiple Sensors → Time Alignment → Interpolation → Aligned Dataset
```

Process:
1. Find common time range across all sensors
2. Resample to common frequency (100 Hz default)
3. Interpolate missing values
4. Save aligned data in HDF5 format

### 3. Analysis Modules

Each module processes aligned data for specific insights:

#### Orientation Analysis
- Extracts gravity vectors from accelerometer data
- Determines sensor mounting orientations
- Validates against expected configurations

#### Timestamp Analysis
- Detects timing anomalies
- Identifies gaps and duplicates
- Generates timing quality reports

#### RPM Estimation
- Processes accelerometer/gyroscope data
- Estimates engine/propeller RPM
- Applies filtering and peak detection

## Configuration System

### Master Configuration

Located at `/config/pipeline.yaml`:

```yaml
project:
  name: "Hovercraft Analysis Pipeline"
  version: "1.0.0"

paths:
  data_root: "${PROJECT_ROOT}/data"
  raw_data: "${data_root}/raw"
  processed_data: "${data_root}/processed"

processing:
  alignment:
    target_frequency: 100
    interpolation_method: "linear"
```

### Environment-Specific Overrides

```yaml
environments:
  development:
    logging:
      level: "DEBUG"
  production:
    logging:
      level: "INFO"
```

### Configuration Access

```python
from hovercraft_analysis.core import get_config

config = get_config()
freq = config.get('processing.alignment.target_frequency')
```

## Key Design Patterns

### 1. Centralized Path Management

All file paths are managed through `core.paths`:
- No hardcoded paths in analysis code
- Easy reconfiguration for different environments
- Consistent path resolution

### 2. Modular Analysis Pipeline

Each analysis module is self-contained:
- Independent configuration
- Standardized input/output formats
- Can be run individually or as part of pipeline

### 3. Lazy Loading

Data is loaded on-demand:
- Reduces memory usage
- Faster startup times
- Allows processing of large datasets

### 4. Experiment Abstraction

Experiments are treated as first-class entities:
```python
experiment = Experiment("007_Fast_stbd_turn_1", "afternoon")
data = experiment.load_data()
results = experiment.run_analysis("alignment")
```

## Data Formats

### Input Format (CSV)

Standard sensor data format:
```csv
Timestamp,AccelX,AccelY,AccelZ,GyroX,GyroY,GyroZ,...
1234567890.123,0.981,-0.001,9.805,0.001,0.002,-0.001,...
```

### Aligned Data Format (HDF5)

Hierarchical structure:
```
experiment.h5
├── /metadata
│   ├── experiment_name
│   ├── time_of_day
│   └── sensors
├── /gps
│   ├── timestamps
│   ├── latitude
│   └── longitude
└── /sensors
    ├── /Sensor_3
    │   ├── accel
    │   └── gyro
    └── /Sensor_4
        ├── accel
        └── gyro
```

## Error Handling

### Validation Levels

1. **Data Validation**: Check file existence, format
2. **Sensor Validation**: Verify expected sensors present
3. **Quality Validation**: Check for gaps, outliers
4. **Result Validation**: Ensure outputs are reasonable

### Error Recovery

- Missing files: Skip with warning
- Corrupted data: Use interpolation or flag
- Configuration errors: Fall back to defaults

## Performance Considerations

### Memory Management

- Chunk processing for large files
- Selective column loading
- Result caching

### Parallel Processing

Where applicable:
- Multi-sensor processing
- Batch experiment analysis
- Independent module execution

## Extension Points

### Adding New Analysis Modules

1. Create module in `analysis/` directory
2. Implement standard interface:
   ```python
   def analyze(experiment_name: str, time_of_day: str) -> Results:
       pass
   ```
3. Add configuration to `/config/processing/`
4. Create CLI script in `scripts/`

### Custom Visualizations

1. Add components to `apps/dashboard/`
2. Register in dashboard layout
3. Connect to data callbacks

## Security Considerations

- Input validation on all file operations
- Path traversal prevention
- Configuration validation
- No execution of arbitrary code

## Testing Strategy

### Unit Tests
- Individual function testing
- Mock external dependencies
- Focus on edge cases

### Integration Tests
- Full pipeline execution
- Real data samples
- End-to-end validation

### Performance Tests
- Large dataset handling
- Memory usage monitoring
- Processing time benchmarks
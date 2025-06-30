# Technical Documentation: Timestamp Analysis Tool

## Overview

This document provides comprehensive technical documentation for the timestamp analysis tool developed as part of the hovercraft simulator validation pipeline. This tool addresses a critical requirement identified in the thesis plan (Week 1): to verify timestamp consistency across multiple sensor systems with < 20ms jitter, forming the foundation for subsequent data alignment and fusion steps.

## 1. Problem Statement and Requirements

### 1.1 Context
The hovercraft data collection system employs multiple sensors operating at different sampling rates:
- GPS: ~1 Hz (positional ground truth)
- IMU sensors (5 units): 100-200 Hz (motion dynamics)

Each sensor system operates independently with its own clock, creating potential timing inconsistencies that must be identified and quantified before data fusion.

### 1.2 Specific Requirements
From the thesis plan specifications:
1. **Jitter Detection**: Identify timestamp jitter > 20ms in IMU data
2. **Gap Detection**: Detect timing gaps > 100ms indicating data loss
3. **Multi-rate Support**: Handle sensors with different expected sampling rates
4. **Configurable Thresholds**: Allow sensor-specific timing requirements
5. **Batch Processing**: Analyze multiple experiments efficiently
6. **Reporting**: Generate both human-readable and machine-parseable outputs

### 1.3 Design Constraints
- Must integrate with existing dashboard data loading infrastructure
- Support for both 't' and 'time_from_sync' timestamp columns
- Handle missing or corrupted data gracefully
- Provide clear pass/fail criteria for each sensor

## 2. Architecture and Design Decisions

### 2.1 Modular Architecture
The tool follows a modular design pattern with clear separation of concerns:

```
timestamp_analysis/
├── config/
│   └── sensor_specs.yaml    # Configuration management
├── data_loader.py           # Data I/O operations
├── timestamp_analyzer.py    # Core analysis algorithms
├── visualizer.py           # Plotting and visualization
├── report_generator.py     # Report generation
└── main.py                 # CLI and orchestration
```

**Rationale**: This structure enables:
- Independent testing of each module
- Easy extension for new analysis methods
- Clear data flow from loading → analysis → visualization → reporting

### 2.2 Configuration-Driven Analysis
The `sensor_specs.yaml` file externalizes all sensor-specific parameters:

```yaml
sensors:
  gps:
    expected_rate_hz: 1
    jitter_threshold_ms: 100
    gap_threshold_factor: 2.0
```

**Rationale**: 
- Accommodates varying sensor configurations without code changes
- Enables experiment-specific overrides
- Facilitates parameter tuning during validation

### 2.3 Statistical Analysis Approach

#### 2.3.1 Jitter Calculation
Jitter is defined as the absolute deviation from the expected sampling interval:

```python
jitter[i] = |interval[i] - expected_interval|
```

Where:
- `interval[i] = timestamp[i+1] - timestamp[i]`
- `expected_interval = 1 / expected_rate_hz`

**Rationale**: This metric directly measures timing consistency and is easily interpretable.

#### 2.3.2 Gap Detection
Gaps are identified when the interval exceeds a threshold:

```python
gap_threshold = expected_interval * gap_threshold_factor
```

**Rationale**: The factor-based approach scales appropriately with sampling rate, preventing false positives for low-rate sensors like GPS.

#### 2.3.3 Rate Estimation
Actual sampling rate is calculated using the entire time series:

```python
actual_rate = (num_samples - 1) / (last_timestamp - first_timestamp)
```

**Rationale**: This provides a robust estimate less sensitive to individual timing variations.

## 3. Implementation Details

### 3.1 Data Loading Strategy

The `data_loader.py` module extends the existing dashboard infrastructure:

```python
def load_experiment_data(experiment_path, specs):
    # Reuses dashboard file discovery logic
    # Handles both GPS and IMU data formats
    # Returns unified data structure
```

**Key Features**:
- Automatic sensor discovery
- Graceful handling of missing files
- Support for multiple timestamp formats

### 3.2 Analysis Pipeline

The `timestamp_analyzer.py` implements a comprehensive analysis pipeline:

1. **Timestamp Extraction**: Convert to consistent numpy arrays
2. **Interval Calculation**: Compute time differences
3. **Statistical Analysis**: Calculate mean, std, max metrics
4. **Threshold Validation**: Check against specifications
5. **Result Packaging**: Structure results for downstream use

### 3.3 Visualization Approach

The `visualizer.py` creates four key plot types:

1. **Interval Time Series**: Shows timing variations over experiment duration
2. **Jitter Histogram**: Displays distribution of timing deviations  
3. **Timeline View**: Highlights data gaps and coverage
4. **Cross-Sensor Alignment**: Compares relative timing between sensors

**Design Choice**: Matplotlib was chosen for static plot generation suitable for thesis inclusion and automated report generation.

### 3.4 Report Generation

The `report_generator.py` produces multiple output formats:

1. **HTML Reports**: Interactive, styled reports with embedded visualizations
2. **CSV Summaries**: Machine-readable results for further analysis
3. **JSON Output**: Complete results with full precision

## 4. Algorithm Complexity and Performance

### 4.1 Time Complexity
- Data Loading: O(n) where n is number of samples
- Timestamp Analysis: O(n) for interval calculations
- Visualization: O(n) for plotting operations
- Overall: O(n) linear complexity

### 4.2 Space Complexity
- Memory usage scales linearly with data size
- Typical experiment (~200Hz, 5 minutes): ~2MB per sensor

### 4.3 Performance Optimizations
- Numpy arrays for vectorized operations
- Lazy loading of sensor data
- Matplotlib figure reuse for batch processing

## 5. Validation and Testing Strategy

### 5.1 Unit Test Coverage
Tests validate:
- Jitter calculation accuracy
- Gap detection sensitivity
- Rate estimation precision
- Edge cases (empty data, single sample)

### 5.2 Integration Testing
- Known synthetic datasets with injected timing issues
- Comparison with manual analysis results
- Cross-validation with MATLAB implementations

### 5.3 Acceptance Criteria
A sensor passes timestamp validation if:
1. Sampling rate deviation < 10% of expected
2. Mean jitter < threshold (sensor-specific)
3. No jitter samples exceed threshold
4. Gap count within acceptable limits

## 6. Usage Examples

### 6.1 Single Experiment Analysis
```bash
python -m hovercraft_data_analysis.timestamp_analysis \
    --experiment "1a_1_Minimum_Radius_Turn/afternoon/007_Fast_stbd_turn_1" \
    --output results/
```

### 6.2 Batch Analysis
```bash
python -m hovercraft_data_analysis.timestamp_analysis \
    --all \
    --spec custom_specs.yaml \
    --output batch_results/
```

### 6.3 Auto-Detection Mode
```bash
python -m hovercraft_data_analysis.timestamp_analysis \
    --experiment "path/to/experiment" \
    --update-spec
```

## 7. Integration with Data Pipeline

### 7.1 Position in Pipeline
```
Raw Data → [Timestamp Analysis] → Alignment → Filtering → Sim Comparison
                     ↓
              Quality Report
```

### 7.2 Output Usage
The tool outputs:
- Timing metadata for `align.py` development
- Quality metrics for experiment selection
- Diagnostic plots for thesis figures

### 7.3 Future Extensions
- Real-time analysis during data collection
- Automatic clock drift correction
- Multi-experiment timing correlation

## 8. Theoretical Foundation

### 8.1 Shannon-Nyquist Considerations
For IMU sensors sampling vehicle dynamics:
- Expected vehicle dynamics: < 10 Hz
- IMU sampling rate: 100-200 Hz
- Oversampling factor: 10-20×

This oversampling provides robustness against timing jitter while maintaining signal fidelity.

### 8.2 Jitter Impact Analysis
Maximum acceptable jitter (20ms) represents:
- 2% of GPS sampling interval (acceptable)
- 10-20% of IMU sampling interval (boundary condition)

This threshold ensures sufficient timing precision for subsequent Kalman filtering operations.

## 9. Limitations and Assumptions

### 9.1 Current Limitations
1. Assumes monotonic timestamps (no backward jumps)
2. Single time base per sensor (no clock switching)
3. Uniform sampling rate expectation (no adaptive sampling)

### 9.2 Assumptions
1. System clock drift is negligible over experiment duration
2. Timestamp precision exceeds analysis requirements
3. Data gaps represent actual missing samples (not buffering delays)

## 10. Conclusions

This timestamp analysis tool provides a robust foundation for validating multi-sensor timing consistency in the hovercraft data collection system. By identifying and quantifying timing issues early in the processing pipeline, it enables informed decisions about data quality and subsequent processing strategies.

The modular architecture and configuration-driven approach ensure the tool can adapt to evolving sensor configurations and analysis requirements throughout the thesis project timeline.

## References

1. IEEE Std 1588-2008: Precision Clock Synchronization Protocol
2. Allan, D.W. (1966). Statistics of atomic frequency standards. Proceedings of the IEEE, 54(2), 221-230.
3. Bar-Shalom, Y., Li, X. R., & Kirubarajan, T. (2001). Estimation with applications to tracking and navigation. John Wiley & Sons.

---

*This documentation serves as both a technical reference and a methodological justification for the timestamp analysis approach employed in the hovercraft simulator validation pipeline.*
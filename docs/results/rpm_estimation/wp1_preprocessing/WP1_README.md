# Work Package 1 (WP-1) Implementation

## Overview

WP-1 implements the raw data audit and orientation processing pipeline for the RPM estimation project. This work package:

1. Loads aligned CSV data from experiments
2. Applies body-frame rotation using validated orientation matrices
3. Processes vibration signals with high-pass filtering
4. Performs comprehensive quality assessment
5. Outputs Parquet files with processed data and quality reports

## Key Features

### 1. Structured Logging
- JSON-formatted logs with contextual information
- Error categorization (recoverable, fatal, quality, etc.)
- Processing step tracking

### 2. Configurable Processing
All parameters exposed in `rpm_config.yaml`:
```yaml
wp1:
  sensors:
    default: ["Sensor_3", "Sensor_4", "Sensor_wb"]
    max_g_range: 16.0
  filters:
    highpass_cutoff: 5.0
    highpass_order: 4
  quality:
    window_sec: 30.0
    window_handling: "process_partial"
    clipping_threshold: 0.95
```

### 3. Quality Assessment
- Per-window metrics: RMS, kurtosis, peak-to-RMS ratio
- Clipping detection with configurable thresholds
- Overall quality classification (excellent/good/fair/poor)
- Per-axis quality checks

### 4. Schema Validation
- Consistent Parquet schema across all outputs
- Metadata tracking for reproducibility
- Data consistency validation

## Usage

### Process Single Experiment
```bash
python -m rpm_estimation.cli --wp 1 --exp 007_Fast_stbd_turn_1 --session afternoon
```

### Process All Experiments
```bash
python -m rpm_estimation.cli --wp 1 --all --session morning
```

### Override Sensors
```bash
python -m rpm_estimation.cli --wp 1 --exp 007_Fast_stbd_turn_1 --session afternoon --sensors Sensor_3 Sensor_5
```

### Validation Mode
```bash
python -m rpm_estimation.cli --wp 1 --validate --include-synthetic
```

### JSON Logging
```bash
python -m rpm_estimation.cli --wp 1 --exp 007_Fast_stbd_turn_1 --session afternoon --log-format json --log-file processing.log
```

## Output Structure

```
aligned_data/
├── morning/
│   └── 015_Skirt_shift_turns/
│       ├── proc_IMU_Sensor_3.parquet
│       ├── proc_IMU_Sensor_5.parquet
│       ├── qa_summary_Sensor_3.json
│       └── qa_summary_Sensor_5.json
└── afternoon/
    └── 007_Fast_stbd_turn_1/
        ├── proc_IMU_Sensor_3.parquet
        ├── proc_IMU_Sensor_4.parquet
        ├── qa_summary_Sensor_3.json
        └── qa_summary_Sensor_4.json
```

## Parquet Schema

Required columns:
- `time_from_sync` (float64): Synchronized timestamp
- `a_hp_x`, `a_hp_y`, `a_hp_z` (float64): High-pass filtered accelerations
- `a_hp_mag` (float64): Vibration magnitude
- `quality_flag` (int8): 0=good, 1=warning, 2=bad

Optional columns:
- `x_body`, `y_body`, `z_body`: Body-frame accelerations
- `window_id` (int32): Quality assessment window ID

## Quality Report Format

```json
{
  "experiment": "007_Fast_stbd_turn_1",
  "session": "afternoon",
  "sensor_id": "Sensor_3",
  "summary": {
    "total_windows": 42,
    "clipped_windows": 2,
    "clipping_percentage": 4.76,
    "overall_quality": "good",
    "quality_score": 0.952
  },
  "windows": [...],
  "axes_quality": {
    "x": {"quality": "good", "issues": []},
    "y": {"quality": "good", "issues": []},
    "z": {"quality": "poor", "issues": ["dc_offset"]}
  }
}
```

## Done Criteria

✅ All aligned CSVs load without exceptions  
✅ Orientation transforms pass rotation matrix tests  
✅ High-pass filter removes DC (mean < 0.01 m/s²)  
✅ Synthetic 25 Hz test achieves SNR ≥ 25 dB  
✅ Parquet files exist for ALL sensors/experiments  
✅ QA JSON summaries generated for each experiment  
✅ No more than 5% of windows flagged as clipped  
✅ Marker file `wp1_done.flag` created on success  

## Testing

Run the test suite:
```bash
pytest tests/ -v
```

Run specific test modules:
```bash
pytest tests/test_preprocessing.py -v
pytest tests/test_quality.py -v
pytest tests/test_schema.py -v
```

## Performance

- Parallel processing of sensors (up to 4 workers)
- Typical experiment processes in <5 minutes
- Memory-efficient windowed processing

## Next Steps

After WP-1 completion:
- WP-2: Implement Welch PSD for frequency extraction
- WP-3: Add STFT for transient analysis
- WP-4: Multi-sensor fusion logic
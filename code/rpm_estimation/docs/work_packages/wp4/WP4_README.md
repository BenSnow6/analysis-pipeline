# WP-4: Multi-Sensor Fusion & Confidence Gating

## Overview

Work Package 4 implements intelligent multi-sensor fusion to combine RPM estimates from multiple IMUs. It uses SNR-based selection, confidence gating, and interpolation to produce a robust, continuous RPM time series suitable for downstream analysis.

## Key Features

### 1. **SNR-Based Sensor Selection**
- Automatically selects the sensor with highest SNR at each time point
- Excludes sensors below 10 dB SNR threshold
- Tracks sensor contributions over time

### 2. **Intelligent Method Selection**
- Combines WP-2 (Welch) and WP-3 (STFT) results
- Uses STFT for transient regions (>150 RPM/s)
- Uses Welch for steady-state (better frequency resolution)

### 3. **Gap Interpolation**
- Fills gaps up to 5 seconds using median of recent valid data
- Marks interpolated data with quality flag
- Preserves data integrity with clear tracking

### 4. **Quality Indicators**
- `rpm_valid` boolean flag for downstream filtering
- Confidence scores based on sensor agreement
- Detailed quality statistics in fusion report

## Usage

### Basic Command

```bash
# Process single experiment
python -m rpm_estimation.cli --wp 4 --exp 026_Engine_rpm_sweep --session afternoon --plot

# Process all experiments in session
python -m rpm_estimation.cli --wp 4 --all --session afternoon
```

### Advanced Options

```bash
# Custom fusion strategy
python -m rpm_estimation.cli --wp 4 --exp 016_Straight_cruise_1 --session afternoon \
    --fusion-strategy weighted \
    --min-sensors 2 \
    --interpolation-window 3.0

# Save intermediate data
python -m rpm_estimation.cli --wp 4 --exp 007_Fast_stbd_turn_1 --session afternoon \
    --save-intermediate \
    --output-dir custom_output/
```

### CLI Options

- `--fusion-strategy [snr_max|median|weighted]`: Fusion algorithm (default: snr_max)
- `--min-sensors N`: Minimum sensors required for valid estimate
- `--interpolation-window S`: Maximum gap to interpolate in seconds (default: 5.0)
- `--save-intermediate`: Save per-sensor fusion data
- `--plot`: Generate diagnostic plots

## Output Files

### 1. **rpm_fused.csv**
Primary output with fused RPM time series:
```csv
time,rpm,snr_db,sensor_id,method,quality,rpm_valid
0.0,1800.5,15.2,fused_Sensor_3,welch,measured,true
0.25,1805.2,12.8,fused_Sensor_4,stft,measured,true
0.5,1810.0,8.5,fused_multi,interpolated,interpolated,false
```

### 2. **fusion_report.json**
Comprehensive fusion statistics:
```json
{
  "experiment": "026_Engine_rpm_sweep",
  "session": "afternoon",
  "processing_time_s": 45.2,
  "sensor_contributions": {
    "Sensor_3": 0.45,
    "Sensor_4": 0.35,
    "Sensor_wb": 0.20
  },
  "quality_statistics": {
    "availability": 96.5,
    "interpolated_fraction": 0.015,
    "mean_snr_db": 14.2
  }
}
```

### 3. **fusion_diagnostic.png**
Three-panel visualization showing:
- RPM time series with quality indicators
- SNR evolution and threshold
- Sensor selection timeline

## Configuration

Key parameters in `rpm_config.yaml`:

```yaml
wp4:
  fusion:
    strategy: 'snr_max'  # Fusion algorithm
    min_sensors_required: 1
    agreement_threshold_rpm: 50
    
  interpolation:
    max_gap_s: 5.0
    method: 'median'
    
  validation:
    target_availability: 95.0  # %
    max_nan_fraction: 0.02    # 2%
```

## Algorithm Details

### Fusion Process

1. **Load Data**: Read WP-2 and WP-3 results for all sensors
2. **Method Selection**: Choose between Welch/STFT based on dynamics
3. **Time Alignment**: Resample to common time grid (0.25s)
4. **SNR Gating**: Exclude estimates below threshold
5. **Best Sensor**: Select highest SNR at each time
6. **Interpolation**: Fill small gaps with median
7. **Quality Flags**: Mark data quality and validity

### Fusion Strategies

- **snr_max** (default): Select sensor with highest SNR
- **median**: Take median of all valid sensors
- **weighted**: Weight by SNR in averaging

## Success Criteria

- ✅ Availability > 95% across experiments
- ✅ NaN fraction < 2% on RPM sweep
- ✅ Smooth sensor transitions
- ✅ Clear quality tracking

## Troubleshooting

### Low Availability
- Check WP-2/WP-3 results exist for all sensors
- Verify sensors are not clipping
- Consider lowering SNR threshold (with caution)

### Large Gaps
- Inspect raw data for sensor failures
- Check time alignment between sensors
- Increase interpolation window if appropriate

### Poor Agreement
- Review sensor mounting and orientation
- Check for sensor-specific noise sources
- Consider excluding problematic sensors

## Next Steps

After successful WP-4 completion:

1. **WP-5**: Validate against ground truth (if available)
2. **WP-6**: Batch process all experiments
3. **Integration**: Feed fused RPM to simulator comparison

## Performance

Typical processing times:
- Single experiment: 30-60 seconds
- Full session (30 experiments): 15-20 minutes
- Memory usage: ~500 MB peak

## References

- `vibration_plan.md`: Original fusion specifications
- `fusion.py`: Core fusion algorithms
- `wp4_process.py`: Main processing pipeline
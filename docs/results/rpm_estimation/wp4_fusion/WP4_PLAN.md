# WP-4: Multi-Sensor Fusion & Confidence Gating Implementation Plan

## Overview

Work Package 4 implements intelligent multi-sensor fusion to combine RPM estimates from multiple IMUs, producing a robust and continuous RPM time series with quality indicators.

## Implementation Timeline

- **Estimated Duration**: 0.5 days (as per vibration_plan.md)
- **Prerequisites**: WP-1, WP-2, and WP-3 completed
- **Target Output**: Fused RPM time series with <2% NaN frames

## Detailed Implementation Steps

### 1. Data Loading and Method Selection (1-2 hours)

#### 1.1 Create `wp4_process.py`
- Load WP-2 results (Welch PSD) from HDF5 files
- Load WP-3 results (STFT) from HDF5 files
- Implement intelligent method selection:
  - Use STFT for high-rate regions (>150 RPM/s)
  - Use Welch for steady-state regions
  - Blend methods in transition zones

#### 1.2 Data Alignment
- Ensure consistent time grids across sensors
- Handle different update rates (Welch: 30s, STFT: 0.25s)
- Create unified time series structure

### 2. Fusion Rules Implementation (2-3 hours)

#### 2.1 Core Fusion Rules (from vibration_plan.md)
- **R-1**: SNR Gating
  ```python
  # Discard estimates with SNR < 10 dB
  valid_mask = rpm_frame.snr_db >= config['snr_thresh_db']
  ```

- **R-2**: Best Sensor Selection
  ```python
  # Choose sensor with max SNR per epoch
  best_sensor = max(valid_sensors, key=lambda s: s.snr_db)
  ```

- **R-3**: Interpolation
  ```python
  # Median of last 5s for gaps
  if no_valid_sensors:
      rpm = median(last_5s_valid_rpms)
      quality = 'interpolated'
  ```

- **R-4**: Quality Flag
  ```python
  rpm_valid = (quality != 'interpolated') and (snr_db >= 10)
  ```

#### 2.2 Advanced Fusion Features
- **Sensor Agreement Score**: Std dev of valid sensors
- **Confidence Weighting**: Weight by SNR in averaging
- **Outlier Detection**: Median filter for spike removal

### 3. Output Generation (1-2 hours)

#### 3.1 Primary Output Format
```csv
time,rpm,snr_db,sensor_id,method,quality,rpm_valid
0.0,1800.5,15.2,fused_Sensor_3,welch,measured,true
0.25,1805.2,12.8,fused_Sensor_4,stft,measured,true
0.5,1810.0,8.5,fused_multi,interpolated,interpolated,false
```

#### 3.2 Metadata and Reports
- Fusion summary statistics
- Sensor contribution percentages
- Quality distribution analysis
- Method usage breakdown

### 4. Visualization (1 hour)

#### 4.1 Diagnostic Plots
- **Multi-panel time series**: RPM, SNR, sensor selection
- **Sensor contribution**: Stacked area chart
- **Quality indicators**: Color-coded background
- **Method transitions**: Vertical lines at switch points

### 5. CLI Integration (30 minutes)

#### 5.1 New CLI Options
```bash
# Basic fusion
python -m rpm_estimation.cli --wp 4 --exp 026_Engine_rpm_sweep --session afternoon

# Custom options
--fusion-strategy [snr_max|median|weighted]
--min-sensors [1-3]
--interpolation-window [seconds]
--save-intermediate
```

### 6. Testing Strategy (1-2 hours)

#### 6.1 Unit Tests
- Test each fusion rule independently
- Verify time alignment logic
- Test edge cases (no valid sensors, all sensors agree/disagree)

#### 6.2 Integration Tests
- Process 026_Engine_rpm_sweep
- Verify <2% NaN target
- Compare with individual sensor results

#### 6.3 Validation Metrics
- Availability percentage
- Inter-sensor agreement
- Transition smoothness
- Computational performance

## Configuration Updates

### rpm_config.yaml additions:
```yaml
wp4:
  # Fusion strategy
  fusion:
    strategy: 'snr_max'  # Options: snr_max, median, weighted
    min_sensors_required: 1
    agreement_threshold_rpm: 50
    
  # Interpolation
  interpolation:
    max_gap_s: 5.0
    method: 'median'  # Options: median, linear, spline
    lookback_window_s: 5.0
    
  # Quality control  
  quality:
    confidence_weights:
      measured: 1.0
      interpolated: 0.5
      extrapolated: 0.3
    
  # Method blending
  method_selection:
    steady_state_threshold: 50  # RPM/s
    blend_window_s: 2.0
    prefer_stft_above: 150  # RPM/s
    
  # Output
  output:
    format: 'csv'
    include_all_sensors: false
    save_fusion_report: true
```

## Risk Mitigation

### Potential Issues and Solutions

1. **Time Grid Mismatch**
   - Solution: Resample to common 0.25s grid
   - Fallback: Use nearest-neighbor matching

2. **All Sensors Invalid**
   - Solution: Extend interpolation window to 10s
   - Fallback: Mark as data gap, document in report

3. **Method Transition Artifacts**
   - Solution: Implement smooth blending over 2s window
   - Fallback: Hard switch with median filter

4. **Memory Usage with Multiple HDF5**
   - Solution: Process in chunks, one experiment at a time
   - Fallback: Reduce to essential data only

## Success Metrics

- [ ] <2% NaN frames on 026_Engine_rpm_sweep
- [ ] >95% availability across all experiments
- [ ] <50 RPM std dev between sensors when all valid
- [ ] <1 minute processing time per experiment
- [ ] Zero data loss from WP-2/WP-3 inputs

## Next Steps After WP-4

1. **WP-5**: Validation against ground truth
2. **WP-6**: Batch processing all experiments
3. **Integration**: Feed fused RPM to simulator comparison
# RPM Estimation Development Checklist

## Work Package Status

### ✅ WP-0: Repository & Config Scaffold (COMPLETED - 2025-06-19)
- [x] Create module structure
- [x] Define rpm_config.yaml schema
- [x] Implement RPMFrame dataclass with validation
- [x] Create CLI skeleton with full argument parser
- [x] Write unit tests (config, dataclass, I/O, imports)
- [x] Setup GitHub Actions CI workflow
- [x] Create documentation (README, this checklist, WP0_PLAN)
- [x] Verify all files created and imports work
- [x] Add RPMTimeSeries dataclass for time series management
- [x] Include placeholder implementations for all modules

### ⏳ WP-1: Raw Data Audit & Orientation (Not Started)
- [ ] Load CSV data via io.py
- [ ] Apply rotation matrices from orientation_config.yaml
- [ ] Compute vibration magnitude |a_body|
- [ ] Implement 5 Hz high-pass filter
- [ ] Calculate quality metrics (RMS, kurtosis)
- [ ] Generate proc_IMU_<id>.parquet files
- [ ] Validate with synthetic 25 Hz sine test

### ⏳ WP-2: Welch PSD Core (Not Started)
- [ ] Implement welch_psd() in spectral.py
- [ ] Add peak detection algorithm
- [ ] Calculate SNR metric
- [ ] Extract harmonics
- [ ] Unit tests with white noise and synthetic signals

### ⏳ WP-3: STFT + Order Tracking (Not Started)
- [ ] Implement stft_mag() in spectral.py
- [ ] Time-resolved RPM extraction
- [ ] Optional Vold-Kalman order tracking
- [ ] Generate HDF5 outputs

### ⏳ WP-4: Multi-Sensor Fusion (Not Started)
- [ ] SNR-based sensor selection
- [ ] Confidence gating logic
- [ ] Interpolation for invalid frames
- [ ] Generate fused RPM series

### ⏳ WP-5: Validation & Blind Test (Not Started)
- [ ] Comparison metrics (MAE, RMSE)
- [ ] Visualization plots
- [ ] CLI integration
- [ ] Blind test on 026_Engine_rpm_sweep

### ⏳ WP-6: Batch Processing (Not Started)
- [ ] Process all experiments
- [ ] Generate quality overview
- [ ] Flag problematic maneuvers

## Testing Status

- [x] Unit tests pass
- [ ] Integration tests pass
- [ ] Validation against ground truth
- [ ] Performance benchmarks

## Documentation Status

- [x] README.md created
- [x] Development checklist created
- [x] WP-0 plan documented
- [ ] API documentation
- [ ] Results documentation

## Key Findings & Notes

### From Orientation Analysis
- Sensors 3, 4, wb: Validated with <3.5° error
- Sensor 5: Has 40° physical mounting angle (steering wheel)
- High vibrations present: 2-11 rad/s even in "static" experiments
- Lift fans create continuous vibrations - perfect for RPM extraction

### Data Specifications
- Sampling rate: 200 Hz
- Data format: CSV with time, x/y/z accel (m/s²), gyro (rad/s)
- Aligned data available in: `/hovercraft_data_analysis/alignment_analysis/aligned_data/`

### Parameter Selection Guidelines
- Window length (Welch): 4-8s for frequency resolution
- Overlap: 50-75% for variance reduction
- HP cutoff: 5 Hz to remove quasi-static motion
- SNR threshold: 10 dB based on literature

## Next Implementation Steps

1. **WP-1 Priority Tasks**:
   - Set up data loading from aligned CSV files
   - Implement vibration magnitude calculation
   - Design quality metric calculations

2. **Critical Path Items**:
   - Welch PSD implementation (WP-2)
   - Peak detection with harmonic handling
   - Multi-sensor fusion logic

3. **Validation Requirements**:
   - Synthetic test signals
   - Comparison with 026_Engine_rpm_sweep
   - Cross-sensor consistency checks
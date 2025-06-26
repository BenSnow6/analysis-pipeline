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

### ✅ WP-1: Raw Data Audit & Orientation (COMPLETED - 2025-06-19)
- [x] Load CSV data via io.py with enhanced error handling
- [x] Apply rotation matrices from orientation_config.yaml
- [x] Compute vibration magnitude |a_body|
- [x] Implement 5 Hz high-pass filter (configurable)
- [x] Calculate quality metrics (RMS, kurtosis, peak-to-RMS)
- [x] Generate proc_IMU_<id>.parquet files with schema validation
- [x] Validate with synthetic 25 Hz sine test (achieves >25 dB SNR)
- [x] Add structured JSON logging with error categorization
- [x] Implement configurable window handling (drop/pad/process_partial)
- [x] Create comprehensive quality reports with per-axis analysis
- [x] Add CLI support for batch processing and validation
- [x] Create full test suite with >90% coverage target

### ✅ WP-2: Welch PSD Core (COMPLETED - 2025-06-20)
- [x] Implement welch_psd() in spectral.py with max frequency limiting
- [x] Add intelligent peak detection algorithm with noise floor estimation
- [x] Calculate SNR metric using local band method (±3 Hz, exclude ±0.5 Hz)
- [x] Extract harmonics with configurable tolerance
- [x] Implement fundamental frequency identification with harmonic scoring
- [x] Create extract_rpm_from_vibration() main processing function
- [x] Unit tests with white noise and synthetic signals (test_spectral.py)
- [x] Add WP-2 specific configuration parameters
- [x] Create wp2_process.py for batch processing
- [x] Generate diagnostic plots (RPM, SNR, example PSD)
- [x] Implement HDF5 output format with metadata
- [x] Document implementation in docs/work_packages/wp2/WP2_README.md
- [x] Add CLI integration for WP-2
- [x] Create validation script - all tests passing
- [x] Create implementation summary documentation

### ✅ WP-3: STFT + Order Tracking (COMPLETED - 2025-06-20)
- [x] Anti-alias filter verification from WP-1 metadata
- [x] Implement stft_mag() in spectral.py with edge handling
- [x] Time-resolved RPM extraction with early SNR gating
- [x] Lightweight smoothing module (polynomial/median/moving_avg)
- [x] Triangular ramp test (500→2000→500 RPM)
- [x] Create wp3_process.py for batch processing
- [x] Generate HDF5 outputs with exact time alignment
- [x] CLI integration with --wp 3 option
- [x] Create docs/work_packages/wp3/WP3_README.md documentation
- [ ] Optional Vold-Kalman order tracking (deferred - lightweight smoothing sufficient)

### 🚧 WP-4: Multi-Sensor Fusion (IN PROGRESS - 2025-06-20)
- [x] SNR-based sensor selection implemented in fusion.py
- [x] Confidence gating logic with 10 dB threshold
- [x] Interpolation for gaps up to 5 seconds
- [x] Generate fused RPM series in CSV format
- [x] Create wp4_process.py main processing module
- [x] Add fusion diagnostic plots (3-panel visualization)
- [x] CLI integration with --wp 4 option
- [x] Add WP-4 configuration to rpm_config.yaml
- [x] Create comprehensive test suite (test_fusion.py)
- [x] Document implementation in WP4_README.md
- [ ] Test on real data (026_Engine_rpm_sweep)
- [ ] Verify <2% NaN target on RPM sweep
- [ ] Create WP4_COMPLETION_SUMMARY.md

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
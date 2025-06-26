# WP-4 Test Plan Summary

## Current Status (2025-06-20)

### Prerequisites Check
- ✅ WP-4 Implementation complete (wp4_process.py, fusion.py, cli.py)
- ✅ WP-2 Results available for:
  - 003_Waiting_for_departure (Sensors 3, 4, wb)
  - 007_Fast_stbd_turn_1 (Sensors 3, 4, wb)  
  - 026_Engine_rpm_sweep (Sensors 3, 4, wb)
- ❌ WP-3 Results not yet generated (required for fusion)
- ❌ Python environment issue preventing immediate testing

### Data Locations
- WP-2 results: `results/wp2/afternoon/*_rpm.h5/*_rpm.h5`
- WP-3 results: `results/wp3/afternoon/` (empty - needs generation)
- WP-4 output: `results/wp4/afternoon/` (created, ready for results)

## Test Execution Plan

### Phase 1: Environment Setup & WP-3 Generation
```bash
# 1. Fix Python environment (install numpy, scipy, etc.)
# 2. Generate WP-3 results for test experiments:
python3 -m rpm_estimation.cli --wp 3 --exp 003_Waiting_for_departure --session afternoon
python3 -m rpm_estimation.cli --wp 3 --exp 007_Fast_stbd_turn_1 --session afternoon
python3 -m rpm_estimation.cli --wp 3 --exp 026_Engine_rpm_sweep --session afternoon
```

### Phase 2: Core WP-4 Testing

#### Test 1: Static Conditions (003_Waiting_for_departure)
- **Purpose**: Test fusion on steady-state data
- **Expected Results**:
  - Steady RPM around 700-800 (idle)
  - High SNR (>15 dB) on all sensors
  - Minimal sensor switching
  - >98% data availability
  - Method: Primarily Welch (steady state)

#### Test 2: Dynamic Maneuver (007_Fast_stbd_turn_1)  
- **Purpose**: Test fusion during vehicle dynamics
- **Expected Results**:
  - RPM variations during turn
  - Possible sensor SNR variations
  - Smooth sensor transitions
  - >95% data availability
  - Method: Mix of Welch and STFT

#### Test 3: RPM Sweep (026_Engine_rpm_sweep) - CRITICAL
- **Purpose**: Test fusion on transient data
- **Expected Results**:
  - Smooth RPM ramp from ~700 to ~2400 RPM
  - <2% NaN frames (key success metric)
  - >95% data availability
  - Method: Primarily STFT (transient)
  - Clear sensor handoffs as SNR changes

### Phase 3: Output Validation

#### Expected Output Files
1. **rpm_fused.csv** - Primary output
   - Columns: time, rpm, snr_db, sensor_id, method, quality, rpm_valid
   - No missing columns or malformed data
   - Reasonable RPM values (600-3000)

2. **fusion_report.json** - Quality metrics
   - sensor_contributions (should sum to ~1.0)
   - availability > 95%
   - interpolated_fraction < 0.02
   - mean_snr_db > 10

3. **fusion_diagnostic.png** - Visualization
   - Three panels: RPM, SNR, Sensor selection
   - Clear visualization of fusion process

### Phase 4: Advanced Testing

#### Fusion Strategy Tests
- Test `--fusion-strategy median` (robust to outliers)
- Test `--min-sensors 2` (stricter requirements)
- Test `--interpolation-window 3.0` (shorter gap filling)

#### Edge Case Tests
- Experiment with all sensors below SNR threshold
- Experiment with sensor dropout scenarios
- Large transient rates (>150 RPM/s)

### Phase 5: Success Criteria Validation

| Criterion | Target | Test Method |
|-----------|--------|-------------|
| NaN fraction (026) | <2% | Check rpm_fused.csv |
| Availability | >95% | fusion_report.json |
| Processing time | <60s/exp | Time CLI execution |
| Smooth transitions | Visual | Check diagnostic plots |
| Sensor agreement | <50 RPM std | When multiple valid |

## Known Issues & Solutions

### Issue 1: Python Dependencies
- **Problem**: numpy, scipy not installed in test environment
- **Solution**: Create venv or use system packages

### Issue 2: Missing WP-3 Data
- **Problem**: STFT results required for fusion
- **Solution**: Run WP-3 processing first

### Issue 3: Data Path Structure
- **Problem**: H5 files in subdirectories (e.g., `*_rpm.h5/*_rpm.h5`)
- **Solution**: Ensure wp4_process.py handles this path structure

## Test Verification Checklist

- [ ] WP-3 results generated for all test experiments
- [ ] WP-4 runs without errors on 003 (static)
- [ ] WP-4 runs without errors on 007 (dynamic)
- [ ] WP-4 runs without errors on 026 (sweep)
- [ ] 026 achieves <2% NaN frames
- [ ] All experiments achieve >95% availability
- [ ] Fusion reports generated correctly
- [ ] Diagnostic plots show reasonable behavior
- [ ] Integration tests pass (test_wp4_integration.py)
- [ ] Unit tests pass (pytest tests/test_fusion.py)

## Next Steps

1. Resolve Python environment issues
2. Generate WP-3 results
3. Execute test plan phases 2-5
4. Document actual results in WP4_TEST_RESULTS.md
5. Create WP4_COMPLETION_SUMMARY.md
6. Proceed to WP-5 (validation)
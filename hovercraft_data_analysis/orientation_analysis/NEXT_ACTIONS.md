# Orientation Analysis - Next Actions

**Updated**: 2025-06-19 (Post Session 2 - Most Actions Completed)
**Purpose**: Quick reference for remaining tasks

## ✅ COMPLETED Actions (2025-06-19)

### 1. ✅ Processed Static Experiments
- Updated orientation_check.py with static paths
- Converted HDF5 to CSV for 002, 003, 010
- Added gyroscope data to all files
- **RESULT**: Discovered bias estimation impossible due to fan vibrations

### 2. ✅ Re-ran 021_Quarter_turn_port
- Successfully validated with new rotation matrices
- All sensors show expected performance

### 3. ✅ Documented Sensor_5 Physical Configuration
- Added physical_mounting_note to config
- Added expected_static_error_deg: 40.0
- Confirmed across all experiments

## 🔴 CRITICAL DISCOVERIES

### Bias Estimation Completely Failed
- Static detection threshold (0.05 rad/s) impossible to meet
- "Static" experiments show 2-11 rad/s from fan vibrations
- **MUST USE ONLINE BIAS ESTIMATION IN KALMAN FILTER**

### Vibration Environment
- Hovercraft lift fans create continuous vibrations
- Traditional static calibration methods won't work
- Consider pre-flight calibration with fans OFF

## For Kalman Filter Implementation

### Rotation Matrices to Use:
| Sensor | Source | Error | Notes |
|--------|--------|-------|-------|
| Sensor_3 | orientation_config.yaml | 1-2° | Validated ✅ |
| Sensor_4 | orientation_config.yaml | 1-2° | Validated ✅ |
| Sensor_5 | orientation_config.yaml | 40° | Physical tilt - needs compensation |
| Sensor_wb | orientation_config.yaml | 2-3° | Validated ✅ |

### Bias Values:
- Current estimates show 0.0000 (unreliable)
- Either:
  1. Process static experiments first, OR
  2. Use online bias estimation in Kalman filter

### Data Units:
- Accelerometer: m/s² (converted from g's)
- Gyroscope: rad/s
- All transformations: Use R_bs.T to go from sensor to body frame

## Optional Improvements

1. **Relax Dynamic Validation Thresholds**
   - Current: forward_accel > 0.3 m/s²
   - Consider: forward_accel > 0.2 m/s²
   - Add noise tolerance

2. **Create Sensor_5 Tilt Compensation**
   ```python
   # Create additional rotation for 40° tilt
   tilt_angle = 40 * np.pi / 180
   R_tilt = create_rotation_matrix_x(tilt_angle)
   R_bs_sensor5_compensated = R_bs_sensor5 @ R_tilt
   ```

3. **Add CSV Export for Validation Results**
   - Export rotation matrices for easy import to Kalman filter
   - Export validation metrics for documentation

## Success Criteria

✅ **Already Achieved**:
- Unit conversion working
- Rotation matrices validated for 3/4 sensors
- Sensor configurations corrected
- Static experiments now loading from correct directory
- All experiments have gyroscope data added
- 021_Quarter_turn_port results updated with fixes
- Sensor_5's 40° physical mounting documented
- Comprehensive validation completed across multiple experiments

⚠️ **Critical Limitations Discovered**:
- Bias estimates cannot be obtained (continuous vibrations prevent static detection)
- All bias values show 0.0000 m/s² - MUST use online estimation in Kalman filter
- Dynamic validation thresholds too strict for noisy environment

## Quick Test Commands

```bash
# Test static experiment loading
python run_orientation.py -e 002_Setup

# Re-run quarter turn
python run_orientation.py -e 021_Quarter_turn_port

# Full validation suite
python run_orientation.py -e 002_Setup 007_Fast_stbd_turn_1 016_Straight_cruise_1 021_Quarter_turn_port
```

## Final Summary - Session 2 Complete

### What We Accomplished:
1. ✅ Fixed static experiment loading (added subdirectory paths)
2. ✅ Processed all static experiments with gyro data
3. ✅ Discovered why bias estimation fails (continuous vibrations)
4. ✅ Re-validated all experiments with comprehensive results
5. ✅ Documented Sensor_5's physical mounting angle
6. ✅ Updated all documentation files

### Key Takeaways for Next Person:
1. **Rotation matrices are GOOD** - Use from orientation_config.yaml
2. **Bias estimation is BROKEN** - Must use online estimation
3. **Sensor_5 has 40° PHYSICAL tilt** - Not an error!
4. **Vibrations are CONTINUOUS** - From lift fans
5. **All data units CORRECT** - m/s² and rad/s

### Ready for Week 2 Kalman Filtering ✅
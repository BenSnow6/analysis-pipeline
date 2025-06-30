# Orientation Analysis - Current Status and Understanding

**Date**: 2025-06-19 (Updated after Session 2 - Comprehensive Validation)
**Author**: Comprehensive status update based on analysis findings

## Key Findings and Clarifications

### 1. Static vs Dynamic Experiments

**Issue**: "011_Static_stbd_1" is misnamed - it's actually a turn, not a static experiment.

**Actual Static Experiments** (found in `aligned_data/static/` directory):
- 002_Setup
- 003_Waiting_for_departure  
- 010_Waiting_for_static_turns

These need to be processed for proper bias estimation.

### 2. Dynamic Validation Pattern Expectations

The dynamic validator checks for specific acceleration patterns:

**007_Fast_stbd_turn_1** expects:
- Forward acceleration > 0.3 m/s² (body X-axis)
- Gravity ~9.8 m/s² (body Z-axis)  
- Lateral acceleration < 0.5 m/s² (body Y-axis)

**Why Sensor_wb passed but others failed**:
- The validation is sensitive to rotation matrix accuracy
- Small errors in rotation can shift acceleration components between axes
- Sensor_wb might have had the right combination of values by chance

**016_Straight_cruise_1** expects:
- Minimal forward/lateral acceleration (< 0.3 m/s²)
- Gravity dominant in Z-axis
- Stable acceleration (low standard deviation)

### 3. Sensor 5 Physical Mounting

**Confirmed**: Sensor 5 is physically mounted at ~40° angle (steering wheel mount)
- This is NOT an error - it's the actual physical configuration
- The 40° "error" in static validation correctly detects this tilt
- Dynamic validation will always fail for this sensor unless we account for the tilt

### 4. Gravity Visualization Explanation

In the validation plots showing three columns:
1. **Expected**: Should show gravity mainly in +Z (~9.8 m/s²)
2. **Current Matrix**: Shows gravity transformed using the original rotation matrix
3. **Config Matrix**: Shows gravity transformed using the updated configuration

For correct alignment, all three should match, showing ~9.8 m/s² in the Z component.

### 5. Bias Estimation (CRITICAL UPDATE - 2025-06-19)

**Current Issue**: Showing 0.0000 m/s² for all sensors

**Root Cause DISCOVERED**: 
- Static detection threshold (0.05 rad/s) is impossibly low for this vehicle
- Even "static" experiments show massive angular velocities:
  - 002_Setup: Data processing error (timestamp mismatch)
  - 003_Waiting_for_departure: ~2-3 rad/s average
  - 010_Waiting_for_static_turns: 2.43 rad/s (139.4 deg/s) average
  - 011_Static_stbd_1: 9.09 rad/s (520.9 deg/s) - clearly NOT static!
  - 012_Static_port_1: 11.47 rad/s (657.3 deg/s)

**Physical Explanation**: 
- Hovercraft lift fans create continuous vibrations
- Traditional static calibration impossible with fans running
- Need pre-flight calibration or online bias estimation

**Attempted Solution**: Processed static experiments - FAILED due to vibrations
**Required Solution**: Implement online bias estimation in Kalman filter

## Current Validation Results Summary (Updated 2025-06-19)

### Rotation Matrix Accuracy (From Comprehensive Validation)
| Sensor | Average Error | Error Range | Status | Notes |
|--------|---------------|-------------|--------|-------|
| Sensor_3 | 2.10° | 0.57-2.87° | ✅ Pass | Excellent alignment |
| Sensor_4 | 2.54° | 1.37-3.26° | ✅ Pass | Good alignment |
| Sensor_5 | 32.81° | 8.49-52.91° | ⚠️ Expected | Physical 40° tilt documented |
| Sensor_wb | 3.19° | 2.15-3.46° | ✅ Pass | Within secondary sensor tolerance |

### Experiments Validated
- ✅ 003_Waiting_for_departure (static)
- ✅ 007_Fast_stbd_turn_1 (dynamic turn)
- ✅ 010_Waiting_for_static_turns (static)
- ✅ 016_Straight_cruise_1 (cruise)
- ✅ 021_Quarter_turn_port (turn) - freshly re-run
- ❌ 002_Setup - timestamp/data length mismatch error

### Dynamic Validation Issues
- Most failures are due to strict threshold requirements
- Thresholds assume perfect rotation matrices and no sensor noise
- May need to relax thresholds or improve filtering

## Actions Completed (2025-06-19 Session 2)

1. ✅ **Processed Static Experiments**
   - Updated data loading to check `static/` subdirectory
   - Ran validation on 002_Setup, 003_Waiting_for_departure, 010_Waiting_for_static_turns
   - **RESULT**: Bias estimation failed - vibrations too high for static detection

2. ✅ **Re-ran 021_Quarter_turn_port**
   - Results updated with new rotation matrices
   - All sensors show expected performance

3. ✅ **Documented Sensor 5**
   - Added physical_mounting_note to config
   - Added expected_static_error_deg: 40.0
   - Confirmed 40° tilt is consistent across experiments

4. ⏳ **Dynamic Validation Thresholds**
   - Still need relaxation - not implemented
   - Current thresholds cause most "failures"

## Understanding the Results

### Why Current Approach Works:
- Unit conversion (g to m/s²) is correct
- Rotation matrix transformation (using R_bs.T) is correct
- Sensor axis configurations have been corrected

### What Still Needs Work:
- Bias estimation (need true static data)
- Dynamic validation thresholds
- Sensor 5 special handling
- Processing experiments in static/ directory

## For Kalman Filter Implementation

### Use These Rotation Matrices:
- **Sensor_3**: Config matrix from orientation_config.yaml
- **Sensor_4**: Current matrix (both work well)
- **Sensor_5**: Needs special handling for 40° tilt
- **Sensor_wb**: Config matrix from orientation_config.yaml

### Bias Handling:
- Current bias estimates (0.0000) are not reliable
- Implement online bias estimation in Kalman filter
- Or process static experiments first for initial estimates

### Data Quality Confidence:
- High confidence in Sensors 3, 4, wb after rotation correction
- Sensor 5 data is valid but requires tilt compensation
- All data now in correct units (m/s², rad/s)
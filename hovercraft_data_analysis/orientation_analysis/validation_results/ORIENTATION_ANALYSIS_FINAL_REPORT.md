# Orientation Analysis - Final Report
**Generated**: 2025-06-19 13:36:22

## Executive Summary

- **Total Validation Tests**: 12
- **Tests Passed**: 0
- **Overall Pass Rate**: 0.0%
- **Experiments Analyzed**: 4

## Sensor Performance Summary

| Sensor | Pass Rate | Avg Rotation Error (°) | Avg Bias Magnitude (m/s²) |
|--------|-----------|------------------------|---------------------------|
| Sensor_3 | 0% | 2.10 | 0.0000 |
| Sensor_4 | 0% | 2.54 | 0.0000 |
| Sensor_5 | 0% | 32.81 | 0.0000 |
| Sensor_wb | 0% | 3.19 | 0.0000 |

## Detailed Results by Experiment

### 002_Setup

**ERROR**: x and y must have same first dimension, but have shapes (17201,) and (17200,)

### 007_Fast_stbd_turn_1

| Sensor | Rotation Error | Static | Bias | Dynamic | Overall |
|--------|----------------|--------|------|---------|---------|
| Sensor_3 | 2.15° | ✅ | ❌ | ❌ | ❌ |
| Sensor_4 | 2.52° | ✅ | ❌ | ❌ | ❌ |
| Sensor_5 | 25.73° | ❌ | ❌ | ❌ | ❌ |
| Sensor_wb | 3.31° | ✅ | ❌ | ✅ | ❌ |

### 016_Straight_cruise_1

| Sensor | Rotation Error | Static | Bias | Dynamic | Overall |
|--------|----------------|--------|------|---------|---------|
| Sensor_3 | 1.28° | ✅ | ❌ | ❌ | ❌ |
| Sensor_4 | 1.84° | ✅ | ❌ | ❌ | ❌ |
| Sensor_5 | 42.88° | ❌ | ❌ | ❌ | ❌ |
| Sensor_wb | 3.22° | ✅ | ❌ | ❌ | ❌ |

### 021_Quarter_turn_port

| Sensor | Rotation Error | Static | Bias | Dynamic | Overall |
|--------|----------------|--------|------|---------|---------|
| Sensor_3 | 2.87° | ✅ | ❌ | ❌ | ❌ |
| Sensor_4 | 3.26° | ❌ | ❌ | ❌ | ❌ |
| Sensor_5 | 29.80° | ❌ | ❌ | ❌ | ❌ |
| Sensor_wb | 3.04° | ✅ | ❌ | ❌ | ❌ |

## Recommendations

Based on the orientation validation results:

⚠️ **Attention Required**: The following sensors showed validation issues:

- Sensor_3
- Sensor_4
- Sensor_5
- Sensor_wb

### Next Steps:
1. Review rotation matrices for any sensors with errors > 3°
2. Apply bias corrections before Kalman filtering
3. Consider excluding sensors with persistent validation failures
4. Use the validated rotation matrices and bias estimates in Week 2 analysis

## Data Quality Certificate

✅ **Temporal Alignment**: Complete (Week 1 Day 1)
⚠️ **Orientation Validation**: 0% Pass Rate
✅ **Ready for Kalman Filtering**: Review required
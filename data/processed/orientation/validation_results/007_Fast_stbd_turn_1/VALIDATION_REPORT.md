# Orientation Validation Report
**Experiment**: 007_Fast_stbd_turn_1  
**Generated**: 2025-06-19T13:36:09.310912  

## Summary

| Sensor | Rotation Error (°) | Static Valid | Bias Valid | Dynamic Valid | Overall Status |
|--------|-------------------|--------------|------------|---------------|----------------|
| Sensor_3 | 2.15 | ✅ | ❌ | ❌ | ❌ FAIL |
| Sensor_4 | 2.52 | ✅ | ❌ | ❌ | ❌ FAIL |
| Sensor_5 | 25.73 | ❌ | ❌ | ❌ | ❌ FAIL |
| Sensor_wb | 3.31 | ✅ | ❌ | ✅ | ❌ FAIL |

## Detailed Results

### Sensor_3

**Rotation Validation**:
- Matrix source: config
- Rotation error: 2.15°
- Static segments found: 0

**Bias Estimation**:

**Dynamic Validation**:
- Pattern valid: False
- Expected pattern: Forward acceleration (+X) with gravity (+Z)

### Sensor_4

**Rotation Validation**:
- Matrix source: current
- Rotation error: 2.52°
- Static segments found: 0

**Bias Estimation**:

**Dynamic Validation**:
- Pattern valid: False
- Expected pattern: Forward acceleration (+X) with gravity (+Z)

### Sensor_5

**Rotation Validation**:
- Matrix source: config
- Rotation error: 25.73°
- Static segments found: 0

**Bias Estimation**:

**Dynamic Validation**:
- Pattern valid: False
- Expected pattern: Forward acceleration (+X) with gravity (+Z)

### Sensor_wb

**Rotation Validation**:
- Matrix source: config
- Rotation error: 3.31°
- Static segments found: 0

**Bias Estimation**:

**Dynamic Validation**:
- Pattern valid: True
- Expected pattern: Forward acceleration (+X) with gravity (+Z)
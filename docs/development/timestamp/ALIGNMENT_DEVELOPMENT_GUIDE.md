# Data Alignment Development Guide

## Overview

This document provides guidance for developing the `align.py` module based on the timestamp analysis results. The alignment algorithm will synchronize multi-rate sensor data for subsequent processing and simulator validation.

## Timestamp Analysis Summary

### Sensor Characteristics
| Sensor | Actual Rate | Timing Quality | Recommendation |
|--------|-------------|----------------|----------------|
| GPS | 1 Hz | Good (occasional jitter) | Interpolate to higher rate |
| Sensor_3 | 200 Hz | Excellent | **Use as primary time reference** |
| Sensor_4 | 200 Hz | Excellent | Alternative time reference |
| Sensor_5 | 200 Hz | Excellent | Direct alignment |
| Sensor_wb | 100 Hz | Excellent | 2:1 ratio with reference |
| Sensor_wnb | ~7.5 Hz | Poor | Exclude or heavily filter |

## Recommended Alignment Strategy

### 1. Time Reference Selection
```python
# Pseudocode for reference selection
def select_time_reference(sensor_data):
    # Priority order based on timing analysis
    reference_priority = ['sensor_3', 'sensor_4', 'sensor_5', 'sensor_wb']
    
    for sensor in reference_priority:
        if sensor in sensor_data and validate_timing(sensor_data[sensor]):
            return sensor
    
    raise ValueError("No suitable time reference found")
```

**Rationale**: Sensor_3 and Sensor_4 show perfect 200Hz timing with zero jitter, making them ideal time references.

### 2. Multi-Rate Synchronization Approach

#### High-Rate IMU Sensors (200Hz)
- **Sensors**: sensor_3, sensor_4, sensor_5
- **Method**: Direct timestamp matching with reference
- **Tolerance**: ±2.5ms (half of 5ms interval)

#### Medium-Rate IMU Sensor (100Hz)
- **Sensor**: sensor_wb
- **Method**: Every 2nd sample of 200Hz reference
- **Interpolation**: Not needed - exact 2:1 ratio

#### Low-Rate GPS (1Hz)
- **Sensor**: GPS
- **Method**: Linear interpolation between GPS samples
- **Alternative**: Zero-order hold for position, linear for velocity

#### Degraded Sensor (sensor_wnb)
- **Current Rate**: ~7.5Hz (highly irregular)
- **Options**:
  1. Exclude from analysis
  2. Nearest-neighbor matching with quality flag
  3. Attempt recovery through filtering

### 3. Implementation Architecture

```python
class DataAligner:
    def __init__(self, reference_sensor='sensor_3', target_rate=200):
        self.reference_sensor = reference_sensor
        self.target_rate = target_rate
        self.time_tolerance = 1000.0 / target_rate / 2  # Half interval in ms
    
    def align_data(self, sensor_data_dict):
        """
        Align all sensors to reference time base.
        
        Args:
            sensor_data_dict: Dict[sensor_name, DataFrame with 'time_from_sync' column]
            
        Returns:
            aligned_data: Dict[sensor_name, DataFrame with unified timestamps]
        """
        # Get reference timestamps
        ref_data = sensor_data_dict[self.reference_sensor]
        ref_timestamps = ref_data['time_from_sync'].values
        
        aligned_data = {}
        
        for sensor_name, data in sensor_data_dict.items():
            if sensor_name == self.reference_sensor:
                aligned_data[sensor_name] = data
            else:
                aligned_data[sensor_name] = self._align_sensor(
                    data, ref_timestamps, sensor_name
                )
        
        return aligned_data
    
    def _align_sensor(self, sensor_data, ref_timestamps, sensor_name):
        """Align single sensor to reference timestamps."""
        # Implementation depends on sensor characteristics
        pass
```

### 4. Alignment Algorithms by Sensor Type

#### GPS Alignment (1Hz → 200Hz)
```python
def align_gps(gps_data, ref_timestamps):
    """
    Upsample GPS data to reference rate.
    
    Strategy:
    - Position: Linear interpolation in ECEF coordinates
    - Velocity: Linear interpolation if available
    - Quality metrics: Propagate from nearest GPS sample
    """
    from scipy.interpolate import interp1d
    
    # Convert lat/lon to ECEF for linear interpolation
    ecef_coords = latlon_to_ecef(gps_data[['Lat', 'Lng', 'Alt']])
    
    # Create interpolators
    interp_x = interp1d(gps_data['time_from_sync'], ecef_coords[:, 0], 
                        kind='linear', fill_value='extrapolate')
    interp_y = interp1d(gps_data['time_from_sync'], ecef_coords[:, 1], 
                        kind='linear', fill_value='extrapolate')
    interp_z = interp1d(gps_data['time_from_sync'], ecef_coords[:, 2], 
                        kind='linear', fill_value='extrapolate')
    
    # Interpolate to reference timestamps
    aligned_ecef = np.column_stack([
        interp_x(ref_timestamps),
        interp_y(ref_timestamps),
        interp_z(ref_timestamps)
    ])
    
    # Convert back to lat/lon
    aligned_latlon = ecef_to_latlon(aligned_ecef)
    
    return aligned_latlon
```

#### IMU Alignment (100/200Hz)
```python
def align_imu(imu_data, ref_timestamps, sensor_rate):
    """
    Align IMU data to reference timestamps.
    
    Strategy:
    - 200Hz sensors: Direct matching within tolerance
    - 100Hz sensors: Match every 2nd reference timestamp
    """
    if sensor_rate == 200:
        # Find nearest neighbor matches
        aligned_indices = find_nearest_timestamps(
            imu_data['time_from_sync'], 
            ref_timestamps,
            tolerance_ms=2.5
        )
    elif sensor_rate == 100:
        # Take every 2nd reference timestamp
        ref_subset = ref_timestamps[::2]
        aligned_indices = find_nearest_timestamps(
            imu_data['time_from_sync'], 
            ref_subset,
            tolerance_ms=5.0
        )
    
    return imu_data.iloc[aligned_indices]
```

### 5. Quality Control and Validation

#### Timing Quality Metrics
```python
def validate_alignment(aligned_data, reference_sensor):
    """
    Validate alignment quality.
    
    Checks:
    1. No duplicate timestamps
    2. Consistent time intervals
    3. No data gaps > threshold
    4. Cross-correlation of similar signals
    """
    metrics = {}
    
    # Check timestamp consistency
    ref_times = aligned_data[reference_sensor]['time_from_sync']
    intervals = np.diff(ref_times)
    
    metrics['mean_interval'] = np.mean(intervals)
    metrics['std_interval'] = np.std(intervals)
    metrics['max_gap'] = np.max(intervals)
    
    # Check data completeness
    for sensor, data in aligned_data.items():
        metrics[f'{sensor}_missing'] = len(ref_times) - len(data)
    
    return metrics
```

### 6. Gap Handling Strategies

Based on the timestamp analysis, implement gap handling:

```python
def handle_gaps(data, max_gap_ms):
    """
    Handle gaps in sensor data.
    
    Strategies by gap size:
    - < 2x expected interval: Linear interpolation
    - 2-5x expected interval: Marker insertion + interpolation
    - > 5x expected interval: Split into segments
    """
    gaps = find_gaps(data['time_from_sync'], max_gap_ms)
    
    for gap in gaps:
        if gap['duration'] < 2 * expected_interval:
            # Linear interpolation
            interpolate_gap(data, gap)
        elif gap['duration'] < 5 * expected_interval:
            # Insert NaN markers
            insert_gap_markers(data, gap)
        else:
            # Split data into segments
            segments.append(split_at_gap(data, gap))
    
    return data
```

### 7. Output Format

The aligned data should maintain traceability:

```python
# Suggested output structure
aligned_output = {
    'metadata': {
        'reference_sensor': 'sensor_3',
        'target_rate_hz': 200,
        'alignment_method': 'nearest_neighbor',
        'timestamp': datetime.now().isoformat()
    },
    'timestamps': unified_timestamps,  # Common time vector
    'data': {
        'gps': {
            'values': aligned_gps_data,
            'quality': gps_quality_flags,
            'original_rate_hz': 1
        },
        'sensor_3': {
            'values': sensor3_data,
            'quality': sensor3_quality,
            'original_rate_hz': 200
        },
        # ... other sensors
    }
}
```

## Testing Strategy

### Unit Tests
1. **Perfect alignment**: Synthetic data with exact timestamps
2. **Jittered data**: Add realistic jitter based on analysis results
3. **Missing data**: Test gap handling
4. **Edge cases**: Start/end alignment, single sample sensors

### Integration Tests
1. **Full experiment alignment**: Use actual experiment data
2. **Cross-validation**: Compare interpolated GPS with high-rate IMU
3. **Performance**: Ensure < 1 second processing for 5-minute experiments

### Validation Metrics
1. **Temporal alignment error**: < 2.5ms for 200Hz sensors
2. **Interpolation error**: Minimize for GPS upsampling
3. **Data preservation**: No loss of valid samples
4. **Computational efficiency**: Real-time capable

## Example Usage

```python
# Load data using existing data_loader
from hovercraft_data_analysis.timestamp_analysis import data_loader

# Get experiment data
experiment_path = "path/to/experiment"
sensor_data = data_loader.load_experiment_data(experiment_path)

# Initialize aligner
aligner = DataAligner(reference_sensor='sensor_3', target_rate=200)

# Perform alignment
aligned_data = aligner.align_data(sensor_data)

# Validate results
quality_metrics = aligner.validate_alignment(aligned_data)

# Save aligned data
save_aligned_data(aligned_data, "aligned_output.hdf5")
```

## Implementation Priority

1. **Phase 1**: Basic alignment for good sensors (3, 4, 5, wb)
2. **Phase 2**: GPS interpolation and upsampling
3. **Phase 3**: Gap handling and quality metrics
4. **Phase 4**: Sensor_wnb recovery (if needed)

## Notes and Warnings

1. **Sensor_wnb**: Currently achieving only ~7.5Hz. Consider excluding from initial implementation.
2. **GPS Gaps**: Some experiments show GPS jitter > 100ms. Implement robust gap detection.
3. **Time Base**: All sensors use 'time_from_sync' column - maintain this convention.
4. **Precision**: Maintain microsecond precision in timestamps to preserve IMU timing quality.

---

This guide provides a foundation for implementing the data alignment module. The timestamp analysis has revealed that most sensors have excellent timing characteristics, making the alignment task straightforward for the primary sensors. Focus initial efforts on the well-behaved sensors (3, 4, 5, wb) and GPS interpolation.
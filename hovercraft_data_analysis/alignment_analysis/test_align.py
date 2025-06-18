"""
Unit tests for the data alignment module.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from align import DataAligner


class TestDataAligner:
    """Test cases for DataAligner class."""
    
    @pytest.fixture
    def temp_config(self):
        """Create a temporary config file for testing."""
        config = {
            'reference_sensor': 'sensor_3',
            'target_rate': 200,
            'tolerances': {200: 2.5, 100: 5.0, 1: 20.0},
            'sensor_rates': {
                'sensor_3': 200,
                'sensor_4': 200,
                'sensor_5': 200,
                'sensor_wb': 100
            },
            'max_cross_sensor_offset_ms': 1.0
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_path = Path(f.name)
        
        yield temp_path
        temp_path.unlink()
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample sensor data for testing."""
        # Create 5 seconds of data
        duration = 5.0
        
        # Reference sensor at 200Hz
        ref_times = np.arange(0, duration, 1/200)
        sensor_3 = pd.DataFrame({
            'time_from_sync': ref_times,
            'x': np.sin(2 * np.pi * ref_times),
            'y': np.cos(2 * np.pi * ref_times),
            'z': np.ones_like(ref_times) * -9.81
        })
        
        # Another 200Hz sensor with slight jitter
        jitter = np.random.normal(0, 0.0001, len(ref_times))  # 0.1ms jitter
        sensor_4 = pd.DataFrame({
            'time_from_sync': ref_times + jitter,
            'x': np.sin(2 * np.pi * ref_times),
            'y': np.cos(2 * np.pi * ref_times),
            'z': np.ones_like(ref_times) * -9.81
        })
        
        # 100Hz sensor (every other sample)
        wb_times = ref_times[::2]
        sensor_wb = pd.DataFrame({
            'time_from_sync': wb_times,
            'x': np.sin(2 * np.pi * wb_times),
            'y': np.cos(2 * np.pi * wb_times),
            'z': np.ones_like(wb_times) * -9.81
        })
        
        return {
            'sensor_3': sensor_3,
            'sensor_4': sensor_4,
            'sensor_wb': sensor_wb
        }
    
    def test_initialization(self, temp_config):
        """Test DataAligner initialization."""
        aligner = DataAligner(temp_config)
        
        assert aligner.reference_sensor == 'sensor_3'
        assert aligner.target_rate == 200
        assert aligner.tolerances[200] == 2.5
        assert aligner.max_cross_sensor_offset == 1.0
    
    def test_nearest_neighbor_matching(self, temp_config, sample_data):
        """Test nearest-neighbor timestamp matching."""
        aligner = DataAligner(temp_config)
        aligned = aligner.align_all_sensors(sample_data)
        
        # Check that all sensors are aligned
        assert 'sensor_3' in aligned
        assert 'sensor_4' in aligned
        assert 'sensor_wb' in aligned
        
        # Check that sensor_4 is aligned to sensor_3 timestamps
        assert 'aligned_time' in aligned['sensor_4'].columns
        assert 'time_diff_ms' in aligned['sensor_4'].columns
        
        # Verify time differences are within tolerance
        assert aligned['sensor_4']['time_diff_ms'].max() < 2.5
    
    def test_downsampling_100hz(self, temp_config, sample_data):
        """Test 2:1 downsampling for 100Hz sensor."""
        aligner = DataAligner(temp_config)
        aligned = aligner.align_all_sensors(sample_data)
        
        # sensor_wb should have half the samples of sensor_3
        expected_len = len(sample_data['sensor_3']) // 2
        actual_len = len(aligned['sensor_wb'])
        
        # Allow for some edge effects
        assert abs(actual_len - expected_len) <= 2
        
        # Check that aligned times match every 2nd reference timestamp
        ref_times = aligned['sensor_3']['time_from_sync'].values
        wb_aligned_times = aligned['sensor_wb']['aligned_time'].values
        
        # Every aligned time in sensor_wb should be in reference times
        for t in wb_aligned_times[:10]:  # Check first 10
            assert t in ref_times
    
    def test_synthetic_drift_detection(self, temp_config):
        """Test that synthetic drift is detected and causes assertion failure."""
        aligner = DataAligner(temp_config)
        
        # Create data with significant drift between sensors
        ref_times = np.arange(0, 1.0, 1/200)  # 1 second of data
        
        sensor_3 = pd.DataFrame({
            'time_from_sync': ref_times,
            'x': np.zeros_like(ref_times)
        })
        
        # Add 3ms drift to sensor_4
        sensor_4 = pd.DataFrame({
            'time_from_sync': ref_times + 0.003,  # 3ms offset
            'x': np.zeros_like(ref_times)
        })
        
        data = {'sensor_3': sensor_3, 'sensor_4': sensor_4}
        
        # This should raise an assertion error due to cross-sensor validation
        with pytest.raises(AssertionError, match="Cross-sensor offset.*exceeds.*limit"):
            aligner.align_all_sensors(data)
    
    def test_output_lengths(self, temp_config, sample_data):
        """Test that output lengths match expected values."""
        aligner = DataAligner(temp_config)
        aligned = aligner.align_all_sensors(sample_data)
        
        # Reference sensor should maintain its length
        assert len(aligned['sensor_3']) == len(sample_data['sensor_3'])
        
        # sensor_4 should have similar length (may lose a few edge samples)
        assert abs(len(aligned['sensor_4']) - len(sample_data['sensor_3'])) <= 5
        
        # sensor_wb should have approximately half the length
        expected_wb_len = len(sample_data['sensor_3']) // 2
        assert abs(len(aligned['sensor_wb']) - expected_wb_len) <= 5
    
    def test_runtime_performance(self, temp_config):
        """Test that alignment completes within 1 second for 5-minute dataset."""
        import time
        
        # Generate 5 minutes of data
        duration = 300.0  # 5 minutes
        ref_times = np.arange(0, duration, 1/200)
        
        # Create minimal dataset for performance testing
        data = {}
        for sensor in ['sensor_3', 'sensor_4', 'sensor_5', 'sensor_wb']:
            if sensor == 'sensor_wb':
                times = ref_times[::2]  # 100Hz
            else:
                times = ref_times + np.random.normal(0, 0.0001, len(ref_times))
            
            data[sensor] = pd.DataFrame({
                'time_from_sync': times,
                'x': np.zeros_like(times)
            })
        
        aligner = DataAligner(temp_config)
        
        start_time = time.time()
        aligned = aligner.align_all_sensors(data)
        elapsed = time.time() - start_time
        
        # Should complete in under 1 second
        assert elapsed < 1.0, f"Alignment took {elapsed:.3f}s, exceeding 1s limit"
    
    def test_missing_reference_sensor(self, temp_config):
        """Test error handling when reference sensor is missing."""
        aligner = DataAligner(temp_config)
        
        # Data without reference sensor
        data = {
            'sensor_4': pd.DataFrame({'time_from_sync': [0, 1, 2], 'x': [1, 2, 3]})
        }
        
        with pytest.raises(ValueError, match="Reference sensor.*not found"):
            aligner.align_all_sensors(data)
    
    def test_save_and_load_hdf5(self, temp_config, sample_data):
        """Test saving and loading aligned data from HDF5."""
        aligner = DataAligner(temp_config)
        aligned = aligner.align_all_sensors(sample_data)
        
        # Save to temporary file
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_aligned.h5"
            store = aligner.save_aligned_data(output_path)
            
            # Check that file was created
            assert output_path.exists()
            
            # Load and verify data
            for sensor in aligned.keys():
                loaded_df = store[sensor]
                pd.testing.assert_frame_equal(loaded_df, aligned[sensor])
            
            # Check metadata
            metadata = store['metadata']
            assert metadata['reference_sensor'].iloc[0] == 'sensor_3'
            assert metadata['target_rate'].iloc[0] == 200
            
            store.close()
    
    def test_edge_cases(self, temp_config):
        """Test edge cases like empty data, single sample, etc."""
        aligner = DataAligner(temp_config)
        
        # Test with single sample
        data = {
            'sensor_3': pd.DataFrame({'time_from_sync': [0], 'x': [1]}),
            'sensor_4': pd.DataFrame({'time_from_sync': [0.0001], 'x': [1]})
        }
        
        aligned = aligner.align_all_sensors(data)
        assert len(aligned['sensor_3']) == 1
        assert len(aligned['sensor_4']) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
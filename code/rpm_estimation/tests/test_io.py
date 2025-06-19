"""
Tests for I/O operations.
"""

import pytest
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from rpm_estimation.io import (
    find_experiment_data, load_sensor_data, save_rpm_results,
    list_available_experiments
)


class TestDataPaths:
    """Test data path finding operations."""
    
    def test_find_experiment_data_structure(self):
        """Test the expected structure of experiment data paths."""
        # This test documents the expected directory structure
        # Actual data may not exist in test environment
        
        expected_structure = {
            'base_path': 'hovercraft_data_analysis/alignment_analysis/aligned_data',
            'csv_export': '{experiment}_csv/',
            'sensor_files': [
                'Sensor_3.csv',
                'Sensor_4.csv', 
                'Sensor_5.csv',
                'Sensor_wb.csv',
                'gps.csv',
                'metadata.csv'
            ]
        }
        
        # Verify structure is as expected
        assert 'aligned_data' in expected_structure['base_path']
        assert '_csv' in expected_structure['csv_export']
    
    @pytest.mark.requires_data
    def test_find_experiment_data_real(self):
        """Test finding real experiment data (requires data files)."""
        # This test will be skipped if data files don't exist
        try:
            data_dir = find_experiment_data('007_Fast_stbd_turn_1', 'afternoon')
            assert data_dir.exists()
            assert data_dir.name.endswith('_csv')
        except FileNotFoundError:
            pytest.skip("Test data not available")
    
    def test_list_available_experiments_structure(self):
        """Test listing available experiments."""
        # This tests the function logic, not actual data
        experiments = list_available_experiments(Path('nonexistent_path'))
        
        assert isinstance(experiments, dict)
        assert 'morning' in experiments
        assert 'afternoon' in experiments
        assert isinstance(experiments['morning'], list)
        assert isinstance(experiments['afternoon'], list)


class TestDataLoading:
    """Test data loading operations."""
    
    def create_test_csv(self, path: Path):
        """Create a test CSV file with sensor data."""
        data = {
            't': np.arange(0, 10, 0.005),  # 200 Hz for 10 seconds
            'x': np.random.randn(2000) * 0.1,
            'y': np.random.randn(2000) * 0.1,
            'z': np.random.randn(2000) * 0.1 + 9.8,  # Gravity bias
            'time_from_sync': np.arange(0, 10, 0.005),
            'gyro_x': np.random.randn(2000) * 0.01,
            'gyro_y': np.random.randn(2000) * 0.01,
            'gyro_z': np.random.randn(2000) * 0.01
        }
        
        df = pd.DataFrame(data)
        df.to_csv(path, index=False)
        return df
    
    @pytest.mark.requires_data  
    def test_load_sensor_data_real(self):
        """Test loading real sensor data (requires data files)."""
        try:
            df = load_sensor_data('007_Fast_stbd_turn_1', 'afternoon', 'Sensor_3')
            
            # Check data structure
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            
            # Check required columns
            required_cols = ['t', 'x', 'y', 'z', 'gyro_x', 'gyro_y', 'gyro_z']
            for col in required_cols:
                assert col in df.columns
                
        except FileNotFoundError:
            pytest.skip("Test data not available")
    
    def test_load_sensor_data_mock(self):
        """Test loading sensor data with mock file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock directory structure
            exp_dir = Path(tmpdir) / "test_experiment_csv"
            exp_dir.mkdir()
            
            # Create mock sensor file
            sensor_file = exp_dir / "Sensor_3.csv"
            expected_df = self.create_test_csv(sensor_file)
            
            # Mock the find_experiment_data function
            import rpm_estimation.io
            original_find = rpm_estimation.io.find_experiment_data
            
            def mock_find(exp, session, base_path=None):
                return exp_dir
            
            rpm_estimation.io.find_experiment_data = mock_find
            
            try:
                # Load data
                df = load_sensor_data('test_experiment', 'afternoon', 'Sensor_3')
                
                # Verify
                assert len(df) == len(expected_df)
                assert all(df.columns == expected_df.columns)
                assert df['t'].iloc[0] == 0.0
                assert df['t'].iloc[-1] == pytest.approx(9.995, rel=0.01)
                
            finally:
                # Restore original function
                rpm_estimation.io.find_experiment_data = original_find


class TestResultSaving:
    """Test saving RPM results."""
    
    def test_save_rpm_results(self):
        """Test saving RPM results to CSV."""
        # Create test data
        rpm_data = pd.DataFrame({
            'time': np.arange(0, 10, 0.1),
            'rpm': np.random.uniform(1000, 2000, 100),
            'snr_db': np.random.uniform(5, 25, 100),
            'sensor_id': ['Sensor_3'] * 100,
            'method': ['welch'] * 100
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'test_results.csv'
            
            # Save results
            save_rpm_results(rpm_data, output_path)
            
            # Verify file exists
            assert output_path.exists()
            
            # Load and verify
            loaded_df = pd.read_csv(output_path)
            assert len(loaded_df) == len(rpm_data)
            assert all(loaded_df.columns == rpm_data.columns)
            assert loaded_df['rpm'].mean() == pytest.approx(rpm_data['rpm'].mean(), rel=0.01)
    
    def test_save_rpm_results_creates_directory(self):
        """Test that save creates parent directories if needed."""
        rpm_data = pd.DataFrame({
            'time': [0, 1, 2],
            'rpm': [1000, 1100, 1200],
            'snr_db': [15, 16, 17],
            'sensor_id': ['test'] * 3,
            'method': ['welch'] * 3
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Nested path that doesn't exist
            output_path = Path(tmpdir) / 'results' / 'experiment' / 'rpm.csv'
            
            # Should create directories
            save_rpm_results(rpm_data, output_path)
            
            assert output_path.exists()
            assert output_path.parent.exists()
            assert output_path.parent.parent.exists()
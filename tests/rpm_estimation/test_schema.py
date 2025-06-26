"""
Tests for Parquet schema validation.
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pathlib import Path
import sys

# Import schema module
from src.analysis.rpm.schema import (
    validate_parquet_schema, create_parquet_metadata,
    validate_data_consistency, PROCESSED_IMU_SCHEMA, REQUIRED_COLUMNS
)


class TestSchemaValidation:
    """Test Parquet schema validation."""
    
    def test_valid_schema(self, tmp_path):
        """Test validation of correct schema."""
        # Create valid DataFrame
        n_samples = 1000
        df = pd.DataFrame({
            'time_from_sync': np.arange(n_samples, dtype=float),
            't': np.arange(n_samples, dtype=float),
            'x': np.random.randn(n_samples),
            'y': np.random.randn(n_samples),
            'z': np.random.randn(n_samples),
            'x_body': np.random.randn(n_samples),
            'y_body': np.random.randn(n_samples),
            'z_body': np.random.randn(n_samples),
            'a_hp_x': np.random.randn(n_samples),
            'a_hp_y': np.random.randn(n_samples),
            'a_hp_z': np.random.randn(n_samples),
            'a_hp_mag': np.random.randn(n_samples),
            'quality_flag': np.zeros(n_samples, dtype=np.int8),
            'window_id': np.arange(n_samples, dtype=np.int32)
        })
        
        # Save to Parquet
        file_path = tmp_path / 'test_valid.parquet'
        table = pa.Table.from_pandas(df)
        metadata = {b'schema_version': b'1.0'}
        table = table.replace_schema_metadata(metadata)
        pq.write_table(table, file_path)
        
        # Validate
        is_valid, issues = validate_parquet_schema(file_path)
        
        assert is_valid == True
        assert len(issues) == 0
    
    def test_missing_columns(self, tmp_path):
        """Test detection of missing required columns."""
        # Create DataFrame missing required columns
        df = pd.DataFrame({
            'time_from_sync': np.arange(100),
            'x': np.random.randn(100)
        })
        
        file_path = tmp_path / 'test_missing.parquet'
        df.to_parquet(file_path)
        
        is_valid, issues = validate_parquet_schema(file_path)
        
        assert is_valid == False
        assert any('Missing required columns' in issue for issue in issues)
    
    def test_type_mismatch(self, tmp_path):
        """Test detection of type mismatches."""
        # Create DataFrame with wrong types
        df = pd.DataFrame({
            'time_from_sync': np.arange(100, dtype=float),
            'a_hp_x': np.random.randn(100),
            'a_hp_y': np.random.randn(100),
            'a_hp_z': np.random.randn(100),
            'a_hp_mag': np.random.randn(100),
            'quality_flag': np.zeros(100, dtype=float)  # Should be int8
        })
        
        file_path = tmp_path / 'test_types.parquet'
        df.to_parquet(file_path)
        
        is_valid, issues = validate_parquet_schema(file_path)
        
        assert is_valid == False
        assert any('Type mismatch' in issue for issue in issues)
    
    def test_null_detection(self, tmp_path):
        """Test detection of null values in critical columns."""
        # Create DataFrame with nulls
        df = pd.DataFrame({
            'time_from_sync': pd.Series([1.0, 2.0, None, 4.0]),
            'a_hp_x': np.random.randn(4),
            'a_hp_y': np.random.randn(4),
            'a_hp_z': np.random.randn(4),
            'a_hp_mag': np.random.randn(4),
            'quality_flag': np.zeros(4, dtype=np.int8)
        })
        
        file_path = tmp_path / 'test_nulls.parquet'
        df.to_parquet(file_path)
        
        is_valid, issues = validate_parquet_schema(file_path, check_nulls=True)
        
        assert is_valid == False
        assert any('nulls' in issue for issue in issues)


class TestMetadataCreation:
    """Test metadata creation for Parquet files."""
    
    def test_metadata_generation(self):
        """Test creation of metadata dictionary."""
        config = {
            'fs': 200,
            'wp1': {
                'output': {'schema_version': '1.0'},
                'filters': {'highpass_cutoff': 5.0},
                'quality': {'window_sec': 30.0}
            }
        }
        
        metadata = create_parquet_metadata('test_exp', 'morning', 'Sensor_3', config)
        
        # Check all fields are bytes
        assert all(isinstance(k, bytes) for k in metadata.keys())
        assert all(isinstance(v, bytes) for v in metadata.values())
        
        # Check content
        assert metadata[b'experiment'] == b'test_exp'
        assert metadata[b'session'] == b'morning'
        assert metadata[b'sensor_id'] == b'Sensor_3'
        assert metadata[b'schema_version'] == b'1.0'
        assert b'processing_timestamp' in metadata


class TestDataConsistency:
    """Test internal data consistency validation."""
    
    def test_time_monotonicity(self):
        """Test detection of non-monotonic time."""
        df = pd.DataFrame({
            'time_from_sync': [1.0, 2.0, 1.5, 3.0],  # Non-monotonic
            'a_hp_x': [1, 2, 3, 4]
        })
        
        is_valid, issues = validate_data_consistency(df)
        
        assert is_valid == False
        assert any('monotonic' in issue for issue in issues)
    
    def test_magnitude_calculation(self):
        """Test validation of magnitude calculation."""
        # Create data with incorrect magnitude
        df = pd.DataFrame({
            'a_hp_x': [3.0, 0.0],
            'a_hp_y': [4.0, 0.0],
            'a_hp_z': [0.0, 5.0],
            'a_hp_mag': [5.0, 6.0]  # Second value wrong (should be 5)
        })
        
        is_valid, issues = validate_data_consistency(df)
        
        assert is_valid == False
        assert any('Magnitude calculation error' in issue for issue in issues)
    
    def test_invalid_quality_flags(self):
        """Test detection of invalid quality flags."""
        df = pd.DataFrame({
            'quality_flag': [0, 1, 2, 3, 4]  # 3 and 4 are invalid
        })
        
        is_valid, issues = validate_data_consistency(df)
        
        assert is_valid == False
        assert any('Invalid quality flags' in issue for issue in issues)
    
    def test_inf_values(self):
        """Test detection of inf values."""
        df = pd.DataFrame({
            'a_hp_x': [1.0, np.inf, 3.0],
            'a_hp_y': [1.0, 2.0, 3.0]
        })
        
        is_valid, issues = validate_data_consistency(df)
        
        assert is_valid == False
        assert any('Inf values' in issue for issue in issues)


def test_schema_report():
    """Test schema validation report generation."""
    from schema import generate_schema_report
    
    validation_results = [
        {'file': 'file1.parquet', 'is_valid': True, 'issues': []},
        {'file': 'file2.parquet', 'is_valid': False, 'issues': ['Missing columns']},
        {'file': 'file3.parquet', 'is_valid': False, 'issues': ['Missing columns', 'Type mismatch']},
    ]
    
    report = generate_schema_report(validation_results)
    
    assert report['total_files'] == 3
    assert report['valid_files'] == 1
    assert report['invalid_files'] == 2
    assert report['validation_rate'] == pytest.approx(33.3, 0.1)
    assert 'Missing columns' in report['common_issues']
    assert report['common_issues']['Missing columns'] == 2


def test_parquet_info(tmp_path):
    """Test extraction of Parquet file information."""
    from schema import get_parquet_info
    
    # Create test Parquet file
    df = pd.DataFrame({
        'col1': np.arange(1000),
        'col2': np.random.randn(1000)
    })
    
    file_path = tmp_path / 'test_info.parquet'
    
    # Add metadata
    table = pa.Table.from_pandas(df)
    metadata = {b'test_key': b'test_value'}
    table = table.replace_schema_metadata(metadata)
    pq.write_table(table, file_path, compression='snappy')
    
    # Get info
    info = get_parquet_info(file_path)
    
    assert info['num_rows'] == 1000
    assert info['num_columns'] == 2
    assert 'col1' in info['columns']
    assert info['compression'] == 'SNAPPY'
    assert 'metadata' in info
    assert info['metadata']['test_key'] == 'test_value'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
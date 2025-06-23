"""
Schema validation for Parquet files.

Ensures consistent data structure across all processed files.
"""

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from .logging_config import get_logger, ProcessingError

logger = get_logger("schema")


# Define expected schema for processed IMU data
PROCESSED_IMU_SCHEMA = pa.schema([
    # Time columns
    ('time_from_sync', pa.float64()),
    ('t', pa.float64()),
    
    # Raw sensor frame accelerations (if included)
    ('x', pa.float64()),
    ('y', pa.float64()),
    ('z', pa.float64()),
    
    # Body frame accelerations
    ('x_body', pa.float64()),
    ('y_body', pa.float64()),
    ('z_body', pa.float64()),
    
    # High-pass filtered accelerations
    ('a_hp_x', pa.float64()),
    ('a_hp_y', pa.float64()),
    ('a_hp_z', pa.float64()),
    ('a_hp_mag', pa.float64()),
    
    # Quality indicators
    ('quality_flag', pa.int8()),  # 0=good, 1=warning, 2=bad
    ('window_id', pa.int32()),     # Quality window identifier
    
    # Optional gyro data (if available)
    ('gyro_x', pa.float64()),
    ('gyro_y', pa.float64()),
    ('gyro_z', pa.float64()),
])

# Minimal required columns
REQUIRED_COLUMNS = [
    'time_from_sync',
    'a_hp_x', 
    'a_hp_y', 
    'a_hp_z',
    'a_hp_mag',
    'quality_flag'
]


def validate_parquet_schema(file_path: Path, 
                          schema: Optional[pa.Schema] = None,
                          check_nulls: bool = True) -> Tuple[bool, List[str]]:
    """
    Validate Parquet file meets expected schema.
    
    Args:
        file_path: Path to Parquet file
        schema: Expected schema (defaults to PROCESSED_IMU_SCHEMA)
        check_nulls: Whether to check for null values in critical columns
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    if schema is None:
        schema = PROCESSED_IMU_SCHEMA
    
    issues = []
    
    try:
        # Read schema without loading data
        parquet_file = pq.ParquetFile(file_path)
        actual_schema = parquet_file.schema_arrow
        
        # Check for required columns
        actual_columns = set(actual_schema.names)
        required_columns = set(REQUIRED_COLUMNS)
        missing_columns = required_columns - actual_columns
        
        if missing_columns:
            issues.append(f"Missing required columns: {sorted(missing_columns)}")
        
        # Check data types for existing columns
        for field in schema:
            if field.name in actual_columns:
                actual_field = actual_schema.field(field.name)
                if actual_field.type != field.type:
                    issues.append(
                        f"Type mismatch for '{field.name}': "
                        f"expected {field.type}, got {actual_field.type}"
                    )
        
        # Check for null values in critical columns if requested
        if check_nulls and len(issues) == 0:
            metadata = parquet_file.metadata
            
            # Check first row group
            if metadata.num_row_groups > 0:
                row_group = metadata.row_group(0)
                
                critical_columns = ['time_from_sync', 'a_hp_mag']
                for col_name in critical_columns:
                    if col_name in actual_columns:
                        col_idx = actual_columns.index(col_name)
                        column_chunk = row_group.column(col_idx)
                        
                        if column_chunk.statistics and column_chunk.statistics.null_count > 0:
                            issues.append(f"Unexpected nulls in column '{col_name}'")
        
        # Check metadata
        if parquet_file.metadata.metadata:
            file_metadata = parquet_file.schema_arrow.metadata
            if not file_metadata or b'schema_version' not in file_metadata:
                issues.append("Missing schema version in metadata")
        else:
            issues.append("No metadata found in Parquet file")
            
    except Exception as e:
        issues.append(f"Failed to read Parquet file: {str(e)}")
        logger.error(
            f"Schema validation failed for {file_path}",
            error_type=ProcessingError.VALIDATION,
            exception=str(e)
        )
    
    is_valid = len(issues) == 0
    
    if not is_valid:
        logger.warning(
            f"Schema validation issues for {file_path}: {'; '.join(issues)}",
            error_type=ProcessingError.VALIDATION
        )
    else:
        logger.debug(f"Schema validation passed for {file_path}")
    
    return is_valid, issues


def create_parquet_metadata(experiment: str, 
                          session: str,
                          sensor_id: str,
                          config: Dict[str, Any]) -> Dict[bytes, bytes]:
    """
    Create metadata dictionary for Parquet files.
    
    Args:
        experiment: Experiment name
        session: Session (morning/afternoon)
        sensor_id: Sensor identifier
        config: Processing configuration
        
    Returns:
        Metadata dictionary with byte keys/values
    """
    import json
    from datetime import datetime
    
    metadata = {
        'experiment': experiment,
        'session': session,
        'sensor_id': sensor_id,
        'schema_version': config.get('wp1', {}).get('output', {}).get('schema_version', '1.0'),
        'processing_timestamp': datetime.utcnow().isoformat() + 'Z',
        'sampling_rate_hz': str(config.get('fs', 200)),
        'highpass_cutoff_hz': str(config.get('wp1', {}).get('filters', {}).get('highpass_cutoff', 5.0)),
        'quality_window_sec': str(config.get('wp1', {}).get('quality', {}).get('window_sec', 30.0))
    }
    
    # Convert to bytes for PyArrow
    byte_metadata = {
        k.encode('utf-8'): v.encode('utf-8') if isinstance(v, str) else json.dumps(v).encode('utf-8')
        for k, v in metadata.items()
    }
    
    return byte_metadata


def validate_data_consistency(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate internal consistency of processed data.
    
    Args:
        df: DataFrame with processed data
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    import pandas as pd
    import numpy as np
    
    issues = []
    
    # Check time monotonicity
    if 'time_from_sync' in df.columns:
        time_diff = df['time_from_sync'].diff()
        if not (time_diff[1:] > 0).all():
            issues.append("Time is not monotonically increasing")
    
    # Check magnitude calculation
    if all(col in df.columns for col in ['a_hp_x', 'a_hp_y', 'a_hp_z', 'a_hp_mag']):
        expected_mag = np.sqrt(df['a_hp_x']**2 + df['a_hp_y']**2 + df['a_hp_z']**2)
        mag_error = np.abs(df['a_hp_mag'] - expected_mag)
        
        if mag_error.max() > 1e-6:
            issues.append(f"Magnitude calculation error: max deviation = {mag_error.max():.2e}")
    
    # Check quality flags
    if 'quality_flag' in df.columns:
        valid_flags = [0, 1, 2]
        invalid_flags = ~df['quality_flag'].isin(valid_flags)
        if invalid_flags.any():
            issues.append(f"Invalid quality flags found: {df.loc[invalid_flags, 'quality_flag'].unique()}")
    
    # Check for inf or nan
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            issues.append(f"NaN values found in column '{col}'")
        if np.isinf(df[col]).any():
            issues.append(f"Inf values found in column '{col}'")
    
    is_valid = len(issues) == 0
    return is_valid, issues


def generate_schema_report(validation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate summary report of schema validation across multiple files.
    
    Args:
        validation_results: List of validation result dictionaries
        
    Returns:
        Summary report dictionary
    """
    total_files = len(validation_results)
    valid_files = sum(1 for r in validation_results if r['is_valid'])
    
    # Collect all unique issues
    all_issues = {}
    for result in validation_results:
        for issue in result.get('issues', []):
            if issue not in all_issues:
                all_issues[issue] = []
            all_issues[issue].append(result['file'])
    
    report = {
        'total_files': total_files,
        'valid_files': valid_files,
        'invalid_files': total_files - valid_files,
        'validation_rate': round(100.0 * valid_files / total_files, 1) if total_files > 0 else 0.0,
        'common_issues': {
            issue: len(files) for issue, files in all_issues.items()
        },
        'files_by_issue': all_issues
    }
    
    return report


def get_parquet_info(file_path: Path) -> Dict[str, Any]:
    """
    Extract information from a Parquet file.
    
    Args:
        file_path: Path to Parquet file
        
    Returns:
        Dictionary with file information
    """
    try:
        parquet_file = pq.ParquetFile(file_path)
        metadata = parquet_file.metadata
        schema = parquet_file.schema_arrow
        
        info = {
            'num_rows': metadata.num_rows,
            'num_columns': len(schema.names),
            'columns': schema.names,
            'file_size_mb': round(file_path.stat().st_size / (1024 * 1024), 2),
            'compression': str(metadata.row_group(0).column(0).compression if metadata.num_row_groups > 0 else 'unknown'),
            'created_by': str(metadata.created_by) if metadata.created_by else 'unknown'
        }
        
        # Extract custom metadata
        if schema.metadata:
            custom_metadata = {
                k.decode('utf-8'): v.decode('utf-8') 
                for k, v in schema.metadata.items()
            }
            info['metadata'] = custom_metadata
        
        return info
        
    except Exception as e:
        logger.error(f"Failed to read Parquet info from {file_path}: {str(e)}")
        return {'error': str(e)}
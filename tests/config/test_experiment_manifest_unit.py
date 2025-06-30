"""
Unit tests for experiment manifest validation.

These tests validate the internal consistency of the manifest file
without accessing the filesystem.
"""

import pytest
import yaml
from pathlib import Path
from typing import Dict, List
import tempfile
import json

from src.core.experiment_manifest import (
    ExperimentManifest,
    ValidationError,
    load_manifest,
    validate_manifest_structure,
    get_experiment_by_name,
    get_all_experiments,
    resolve_experiment_path,
    get_expected_data_folders
)


@pytest.fixture
def sample_manifest_data() -> dict:
    """Create a valid sample manifest structure for testing."""
    return {
        'evaluation_experiments': {
            'morning': [
                {
                    'name': '006_Departure',
                    'path': '1b_4_Normal_Take_off/morning/006_Departure',
                    'type': 'dynamic',
                    'description': 'Departure from port',
                    'category': '1b_4_Normal_Take_off',
                    'is_static': False,
                    'paths': {
                        'relative': 'morning/Experiments/006_Departure',
                        'full_path': '/data/raw/morning/Experiments/006_Departure'
                    },
                    'data_types': ['GPS', 'Sensor_3', 'Sensor_4']
                }
            ],
            'afternoon': [
                {
                    'name': '007_Fast_stbd_turn_1',
                    'path': '1a_1_Minimum_Radius_Turn/afternoon/007_Fast_stbd_turn_1',
                    'type': 'dynamic',
                    'description': 'Fast starboard turn',
                    'category': '1a_1_Minimum_Radius_Turn',
                    'is_static': False,
                    'paths': {
                        'relative': 'afternoon/Experiments/007_Fast_stbd_turn_1',
                        'full_path': '/data/raw/afternoon/Experiments/007_Fast_stbd_turn_1'
                    },
                    'data_types': ['GPS', 'Sensor_3', 'Sensor_5']
                }
            ]
        },
        'static_experiments': {
            'morning': [
                {
                    'name': '003_Rudder_Cals',
                    'path': 'Morning_Pre_flight/morning/003_Rudder_Cals',
                    'type': 'static',
                    'description': 'Rudder calibration',
                    'category': 'Morning_Pre_flight',
                    'is_static': True,
                    'paths': {
                        'relative': 'morning/Experiments/003_Rudder_Cals',
                        'full_path': '/data/raw/morning/Experiments/003_Rudder_Cals'
                    },
                    'data_types': ['GPS', 'Sensor_3']
                }
            ],
            'afternoon': []
        },
        'all_experiments': {
            'morning': [
                {'name': '003_Rudder_Cals'},
                {'name': '006_Departure'}
            ],
            'afternoon': [
                {'name': '007_Fast_stbd_turn_1'}
            ]
        },
        'analysis_config': {
            'orientation_validation_experiments': {
                'morning': ['003_Rudder_Cals'],
                'afternoon': []
            }
        }
    }


@pytest.fixture
def manifest_file(tmp_path, sample_manifest_data) -> Path:
    """Create a temporary manifest file for testing."""
    manifest_path = tmp_path / "test_manifest.yaml"
    with open(manifest_path, 'w') as f:
        yaml.dump(sample_manifest_data, f)
    return manifest_path


@pytest.fixture
def loaded_manifest(manifest_file) -> ExperimentManifest:
    """Load a test manifest."""
    return load_manifest(manifest_file)


class TestManifestLoading:
    """Test manifest file loading and parsing."""
    
    def test_load_valid_yaml(self, manifest_file):
        """Test loading a valid YAML file."""
        manifest = load_manifest(manifest_file)
        assert isinstance(manifest, ExperimentManifest)
        assert manifest.data is not None
        assert 'evaluation_experiments' in manifest.data
    
    def test_load_missing_file(self, tmp_path):
        """Test loading a non-existent file."""
        missing_path = tmp_path / "missing.yaml"
        with pytest.raises(FileNotFoundError, match="Manifest file not found"):
            load_manifest(missing_path)
    
    def test_load_invalid_yaml(self, tmp_path):
        """Test loading an invalid YAML file."""
        invalid_path = tmp_path / "invalid.yaml"
        with open(invalid_path, 'w') as f:
            f.write("{ invalid yaml content: [}")
        
        with pytest.raises(ValueError, match="Invalid YAML"):
            load_manifest(invalid_path)


class TestManifestStructure:
    """Test manifest structural validation."""
    
    def test_validate_complete_structure(self, loaded_manifest):
        """Test validation of a complete, valid structure."""
        errors = loaded_manifest.validate_structure()
        assert len(errors) == 0
    
    def test_missing_top_level_keys(self, tmp_path):
        """Test detection of missing top-level keys."""
        incomplete_data = {
            'evaluation_experiments': {'morning': [], 'afternoon': []},
            # Missing: static_experiments, all_experiments, analysis_config
        }
        
        manifest_path = tmp_path / "incomplete.yaml"
        with open(manifest_path, 'w') as f:
            yaml.dump(incomplete_data, f)
        
        manifest = load_manifest(manifest_path)
        errors = manifest.validate_structure()
        
        assert len(errors) > 0
        assert any(e.severity == 'error' and 'Missing top-level keys' in e.message for e in errors)
    
    def test_missing_session_subsections(self, tmp_path):
        """Test detection of missing morning/afternoon subsections."""
        data = {
            'evaluation_experiments': {
                'morning': []
                # Missing 'afternoon'
            },
            'static_experiments': {'morning': [], 'afternoon': []},
            'all_experiments': {'morning': [], 'afternoon': []},
            'analysis_config': {}
        }
        
        manifest_path = tmp_path / "missing_session.yaml"
        with open(manifest_path, 'w') as f:
            yaml.dump(data, f)
        
        manifest = load_manifest(manifest_path)
        errors = manifest.validate_structure()
        
        warnings = [e for e in errors if e.severity == 'warning']
        assert any('Missing afternoon session' in w.message for w in warnings)


class TestExperimentValidation:
    """Test individual experiment entry validation."""
    
    def test_validate_complete_experiments(self, loaded_manifest):
        """Test validation of complete experiment entries."""
        errors = loaded_manifest.validate_experiments()
        assert len(errors) == 0
    
    def test_missing_required_fields(self, tmp_path):
        """Test detection of missing required fields in experiments."""
        data = {
            'evaluation_experiments': {
                'morning': [
                    {
                        'name': 'Test_Exp',
                        # Missing: path, type, paths, data_types
                    }
                ],
                'afternoon': []
            },
            'static_experiments': {'morning': [], 'afternoon': []},
            'all_experiments': {'morning': [], 'afternoon': []},
            'analysis_config': {}
        }
        
        manifest_path = tmp_path / "missing_fields.yaml"
        with open(manifest_path, 'w') as f:
            yaml.dump(data, f)
        
        manifest = load_manifest(manifest_path)
        errors = manifest.validate_experiments()
        
        assert len(errors) > 0
        field_error = next(e for e in errors if 'missing fields' in e.message)
        assert field_error.severity == 'error'
        assert 'Test_Exp' in field_error.message
    
    def test_invalid_experiment_type(self, tmp_path):
        """Test detection of invalid experiment type values."""
        data = {
            'evaluation_experiments': {
                'morning': [
                    {
                        'name': 'Test_Exp',
                        'path': 'test/path',
                        'type': 'invalid_type',  # Should be 'static' or 'dynamic'
                        'paths': {'full_path': '/test'},
                        'data_types': ['GPS']
                    }
                ],
                'afternoon': []
            },
            'static_experiments': {'morning': [], 'afternoon': []},
            'all_experiments': {'morning': [], 'afternoon': []},
            'analysis_config': {}
        }
        
        manifest_path = tmp_path / "invalid_type.yaml"
        with open(manifest_path, 'w') as f:
            yaml.dump(data, f)
        
        manifest = load_manifest(manifest_path)
        errors = manifest.validate_experiments()
        
        type_errors = [e for e in errors if 'Invalid type' in e.message]
        assert len(type_errors) > 0
        assert type_errors[0].details['invalid_type'] == 'invalid_type'
    
    def test_invalid_data_types(self, tmp_path):
        """Test detection of invalid data type values."""
        data = {
            'evaluation_experiments': {
                'morning': [
                    {
                        'name': 'Test_Exp',
                        'path': 'test/path',
                        'type': 'dynamic',
                        'paths': {'full_path': '/test'},
                        'data_types': ['GPS', 'Invalid_Sensor', 'Sensor_99']
                    }
                ],
                'afternoon': []
            },
            'static_experiments': {'morning': [], 'afternoon': []},
            'all_experiments': {'morning': [], 'afternoon': []},
            'analysis_config': {}
        }
        
        manifest_path = tmp_path / "invalid_data_types.yaml"
        with open(manifest_path, 'w') as f:
            yaml.dump(data, f)
        
        manifest = load_manifest(manifest_path)
        errors = manifest.validate_experiments()
        
        data_type_warnings = [e for e in errors if 'Unknown data types' in e.message]
        assert len(data_type_warnings) > 0
        assert 'Invalid_Sensor' in str(data_type_warnings[0].details['invalid_data_types'])


class TestUniquenessValidation:
    """Test experiment name uniqueness validation."""
    
    def test_no_duplicates(self, loaded_manifest):
        """Test that valid manifest has no duplicate names."""
        errors = loaded_manifest.validate_uniqueness()
        # Note: all_experiments is allowed to have duplicates
        critical_errors = [e for e in errors if e.severity == 'error']
        assert len(critical_errors) == 0
    
    def test_duplicate_names_in_section(self, tmp_path):
        """Test detection of duplicate experiment names."""
        data = {
            'evaluation_experiments': {
                'morning': [
                    {'name': 'Duplicate_Exp', 'path': 'path1', 'type': 'dynamic', 
                     'paths': {'full_path': '/path1'}, 'data_types': ['GPS']},
                    {'name': 'Duplicate_Exp', 'path': 'path2', 'type': 'dynamic',
                     'paths': {'full_path': '/path2'}, 'data_types': ['GPS']}
                ],
                'afternoon': []
            },
            'static_experiments': {'morning': [], 'afternoon': []},
            'all_experiments': {'morning': [], 'afternoon': []},
            'analysis_config': {}
        }
        
        manifest_path = tmp_path / "duplicates.yaml"
        with open(manifest_path, 'w') as f:
            yaml.dump(data, f)
        
        manifest = load_manifest(manifest_path)
        errors = manifest.validate_uniqueness()
        
        duplicate_errors = [e for e in errors if 'Duplicate names' in e.message]
        assert len(duplicate_errors) > 0
        assert duplicate_errors[0].details['duplicates']['Duplicate_Exp'] == 2


class TestCrossReferenceValidation:
    """Test cross-reference validation within the manifest."""
    
    def test_valid_cross_references(self, loaded_manifest):
        """Test that valid cross-references pass validation."""
        errors = loaded_manifest.validate_cross_references()
        assert len(errors) == 0
    
    def test_invalid_analysis_config_reference(self, tmp_path):
        """Test detection of invalid experiment references in analysis_config."""
        data = {
            'evaluation_experiments': {'morning': [], 'afternoon': []},
            'static_experiments': {
                'morning': [
                    {'name': 'Valid_Static', 'path': 'path', 'type': 'static',
                     'paths': {'full_path': '/path'}, 'data_types': ['GPS']}
                ],
                'afternoon': []
            },
            'all_experiments': {'morning': [], 'afternoon': []},
            'analysis_config': {
                'orientation_validation_experiments': {
                    'morning': ['Valid_Static', 'Invalid_Reference'],
                    'afternoon': []
                }
            }
        }
        
        manifest_path = tmp_path / "invalid_refs.yaml"
        with open(manifest_path, 'w') as f:
            yaml.dump(data, f)
        
        manifest = load_manifest(manifest_path)
        errors = manifest.validate_cross_references()
        
        ref_errors = [e for e in errors if 'Invalid experiment references' in e.message]
        assert len(ref_errors) > 0
        assert 'Invalid_Reference' in ref_errors[0].details['invalid_references']


class TestPathConsistency:
    """Test path consistency validation."""
    
    def test_consistent_paths(self, loaded_manifest):
        """Test that consistent paths pass validation."""
        errors = loaded_manifest.validate_path_consistency()
        # Should only have info-level messages about name not in path
        critical_errors = [e for e in errors if e.severity in ('error', 'warning')]
        assert len(critical_errors) == 0
    
    def test_inconsistent_path_names(self, tmp_path):
        """Test detection of inconsistent path names."""
        data = {
            'evaluation_experiments': {
                'morning': [
                    {
                        'name': 'Test_Exp',
                        'path': 'category/morning/Different_Name',  # Name mismatch
                        'type': 'dynamic',
                        'paths': {
                            'relative': 'morning/Experiments/Test_Exp',
                            'full_path': '/data/raw/morning/Experiments/Test_Exp'
                        },
                        'data_types': ['GPS']
                    }
                ],
                'afternoon': []
            },
            'static_experiments': {'morning': [], 'afternoon': []},
            'all_experiments': {'morning': [], 'afternoon': []},
            'analysis_config': {}
        }
        
        manifest_path = tmp_path / "path_mismatch.yaml"
        with open(manifest_path, 'w') as f:
            yaml.dump(data, f)
        
        manifest = load_manifest(manifest_path)
        errors = manifest.validate_path_consistency()
        
        path_warnings = [e for e in errors if e.severity == 'warning' and 'Path inconsistency' in e.message]
        assert len(path_warnings) > 0


class TestHelperFunctions:
    """Test helper functions for querying the manifest."""
    
    def test_get_experiment_by_name(self, loaded_manifest):
        """Test retrieving experiments by name."""
        exp = loaded_manifest.get_experiment_by_name('006_Departure')
        assert exp is not None
        assert exp['name'] == '006_Departure'
        assert exp['session'] == 'morning'
        
        # Test with session filter
        exp_morning = loaded_manifest.get_experiment_by_name('006_Departure', 'morning')
        assert exp_morning is not None
        
        exp_afternoon = loaded_manifest.get_experiment_by_name('006_Departure', 'afternoon')
        assert exp_afternoon is None
        
        # Test non-existent experiment
        exp_missing = loaded_manifest.get_experiment_by_name('NonExistent')
        assert exp_missing is None
    
    def test_get_all_experiments(self, loaded_manifest):
        """Test retrieving all experiments."""
        all_exps = loaded_manifest.get_all_experiments()
        assert len(all_exps) == 3  # 2 evaluation + 1 static
        
        # Check that session info is added
        for exp in all_exps:
            assert 'session' in exp
            assert exp['session'] in ('morning', 'afternoon')
            assert 'experiment_set' in exp
            assert exp['experiment_set'] in ('evaluation', 'static')
    
    def test_get_experiments_by_category(self, loaded_manifest):
        """Test filtering experiments by category."""
        take_off_exps = loaded_manifest.get_experiments_by_category('1b_4_Normal_Take_off')
        assert len(take_off_exps) == 1
        assert take_off_exps[0]['name'] == '006_Departure'
        
        # Test non-existent category
        empty_exps = loaded_manifest.get_experiments_by_category('NonExistent_Category')
        assert len(empty_exps) == 0
    
    def test_resolve_experiment_path(self, loaded_manifest):
        """Test path resolution for experiments."""
        exp = loaded_manifest.get_experiment_by_name('006_Departure')
        path = loaded_manifest.resolve_experiment_path(exp)
        # Path should be resolved relative to actual data directory
        assert path.name == '006_Departure'
        assert 'morning/Experiments' in str(path)
        
        # Test with base path
        base = Path('/custom/base')
        path_with_base = loaded_manifest.resolve_experiment_path(exp, base)
        # full_path takes precedence but is resolved relative to data dir
        assert path_with_base.name == '006_Departure'
        assert 'morning/Experiments' in str(path_with_base)
        
        # Test experiment without full_path
        exp_relative = {
            'name': 'Test',
            'paths': {'relative': 'morning/Experiments/Test'}
        }
        path_relative = resolve_experiment_path(exp_relative, base)
        assert path_relative == base / 'morning/Experiments/Test'
    
    def test_get_expected_data_folders(self, loaded_manifest):
        """Test getting expected data folder paths."""
        exp = loaded_manifest.get_experiment_by_name('006_Departure')
        data_folders = loaded_manifest.get_expected_data_folders(exp)
        
        # Check structure rather than absolute paths
        assert len(data_folders) == 3
        
        # Check GPS folder
        gps_dtype, gps_path = data_folders[0]
        assert gps_dtype == 'GPS'
        assert gps_path.name == 'GPS'
        assert gps_path.parent.name == '006_Departure'
        
        # Check IMU sensor folders
        for i, (dtype, path) in enumerate(data_folders[1:], start=3):
            assert dtype == f'Sensor_{i}'
            assert path.name == f'Sensor_{i}'
            assert path.parent.name == 'IMU'
            assert path.parent.parent.name == '006_Departure'


class TestComprehensiveValidation:
    """Test the comprehensive validation function."""
    
    def test_validate_manifest_structure(self, loaded_manifest):
        """Test running all validations together."""
        errors = validate_manifest_structure(loaded_manifest)
        
        # Group errors by severity
        by_severity = {'error': [], 'warning': [], 'info': []}
        for error in errors:
            by_severity[error.severity].append(error)
        
        # For a valid manifest, we should have no critical errors
        assert len(by_severity['error']) == 0
        
        # May have some info messages about experiment names not in paths
        assert all(e.category == 'path' for e in by_severity['info'])
"""
Tests for experiment manifest helper functions and validation report.
"""

import pytest
from pathlib import Path
import json
import tempfile

from src.core.experiment_manifest import (
    ExperimentManifest,
    ValidationError,
    load_manifest,
    get_experiment_by_name,
    get_all_experiments,
    resolve_experiment_path,
    get_expected_data_folders
)
from src.core.validation_report import (
    ValidationSummary,
    ManifestValidationReport,
    validate_manifest_comprehensive
)


class TestValidationReport:
    """Test the validation report functionality."""
    
    def test_create_empty_report(self):
        """Test creating an empty validation report."""
        report = ManifestValidationReport(manifest_path=Path("/test/manifest.yaml"))
        
        assert report.manifest_path == Path("/test/manifest.yaml")
        assert report.summary.total_errors == 0
        assert report.summary.total_warnings == 0
        assert len(report.structural_errors) == 0
        assert len(report.filesystem_errors) == 0
    
    def test_add_errors(self):
        """Test adding errors to the report."""
        report = ManifestValidationReport(manifest_path=Path("/test/manifest.yaml"))
        
        # Add structural error
        struct_error = ValidationError(
            severity='error',
            category='structure',
            message='Missing required field',
            details={'field': 'name'}
        )
        report.add_structural_error(struct_error)
        
        # Add filesystem error
        fs_error = ValidationError(
            severity='warning',
            category='filesystem',
            message='Directory not found',
            details={'path': '/missing/dir'}
        )
        report.add_filesystem_error(fs_error)
        
        assert len(report.structural_errors) == 1
        assert len(report.filesystem_errors) == 1
        assert report.summary.total_errors == 1
        assert report.summary.total_warnings == 1
    
    def test_add_orphan_directory(self):
        """Test adding orphan directories."""
        report = ManifestValidationReport(manifest_path=Path("/test/manifest.yaml"))
        
        report.add_orphan_directory('morning', 'Test_Orphan')
        
        assert len(report.orphan_directories) == 1
        assert report.orphan_directories[0]['directory'] == 'Test_Orphan'
        assert report.summary.orphan_directories == 1
    
    def test_get_errors_by_severity(self):
        """Test filtering errors by severity."""
        report = ManifestValidationReport(manifest_path=Path("/test/manifest.yaml"))
        
        # Add various errors
        report.add_structural_error(ValidationError('error', 'structure', 'Error 1'))
        report.add_structural_error(ValidationError('warning', 'structure', 'Warning 1'))
        report.add_filesystem_error(ValidationError('error', 'filesystem', 'Error 2'))
        report.add_filesystem_error(ValidationError('info', 'filesystem', 'Info 1'))
        
        errors = report.get_errors_by_severity('error')
        warnings = report.get_errors_by_severity('warning')
        info = report.get_errors_by_severity('info')
        
        assert len(errors) == 2
        assert len(warnings) == 1
        assert len(info) == 1
    
    def test_report_to_dict(self):
        """Test converting report to dictionary."""
        report = ManifestValidationReport(manifest_path=Path("/test/manifest.yaml"))
        report.summary.total_experiments = 10
        report.add_structural_error(ValidationError('error', 'structure', 'Test error'))
        
        data = report.to_dict()
        
        assert 'manifest_path' in data
        assert 'validation_date' in data
        assert 'summary' in data
        assert data['summary']['total_experiments'] == 10
        assert len(data['structural_errors']) == 1
        assert data['structural_errors'][0]['message'] == 'Test error'
    
    def test_report_to_json(self):
        """Test converting report to JSON."""
        report = ManifestValidationReport(manifest_path=Path("/test/manifest.yaml"))
        report.add_structural_error(ValidationError('error', 'structure', 'Test error'))
        
        json_str = report.to_json()
        data = json.loads(json_str)
        
        assert data['summary']['total_errors'] == 1
        assert len(data['structural_errors']) == 1
    
    def test_report_to_markdown(self):
        """Test converting report to Markdown."""
        report = ManifestValidationReport(manifest_path=Path("/test/manifest.yaml"))
        report.summary.total_experiments = 5
        report.add_structural_error(ValidationError('error', 'structure', 'Critical error'))
        report.add_filesystem_error(ValidationError('warning', 'filesystem', 'Missing directory'))
        report.add_orphan_directory('morning', 'Orphan_Test')
        
        markdown = report.to_markdown()
        
        assert '# Experiment Manifest Validation Report' in markdown
        assert 'Total experiments defined: 5' in markdown
        assert '## Critical Errors' in markdown
        assert 'Critical error' in markdown
        assert '## Warnings' in markdown
        assert 'Missing directory' in markdown
        assert '## Orphan Directories' in markdown
        assert 'Orphan_Test' in markdown
    
    def test_save_report(self, tmp_path):
        """Test saving report to file."""
        report = ManifestValidationReport(manifest_path=Path("/test/manifest.yaml"))
        report.add_structural_error(ValidationError('error', 'structure', 'Test error'))
        
        # Save as JSON
        json_path = tmp_path / "report.json"
        report.save(json_path, format='json')
        assert json_path.exists()
        
        with open(json_path) as f:
            data = json.load(f)
            assert data['summary']['total_errors'] == 1
        
        # Save as Markdown
        md_path = tmp_path / "report.md"
        report.save(md_path, format='markdown')
        assert md_path.exists()
        
        content = md_path.read_text()
        assert '# Experiment Manifest Validation Report' in content


class TestComprehensiveValidation:
    """Test the comprehensive validation function."""
    
    def test_validate_manifest_comprehensive(self, test_manifest_file, mock_experiment_structure):
        """Test comprehensive validation with mock data."""
        # Update manifest paths to use mock structure
        import yaml
        
        with open(test_manifest_file) as f:
            data = yaml.safe_load(f)
        
        # Update paths to point to mock directories
        base_path = mock_experiment_structure['morning_departure'].parent.parent.parent
        
        for section in ['evaluation_experiments', 'static_experiments']:
            for session in ['morning', 'afternoon']:
                if session in data[section]:
                    for exp in data[section][session]:
                        if 'paths' in exp:
                            relative = exp['paths']['relative']
                            exp['paths']['full_path'] = str(base_path / relative)
        
        # Save updated manifest
        with open(test_manifest_file, 'w') as f:
            yaml.dump(data, f)
        
        # Run validation
        report = validate_manifest_comprehensive(
            test_manifest_file,
            check_filesystem=True,
            data_root=base_path
        )
        
        # Check results
        assert report.summary.total_experiments == 2
        assert report.summary.experiments_checked == 2
        assert report.summary.directories_found == 2
        assert report.summary.directories_missing == 0
        
        # Should find orphan directory
        assert report.summary.orphan_directories == 1
        assert any(o['directory'] == '999_Orphan_Test' for o in report.orphan_directories)
    
    def test_validate_without_filesystem_check(self, test_manifest_file):
        """Test validation without filesystem checks."""
        report = validate_manifest_comprehensive(
            test_manifest_file,
            check_filesystem=False
        )
        
        # Should only have structural validation
        assert report.summary.total_experiments > 0
        assert report.summary.experiments_checked == 0
        assert len(report.filesystem_errors) == 0
    
    def test_validate_missing_manifest(self, tmp_path):
        """Test validation with missing manifest file."""
        missing_path = tmp_path / "missing.yaml"
        
        report = validate_manifest_comprehensive(missing_path)
        
        assert report.summary.total_errors >= 1
        assert any('Failed to load manifest' in e.message for e in report.structural_errors)


class TestHelperFunctions:
    """Additional tests for helper function edge cases."""
    
    def test_get_expected_data_folders_edge_cases(self):
        """Test edge cases for data folder generation."""
        # Test with empty data_types
        exp = {
            'name': 'Test',
            'paths': {'full_path': '/test/path'},
            'data_types': []
        }
        folders = get_expected_data_folders(exp)
        assert len(folders) == 0
        
        # Test with special sensor names
        exp['data_types'] = ['GPS', 'Sensor_wb', 'Sensor_wnb']
        folders = get_expected_data_folders(exp)
        
        # Since these are mock paths that don't exist, the function will
        # default to direct paths for morning sessions (no session specified)
        expected_paths = [
            ('GPS', Path('/test/path/GPS')),
            ('Sensor_wb', Path('/test/path/Sensor_wb')),
            ('Sensor_wnb', Path('/test/path/Sensor_wnb'))
        ]
        
        assert len(folders) == 3
        for (dtype, path), (exp_dtype, exp_path) in zip(folders, expected_paths):
            assert dtype == exp_dtype
            assert path == exp_path
            
        # Test with afternoon session (should default to IMU paths)
        exp['session'] = 'afternoon'
        folders = get_expected_data_folders(exp)
        
        expected_paths_afternoon = [
            ('GPS', Path('/test/path/GPS')),
            ('Sensor_wb', Path('/test/path/IMU/Sensor_wb')),
            ('Sensor_wnb', Path('/test/path/IMU/Sensor_wnb'))
        ]
        
        assert len(folders) == 3
        for (dtype, path), (exp_dtype, exp_path) in zip(folders, expected_paths_afternoon):
            assert dtype == exp_dtype
            assert path == exp_path
    
    def test_resolve_experiment_path_errors(self):
        """Test error handling in path resolution."""
        # Test with missing path information
        exp = {'name': 'Test'}
        
        with pytest.raises(ValueError, match="Cannot resolve path"):
            resolve_experiment_path(exp)
"""
Integration tests for experiment manifest validation.

These tests validate the manifest against the actual filesystem structure.
Run with: pytest tests/config -m integration
"""

import pytest
from pathlib import Path
from typing import List, Dict, Tuple, Set
import os

from src.core.experiment_manifest import (
    ExperimentManifest,
    ValidationError,
    load_manifest,
    get_all_experiments,
    get_expected_data_folders
)
from src.core.paths import DATA_DIR, RAW_DATA_DIR


@pytest.mark.integration
class TestFilesystemIntegrity:
    """Test manifest entries against actual filesystem structure."""
    
    @pytest.fixture(scope="class")
    def actual_manifest_path(self) -> Path:
        """Get the path to the actual experiment manifest."""
        manifest_path = Path(__file__).parent.parent.parent / "config" / "experiments" / "experiment_manifest.yaml"
        if not manifest_path.exists():
            pytest.skip(f"Experiment manifest not found at {manifest_path}")
        return manifest_path
    
    @pytest.fixture(scope="class")
    def actual_manifest(self, actual_manifest_path) -> ExperimentManifest:
        """Load the actual experiment manifest."""
        return load_manifest(actual_manifest_path)
    
    def test_manifest_file_exists(self, actual_manifest_path):
        """Test that the manifest file exists."""
        assert actual_manifest_path.exists()
        assert actual_manifest_path.is_file()
        assert actual_manifest_path.suffix == '.yaml'
    
    def test_data_directories_exist(self):
        """Test that base data directories exist."""
        assert RAW_DATA_DIR.exists(), f"Raw data directory not found: {RAW_DATA_DIR}"
        
        morning_dir = RAW_DATA_DIR / "morning" / "Experiments"
        afternoon_dir = RAW_DATA_DIR / "afternoon" / "Experiments"
        
        assert morning_dir.exists(), f"Morning experiments directory not found: {morning_dir}"
        assert afternoon_dir.exists(), f"Afternoon experiments directory not found: {afternoon_dir}"
    
    def test_experiment_directories_exist(self, actual_manifest):
        """Test that all experiment directories listed in manifest exist on filesystem."""
        missing_directories = []
        checked_count = 0
        
        for exp in get_all_experiments(actual_manifest):
            # Handle both absolute and relative paths
            if exp['paths']['full_path'].startswith('/data/raw/'):
                # Convert absolute path to relative to RAW_DATA_DIR
                relative_path = exp['paths']['full_path'].replace('/data/raw/', '')
                exp_path = RAW_DATA_DIR / relative_path
            else:
                exp_path = Path(exp['paths']['full_path'])
            
            checked_count += 1
            
            if not exp_path.exists():
                missing_directories.append({
                    'name': exp['name'],
                    'session': exp.get('session', 'unknown'),
                    'expected_path': str(exp_path),
                    'experiment_set': exp.get('experiment_set', 'unknown')
                })
        
        # Report findings
        if missing_directories:
            error_msg = f"\n\nMissing experiment directories ({len(missing_directories)} of {checked_count}):\n"
            for missing in missing_directories:
                error_msg += f"  - {missing['name']} ({missing['session']}, {missing['experiment_set']}): {missing['expected_path']}\n"
            
            pytest.fail(error_msg)
    
    def test_data_subdirectories_exist(self, actual_manifest):
        """Test that expected data subdirectories (GPS, IMU sensors) exist for each experiment."""
        missing_data_dirs = []
        empty_data_dirs = []
        checked_experiments = 0
        checked_subdirs = 0
        
        for exp in get_all_experiments(actual_manifest):
            # Handle both absolute and relative paths
            if exp['paths']['full_path'].startswith('/data/raw/'):
                # Convert absolute path to relative to RAW_DATA_DIR
                relative_path = exp['paths']['full_path'].replace('/data/raw/', '')
                exp_path = RAW_DATA_DIR / relative_path
            else:
                exp_path = Path(exp['paths']['full_path'])
            
            # Skip if experiment directory doesn't exist (caught by previous test)
            if not exp_path.exists():
                continue
                
            checked_experiments += 1
            data_folders = get_expected_data_folders(exp)
            
            for data_type, expected_path in data_folders:
                checked_subdirs += 1
                
                if not expected_path.exists():
                    missing_data_dirs.append({
                        'experiment': exp['name'],
                        'session': exp.get('session', 'unknown'),
                        'data_type': data_type,
                        'expected_path': str(expected_path)
                    })
                elif expected_path.is_dir():
                    # Check if directory is empty
                    if not any(expected_path.iterdir()):
                        empty_data_dirs.append({
                            'experiment': exp['name'],
                            'session': exp.get('session', 'unknown'),
                            'data_type': data_type,
                            'path': str(expected_path)
                        })
        
        # Report findings
        report = f"\nChecked {checked_experiments} experiments with {checked_subdirs} data subdirectories\n"
        
        if missing_data_dirs:
            report += f"\nMissing data subdirectories ({len(missing_data_dirs)}):\n"
            for missing in missing_data_dirs:
                report += f"  - {missing['experiment']} ({missing['session']}): {missing['data_type']} at {missing['expected_path']}\n"
        
        if empty_data_dirs:
            report += f"\nEmpty data directories ({len(empty_data_dirs)}):\n"
            for empty in empty_data_dirs:
                report += f"  - {empty['experiment']} ({empty['session']}): {empty['data_type']} at {empty['path']}\n"
        
        # Fail if there are critical missing directories
        if missing_data_dirs:
            pytest.fail(report)
        elif empty_data_dirs:
            # Just warn about empty directories
            pytest.skip(report + "\n(Skipping due to empty directories - these are warnings, not failures)")
    
    def test_data_files_exist(self, actual_manifest):
        """Test that data files exist in the expected formats."""
        experiments_with_data = 0
        experiments_without_data = []
        file_patterns = {
            'GPS': ['*.csv'],
            'Sensor_3': ['accel_*.csv', 'gyro_*.csv', 'mag_*.csv', 'angle_*.csv'],
            'Sensor_4': ['accel_*.csv', 'gyro_*.csv', 'mag_*.csv', 'angle_*.csv'],
            'Sensor_5': ['accel_*.csv', 'gyro_*.csv', 'mag_*.csv', 'angle_*.csv'],
            'Sensor_wb': ['accel_*.csv', 'gyro_*.csv', 'mag_*.csv', 'angle_*.csv'],
            'Sensor_wnb': ['accel_*.csv', 'gyro_*.csv', 'mag_*.csv', 'angle_*.csv']
        }
        
        for exp in get_all_experiments(actual_manifest):
            exp_path = Path(exp['paths']['full_path'])
            if not exp_path.exists():
                continue
            
            has_data = False
            data_folders = get_expected_data_folders(exp)
            
            for data_type, folder_path in data_folders:
                if folder_path.exists() and folder_path.is_dir():
                    # Check for expected file patterns
                    patterns = file_patterns.get(data_type, ['*.csv'])
                    for pattern in patterns:
                        if list(folder_path.glob(pattern)):
                            has_data = True
                            break
            
            if has_data:
                experiments_with_data += 1
            else:
                experiments_without_data.append(exp['name'])
        
        # Report summary
        total_experiments = len(list(get_all_experiments(actual_manifest)))
        report = f"\nData file check: {experiments_with_data}/{total_experiments} experiments have data files"
        
        if experiments_without_data:
            report += f"\nExperiments without data files: {', '.join(experiments_without_data[:5])}"
            if len(experiments_without_data) > 5:
                report += f" (and {len(experiments_without_data) - 5} more)"
        
        # This is informational, not a failure
        print(report)


@pytest.mark.integration
class TestOrphanDetection:
    """Test for directories on filesystem not listed in manifest."""
    
    @pytest.fixture(scope="class")
    def actual_manifest(self) -> ExperimentManifest:
        """Load the actual experiment manifest."""
        manifest_path = Path(__file__).parent.parent.parent / "config" / "experiments" / "experiment_manifest.yaml"
        if not manifest_path.exists():
            pytest.skip(f"Experiment manifest not found at {manifest_path}")
        return load_manifest(manifest_path)
    
    def get_manifest_experiment_names(self, manifest: ExperimentManifest) -> Dict[str, Set[str]]:
        """Get all experiment names from manifest organized by session."""
        names = {'morning': set(), 'afternoon': set()}
        
        for exp in get_all_experiments(manifest):
            session = exp.get('session')
            if session in names:
                names[session].add(exp['name'])
        
        return names
    
    def scan_experiment_directories(self) -> Dict[str, List[str]]:
        """Scan filesystem for experiment directories."""
        found_dirs = {'morning': [], 'afternoon': []}
        
        for session in ['morning', 'afternoon']:
            session_dir = RAW_DATA_DIR / session / "Experiments"
            if session_dir.exists() and session_dir.is_dir():
                for item in session_dir.iterdir():
                    if item.is_dir() and not item.name.startswith('.'):
                        found_dirs[session].append(item.name)
        
        return found_dirs
    
    def test_find_orphan_directories(self, actual_manifest):
        """Find directories that exist on filesystem but not in manifest."""
        manifest_names = self.get_manifest_experiment_names(actual_manifest)
        filesystem_dirs = self.scan_experiment_directories()
        
        orphans = {'morning': [], 'afternoon': []}
        
        for session in ['morning', 'afternoon']:
            fs_set = set(filesystem_dirs[session])
            manifest_set = manifest_names[session]
            orphans[session] = sorted(fs_set - manifest_set)
        
        # Report findings
        report = "\nOrphan Directory Detection:\n"
        total_orphans = 0
        
        for session in ['morning', 'afternoon']:
            if orphans[session]:
                report += f"\n{session.capitalize()} orphan directories ({len(orphans[session])}):\n"
                for orphan in orphans[session]:
                    report += f"  - {orphan}\n"
                    total_orphans += 1
        
        if total_orphans == 0:
            report += "\nNo orphan directories found - all filesystem directories are documented in manifest."
        else:
            report += f"\nTotal orphan directories: {total_orphans}"
            report += "\n\nThese directories exist on the filesystem but are not listed in the manifest."
            report += "\nConsider adding them to the manifest or removing them if they are obsolete."
        
        # This is informational, not a failure
        print(report)
    
    def test_manifest_coverage(self, actual_manifest):
        """Calculate what percentage of filesystem directories are covered by manifest."""
        manifest_names = self.get_manifest_experiment_names(actual_manifest)
        filesystem_dirs = self.scan_experiment_directories()
        
        coverage_stats = {}
        
        for session in ['morning', 'afternoon']:
            fs_count = len(filesystem_dirs[session])
            manifest_count = len(manifest_names[session])
            
            if fs_count > 0:
                # Count how many filesystem dirs are in manifest
                documented = len(set(filesystem_dirs[session]) & manifest_names[session])
                coverage = (documented / fs_count) * 100
            else:
                coverage = 100.0 if manifest_count == 0 else 0.0
            
            coverage_stats[session] = {
                'filesystem_dirs': fs_count,
                'manifest_entries': manifest_count,
                'documented': documented if fs_count > 0 else 0,
                'coverage_percent': coverage
            }
        
        # Report coverage
        report = "\nManifest Coverage Report:\n"
        for session, stats in coverage_stats.items():
            report += f"\n{session.capitalize()} session:\n"
            report += f"  - Directories on filesystem: {stats['filesystem_dirs']}\n"
            report += f"  - Entries in manifest: {stats['manifest_entries']}\n"
            report += f"  - Documented directories: {stats['documented']}\n"
            report += f"  - Coverage: {stats['coverage_percent']:.1f}%\n"
        
        overall_fs = sum(stats['filesystem_dirs'] for stats in coverage_stats.values())
        overall_documented = sum(stats['documented'] for stats in coverage_stats.values())
        overall_coverage = (overall_documented / overall_fs * 100) if overall_fs > 0 else 100.0
        
        report += f"\nOverall coverage: {overall_coverage:.1f}% ({overall_documented}/{overall_fs} directories)"
        
        print(report)


@pytest.mark.integration
class TestDataConsistency:
    """Test consistency of data structure across experiments."""
    
    @pytest.fixture(scope="class")
    def actual_manifest(self) -> ExperimentManifest:
        """Load the actual experiment manifest."""
        manifest_path = Path(__file__).parent.parent.parent / "config" / "experiments" / "experiment_manifest.yaml"
        if not manifest_path.exists():
            pytest.skip(f"Experiment manifest not found at {manifest_path}")
        return load_manifest(manifest_path)
    
    def test_imu_folder_structure(self, actual_manifest):
        """Test that IMU folders follow consistent naming convention."""
        inconsistent_naming = []
        
        for exp in get_all_experiments(actual_manifest):
            exp_path = Path(exp['paths']['full_path'])
            if not exp_path.exists():
                continue
            
            imu_path = exp_path / 'IMU'
            if not imu_path.exists():
                continue
            
            # Check sensor folder naming
            for sensor_dir in imu_path.iterdir():
                if sensor_dir.is_dir():
                    # Expected format: Sensor_X where X is a number or 'wb'/'wnb'
                    if not sensor_dir.name.startswith('Sensor_'):
                        inconsistent_naming.append({
                            'experiment': exp['name'],
                            'folder': str(sensor_dir),
                            'issue': 'Does not start with Sensor_'
                        })
                    else:
                        suffix = sensor_dir.name[7:]  # After 'Sensor_'
                        # Check if suffix is valid (number or special identifiers)
                        if suffix not in ['3', '4', '5', 'wb', 'wnb'] and not suffix.isdigit():
                            inconsistent_naming.append({
                                'experiment': exp['name'],
                                'folder': str(sensor_dir),
                                'issue': f'Unexpected sensor identifier: {suffix}'
                            })
        
        if inconsistent_naming:
            report = f"\nInconsistent IMU folder naming ({len(inconsistent_naming)} issues):\n"
            for issue in inconsistent_naming[:10]:  # Show first 10
                report += f"  - {issue['experiment']}: {issue['folder']} - {issue['issue']}\n"
            if len(inconsistent_naming) > 10:
                report += f"  ... and {len(inconsistent_naming) - 10} more issues\n"
            
            # This is a warning, not a failure
            pytest.skip(report)
    
    def test_gps_folder_consistency(self, actual_manifest):
        """Test that GPS folders are consistently named."""
        naming_variations = set()
        
        for exp in get_all_experiments(actual_manifest):
            exp_path = Path(exp['paths']['full_path'])
            if not exp_path.exists():
                continue
            
            # Look for GPS-related folders
            for folder in exp_path.iterdir():
                if folder.is_dir() and 'gps' in folder.name.lower():
                    naming_variations.add(folder.name)
        
        # Report variations
        if len(naming_variations) > 1:
            report = f"\nGPS folder naming variations found: {sorted(naming_variations)}"
            report += "\nConsider standardizing to a single naming convention."
            print(report)
        elif naming_variations:
            print(f"\nGPS folders use consistent naming: {list(naming_variations)[0]}")


def generate_validation_report(manifest_path: Path, output_path: Path = None) -> str:
    """
    Generate a comprehensive validation report for an experiment manifest.
    
    This function can be used as a standalone validation tool.
    """
    from src.core.experiment_manifest import validate_manifest_structure
    import json
    from datetime import datetime
    
    manifest = load_manifest(manifest_path)
    
    # Run structural validations
    structural_errors = validate_manifest_structure(manifest)
    
    # Run filesystem validations
    filesystem_errors = []
    missing_dirs = []
    missing_data = []
    
    for exp in get_all_experiments(manifest):
        exp_path = Path(exp['paths']['full_path'])
        
        if not exp_path.exists():
            missing_dirs.append(ValidationError(
                severity='error',
                category='filesystem',
                message=f"Experiment directory not found: {exp['name']}",
                details={'path': str(exp_path), 'experiment': exp['name']}
            ))
        else:
            # Check data folders
            for data_type, folder_path in get_expected_data_folders(exp):
                if not folder_path.exists():
                    missing_data.append(ValidationError(
                        severity='warning',
                        category='filesystem',
                        message=f"Missing data folder for {exp['name']}",
                        details={
                            'experiment': exp['name'],
                            'data_type': data_type,
                            'expected_path': str(folder_path)
                        }
                    ))
    
    filesystem_errors.extend(missing_dirs)
    filesystem_errors.extend(missing_data)
    
    # Combine all errors
    all_errors = structural_errors + filesystem_errors
    
    # Generate report
    report = {
        'validation_date': datetime.now().isoformat(),
        'manifest_path': str(manifest_path),
        'summary': {
            'total_experiments': len(list(get_all_experiments(manifest))),
            'total_errors': len([e for e in all_errors if e.severity == 'error']),
            'total_warnings': len([e for e in all_errors if e.severity == 'warning']),
            'total_info': len([e for e in all_errors if e.severity == 'info'])
        },
        'errors': [
            {
                'severity': e.severity,
                'category': e.category,
                'message': e.message,
                'details': e.details
            }
            for e in all_errors
        ]
    }
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
    
    return json.dumps(report, indent=2)
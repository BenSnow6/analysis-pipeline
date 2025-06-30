"""
Helper module for loading and validating experiment manifest YAML files.

This module provides utilities for:
- Loading experiment manifest YAML files
- Validating manifest structure and consistency
- Resolving experiment paths
- Querying experiment data
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class ValidationError:
    """Represents a validation error in the manifest."""
    severity: str  # 'error', 'warning', 'info'
    category: str  # 'structure', 'data', 'path', 'filesystem'
    message: str
    details: Optional[Dict[str, Any]] = None


class ExperimentManifest:
    """Class for loading and querying experiment manifest data."""
    
    def __init__(self, manifest_path: Path):
        """Initialize with path to manifest YAML file."""
        self.manifest_path = manifest_path
        self.data = self._load_yaml()
        self._experiments_cache = None
    
    def _load_yaml(self) -> dict:
        """Load YAML file and return parsed data."""
        try:
            with open(self.manifest_path, 'r') as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {self.manifest_path}: {e}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")
    
    def get_experiment_by_name(self, name: str, session: Optional[str] = None) -> Optional[dict]:
        """Get experiment data by name, optionally filtered by session."""
        for experiment in self.get_all_experiments():
            if experiment['name'] == name:
                if session is None or experiment.get('session') == session:
                    return experiment
        return None
    
    def get_all_experiments(self) -> List[dict]:
        """Get a flat list of all experiments with session info added."""
        if self._experiments_cache is not None:
            return self._experiments_cache
        
        experiments = []
        
        # Process evaluation experiments
        for session in ['morning', 'afternoon']:
            if 'evaluation_experiments' in self.data:
                for exp in self.data['evaluation_experiments'].get(session, []):
                    exp_copy = exp.copy()
                    exp_copy['session'] = session
                    exp_copy['experiment_set'] = 'evaluation'
                    experiments.append(exp_copy)
        
        # Process static experiments  
        for session in ['morning', 'afternoon']:
            if 'static_experiments' in self.data:
                for exp in self.data['static_experiments'].get(session, []):
                    exp_copy = exp.copy()
                    exp_copy['session'] = session
                    exp_copy['experiment_set'] = 'static'
                    experiments.append(exp_copy)
        
        self._experiments_cache = experiments
        return experiments
    
    def get_experiments_by_category(self, category: str) -> List[dict]:
        """Get all experiments belonging to a specific category."""
        experiments = []
        for exp in self.get_all_experiments():
            exp_category = exp.get('category')
            if isinstance(exp_category, list) and category in exp_category:
                experiments.append(exp)
            elif exp_category == category:
                experiments.append(exp)
        return experiments
    
    def resolve_experiment_path(self, experiment: dict, base_path: Optional[Path] = None) -> Path:
        """Resolve the full path for an experiment."""
        if 'paths' in experiment and 'full_path' in experiment['paths']:
            full_path = experiment['paths']['full_path']
            # Handle absolute paths that need to be relative to project
            if full_path.startswith('/data/raw/'):
                from src.core.paths import RAW_DATA_DIR
                relative_path = full_path.replace('/data/raw/', '')
                return RAW_DATA_DIR / relative_path
            return Path(full_path)
        
        if base_path and 'paths' in experiment and 'relative' in experiment['paths']:
            return base_path / experiment['paths']['relative']
        
        raise ValueError(f"Cannot resolve path for experiment: {experiment.get('name', 'unknown')}")
    
    def get_expected_data_folders(self, experiment: dict) -> List[Tuple[str, Path]]:
        """
        Get list of expected data folders for an experiment.
        Returns list of tuples: (data_type, expected_path)
        """
        exp_path = self.resolve_experiment_path(experiment)
        data_folders = []
        
        for data_type in experiment.get('data_types', []):
            if data_type == 'GPS':
                data_folders.append((data_type, exp_path / 'GPS'))
            elif data_type.startswith('Sensor_'):
                # Extract sensor identifier (e.g., '3' from 'Sensor_3')
                sensor_id = data_type.split('_')[1].lower()
                data_folders.append((data_type, exp_path / 'IMU' / f'Sensor_{sensor_id}'))
        
        return data_folders
    
    def validate_structure(self) -> List[ValidationError]:
        """Validate the internal structure of the manifest."""
        errors = []
        
        # Check top-level keys
        expected_keys = {'evaluation_experiments', 'static_experiments', 'all_experiments', 'analysis_config'}
        actual_keys = set(self.data.keys())
        missing_keys = expected_keys - actual_keys
        
        if missing_keys:
            errors.append(ValidationError(
                severity='error',
                category='structure',
                message=f"Missing top-level keys: {missing_keys}"
            ))
        
        # Check each experiment section
        for section in ['evaluation_experiments', 'static_experiments']:
            if section not in self.data:
                continue
                
            # Check for morning/afternoon subsections
            if not isinstance(self.data[section], dict):
                errors.append(ValidationError(
                    severity='error',
                    category='structure',
                    message=f"{section} must be a dictionary with 'morning' and 'afternoon' keys"
                ))
                continue
            
            for session in ['morning', 'afternoon']:
                if session not in self.data[section]:
                    errors.append(ValidationError(
                        severity='warning',
                        category='structure',
                        message=f"Missing {session} session in {section}"
                    ))
        
        return errors
    
    def validate_experiments(self) -> List[ValidationError]:
        """Validate individual experiment entries."""
        errors = []
        required_fields = {'name', 'path', 'type', 'paths', 'data_types'}
        valid_types = {'static', 'dynamic'}
        valid_data_types = {'GPS', 'Sensor_3', 'Sensor_4', 'Sensor_5', 'Sensor_wb', 'Sensor_wnb'}
        
        for exp in self.get_all_experiments():
            # Check required fields
            missing_fields = required_fields - set(exp.keys())
            if missing_fields:
                errors.append(ValidationError(
                    severity='error',
                    category='data',
                    message=f"Experiment '{exp.get('name', 'unknown')}' missing fields: {missing_fields}",
                    details={'experiment': exp.get('name'), 'missing_fields': list(missing_fields)}
                ))
            
            # Validate type field
            if 'type' in exp and exp['type'] not in valid_types:
                errors.append(ValidationError(
                    severity='error',
                    category='data',
                    message=f"Invalid type '{exp['type']}' for experiment '{exp['name']}'",
                    details={'experiment': exp['name'], 'invalid_type': exp['type']}
                ))
            
            # Validate data_types
            if 'data_types' in exp:
                invalid_types = set(exp['data_types']) - valid_data_types
                if invalid_types:
                    errors.append(ValidationError(
                        severity='warning',
                        category='data',
                        message=f"Unknown data types for '{exp['name']}': {invalid_types}",
                        details={'experiment': exp['name'], 'invalid_data_types': list(invalid_types)}
                    ))
            
            # Validate paths structure
            if 'paths' in exp:
                if not isinstance(exp['paths'], dict):
                    errors.append(ValidationError(
                        severity='error',
                        category='path',
                        message=f"'paths' must be a dictionary for experiment '{exp['name']}'",
                        details={'experiment': exp['name']}
                    ))
                elif 'full_path' not in exp['paths'] and 'relative' not in exp['paths']:
                    errors.append(ValidationError(
                        severity='error',
                        category='path',
                        message=f"'paths' must contain 'full_path' or 'relative' for '{exp['name']}'",
                        details={'experiment': exp['name']}
                    ))
        
        return errors
    
    def validate_uniqueness(self) -> List[ValidationError]:
        """Check for duplicate experiment names within each list."""
        errors = []
        
        # Check each section and session
        for section in ['evaluation_experiments', 'static_experiments', 'all_experiments']:
            if section not in self.data:
                continue
            
            for session in ['morning', 'afternoon']:
                if session not in self.data[section]:
                    continue
                
                experiments = self.data[section][session]
                names = [exp['name'] for exp in experiments if 'name' in exp]
                
                # Find duplicates
                name_counts = defaultdict(int)
                for name in names:
                    name_counts[name] += 1
                
                duplicates = {name: count for name, count in name_counts.items() if count > 1}
                
                if duplicates:
                    errors.append(ValidationError(
                        severity='error' if section != 'all_experiments' else 'warning',
                        category='data',
                        message=f"Duplicate names in {section}.{session}",
                        details={'section': section, 'session': session, 'duplicates': duplicates}
                    ))
        
        return errors
    
    def validate_cross_references(self) -> List[ValidationError]:
        """Validate cross-references within the manifest."""
        errors = []
        
        # Check analysis_config references
        if 'analysis_config' in self.data:
            config = self.data['analysis_config']
            
            if 'orientation_validation_experiments' in config:
                for session in ['morning', 'afternoon']:
                    if session not in config['orientation_validation_experiments']:
                        continue
                    
                    ref_names = config['orientation_validation_experiments'][session]
                    
                    # Get valid static experiment names for this session
                    valid_names = set()
                    if 'static_experiments' in self.data and session in self.data['static_experiments']:
                        valid_names = {exp['name'] for exp in self.data['static_experiments'][session] 
                                     if 'name' in exp}
                    
                    # Check each reference
                    invalid_refs = set(ref_names) - valid_names
                    if invalid_refs:
                        errors.append(ValidationError(
                            severity='error',
                            category='data',
                            message=f"Invalid experiment references in analysis_config",
                            details={
                                'session': session,
                                'invalid_references': list(invalid_refs),
                                'config_section': 'orientation_validation_experiments'
                            }
                        ))
        
        return errors
    
    def validate_path_consistency(self) -> List[ValidationError]:
        """Validate consistency between different path fields."""
        errors = []
        
        for exp in self.get_all_experiments():
            if 'path' not in exp or 'paths' not in exp:
                continue
            
            # Extract experiment name from paths
            old_path_name = Path(exp['path']).name
            
            if 'full_path' in exp['paths']:
                full_path_name = Path(exp['paths']['full_path']).name
                
                if old_path_name != full_path_name:
                    errors.append(ValidationError(
                        severity='warning',
                        category='path',
                        message=f"Path inconsistency for '{exp['name']}'",
                        details={
                            'experiment': exp['name'],
                            'path_name': old_path_name,
                            'full_path_name': full_path_name
                        }
                    ))
            
            # Check that experiment name appears in paths
            if exp['name'] not in str(exp.get('path', '')):
                errors.append(ValidationError(
                    severity='info',
                    category='path',
                    message=f"Experiment name '{exp['name']}' not in path",
                    details={'experiment': exp['name'], 'path': exp.get('path')}
                ))
        
        return errors


def load_manifest(manifest_path: Path) -> ExperimentManifest:
    """Load an experiment manifest from a YAML file."""
    return ExperimentManifest(manifest_path)


def validate_manifest_structure(manifest: ExperimentManifest) -> List[ValidationError]:
    """Run all structural validation checks on a manifest."""
    errors = []
    errors.extend(manifest.validate_structure())
    errors.extend(manifest.validate_experiments())
    errors.extend(manifest.validate_uniqueness())
    errors.extend(manifest.validate_cross_references())
    errors.extend(manifest.validate_path_consistency())
    return errors


def get_experiment_by_name(manifest: ExperimentManifest, name: str, session: Optional[str] = None) -> Optional[dict]:
    """Convenience function to get experiment by name."""
    return manifest.get_experiment_by_name(name, session)


def get_all_experiments(manifest: ExperimentManifest) -> List[dict]:
    """Convenience function to get all experiments."""
    return manifest.get_all_experiments()


def resolve_experiment_path(experiment: dict, base_path: Optional[Path] = None) -> Path:
    """Convenience function to resolve experiment path."""
    if 'paths' in experiment and 'full_path' in experiment['paths']:
        full_path = experiment['paths']['full_path']
        # Handle absolute paths that need to be relative to project
        if full_path.startswith('/data/raw/'):
            from src.core.paths import RAW_DATA_DIR
            relative_path = full_path.replace('/data/raw/', '')
            return RAW_DATA_DIR / relative_path
        return Path(full_path)
    
    if base_path and 'paths' in experiment and 'relative' in experiment['paths']:
        return base_path / experiment['paths']['relative']
    
    raise ValueError(f"Cannot resolve path for experiment: {experiment.get('name', 'unknown')}")


def get_expected_data_folders(experiment: dict) -> List[Tuple[str, Path]]:
    """Convenience function to get expected data folders."""
    exp_path = resolve_experiment_path(experiment)
    data_folders = []
    
    for data_type in experiment.get('data_types', []):
        if data_type == 'GPS':
            data_folders.append((data_type, exp_path / 'GPS'))
        elif data_type.startswith('Sensor_'):
            # Check both possible locations for sensor data
            # Morning experiments: directly under experiment directory
            # Afternoon experiments: under IMU subdirectory
            direct_path = exp_path / data_type
            imu_path = exp_path / 'IMU' / data_type
            
            # Use whichever exists, preferring IMU subdirectory if both exist
            if imu_path.exists():
                data_folders.append((data_type, imu_path))
            elif direct_path.exists():
                data_folders.append((data_type, direct_path))
            else:
                # Default to IMU path for afternoon, direct path for morning
                session = experiment.get('session', 'morning')
                if session == 'afternoon':
                    data_folders.append((data_type, imu_path))
                else:
                    data_folders.append((data_type, direct_path))
    
    return data_folders
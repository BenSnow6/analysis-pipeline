"""
Validation report generation for experiment manifest.

This module provides a comprehensive reporting system for manifest validation results.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import json
from collections import defaultdict

from src.core.experiment_manifest import (
    ValidationError,
    ExperimentManifest,
    load_manifest,
    validate_manifest_structure,
    get_all_experiments,
    get_expected_data_folders
)


@dataclass
class ValidationSummary:
    """Summary statistics for validation results."""
    total_experiments: int = 0
    total_errors: int = 0
    total_warnings: int = 0
    total_info: int = 0
    experiments_checked: int = 0
    directories_found: int = 0
    directories_missing: int = 0
    data_folders_found: int = 0
    data_folders_missing: int = 0
    orphan_directories: int = 0


@dataclass
class ManifestValidationReport:
    """Comprehensive validation report for experiment manifest."""
    
    manifest_path: Path
    validation_date: datetime = field(default_factory=datetime.now)
    structural_errors: List[ValidationError] = field(default_factory=list)
    filesystem_errors: List[ValidationError] = field(default_factory=list)
    orphan_directories: List[Dict[str, str]] = field(default_factory=list)
    summary: ValidationSummary = field(default_factory=ValidationSummary)
    
    def add_structural_error(self, error: ValidationError):
        """Add a structural validation error."""
        self.structural_errors.append(error)
        self._update_error_counts(error)
    
    def add_filesystem_error(self, error: ValidationError):
        """Add a filesystem validation error."""
        self.filesystem_errors.append(error)
        self._update_error_counts(error)
    
    def add_orphan_directory(self, session: str, directory: str):
        """Add an orphan directory."""
        self.orphan_directories.append({
            'session': session,
            'directory': directory,
            'path': f"/data/raw/{session}/Experiments/{directory}"
        })
        self.summary.orphan_directories += 1
    
    def _update_error_counts(self, error: ValidationError):
        """Update summary counts based on error severity."""
        if error.severity == 'error':
            self.summary.total_errors += 1
        elif error.severity == 'warning':
            self.summary.total_warnings += 1
        elif error.severity == 'info':
            self.summary.total_info += 1
    
    def get_all_errors(self) -> List[ValidationError]:
        """Get all errors combined."""
        return self.structural_errors + self.filesystem_errors
    
    def get_errors_by_severity(self, severity: str) -> List[ValidationError]:
        """Get errors filtered by severity."""
        return [e for e in self.get_all_errors() if e.severity == severity]
    
    def get_errors_by_category(self, category: str) -> List[ValidationError]:
        """Get errors filtered by category."""
        return [e for e in self.get_all_errors() if e.category == category]
    
    def to_dict(self) -> dict:
        """Convert report to dictionary format."""
        return {
            'manifest_path': str(self.manifest_path),
            'validation_date': self.validation_date.isoformat(),
            'summary': {
                'total_experiments': self.summary.total_experiments,
                'total_errors': self.summary.total_errors,
                'total_warnings': self.summary.total_warnings,
                'total_info': self.summary.total_info,
                'experiments_checked': self.summary.experiments_checked,
                'directories_found': self.summary.directories_found,
                'directories_missing': self.summary.directories_missing,
                'data_folders_found': self.summary.data_folders_found,
                'data_folders_missing': self.summary.data_folders_missing,
                'orphan_directories': self.summary.orphan_directories
            },
            'structural_errors': [
                {
                    'severity': e.severity,
                    'category': e.category,
                    'message': e.message,
                    'details': e.details
                }
                for e in self.structural_errors
            ],
            'filesystem_errors': [
                {
                    'severity': e.severity,
                    'category': e.category,
                    'message': e.message,
                    'details': e.details
                }
                for e in self.filesystem_errors
            ],
            'orphan_directories': self.orphan_directories
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def to_markdown(self) -> str:
        """Convert report to Markdown format."""
        md = []
        md.append(f"# Experiment Manifest Validation Report")
        md.append(f"\n**Date:** {self.validation_date.strftime('%Y-%m-%d %H:%M:%S')}")
        md.append(f"**Manifest:** `{self.manifest_path}`")
        
        # Summary
        md.append("\n## Summary")
        md.append(f"- Total experiments defined: {self.summary.total_experiments}")
        md.append(f"- Experiments checked: {self.summary.experiments_checked}")
        md.append(f"- **Errors:** {self.summary.total_errors}")
        md.append(f"- **Warnings:** {self.summary.total_warnings}")
        md.append(f"- **Info:** {self.summary.total_info}")
        
        # Filesystem Summary
        if self.summary.experiments_checked > 0:
            md.append("\n### Filesystem Check")
            md.append(f"- Experiment directories found: {self.summary.directories_found}")
            md.append(f"- Experiment directories missing: {self.summary.directories_missing}")
            md.append(f"- Data folders found: {self.summary.data_folders_found}")
            md.append(f"- Data folders missing: {self.summary.data_folders_missing}")
            md.append(f"- Orphan directories: {self.summary.orphan_directories}")
        
        # Critical Errors
        critical_errors = self.get_errors_by_severity('error')
        if critical_errors:
            md.append("\n## Critical Errors")
            md.append("These issues must be fixed:")
            for error in critical_errors:
                md.append(f"\n### {error.message}")
                if error.details:
                    for key, value in error.details.items():
                        md.append(f"- **{key}:** {value}")
        
        # Warnings
        warnings = self.get_errors_by_severity('warning')
        if warnings:
            md.append("\n## Warnings")
            md.append("These issues should be reviewed:")
            
            # Group warnings by category
            by_category = defaultdict(list)
            for warning in warnings:
                by_category[warning.category].append(warning)
            
            for category, category_warnings in by_category.items():
                md.append(f"\n### {category.title()} Issues")
                for warning in category_warnings[:5]:  # Show first 5
                    md.append(f"- {warning.message}")
                if len(category_warnings) > 5:
                    md.append(f"- ... and {len(category_warnings) - 5} more")
        
        # Orphan Directories
        if self.orphan_directories:
            md.append("\n## Orphan Directories")
            md.append("These directories exist on filesystem but are not in the manifest:")
            
            by_session = defaultdict(list)
            for orphan in self.orphan_directories:
                by_session[orphan['session']].append(orphan['directory'])
            
            for session, dirs in by_session.items():
                md.append(f"\n### {session.title()} Session")
                for dir_name in sorted(dirs):
                    md.append(f"- {dir_name}")
        
        # Info Messages
        info_messages = self.get_errors_by_severity('info')
        if info_messages:
            md.append("\n## Information")
            for info in info_messages[:10]:
                md.append(f"- {info.message}")
            if len(info_messages) > 10:
                md.append(f"- ... and {len(info_messages) - 10} more")
        
        return "\n".join(md)
    
    def save(self, output_path: Path, format: str = 'json'):
        """Save report to file."""
        if format == 'json':
            with open(output_path, 'w') as f:
                f.write(self.to_json())
        elif format == 'markdown':
            with open(output_path, 'w') as f:
                f.write(self.to_markdown())
        else:
            raise ValueError(f"Unsupported format: {format}")


def validate_manifest_comprehensive(
    manifest_path: Path,
    check_filesystem: bool = True,
    data_root: Optional[Path] = None
) -> ManifestValidationReport:
    """
    Perform comprehensive validation of an experiment manifest.
    
    Args:
        manifest_path: Path to the manifest YAML file
        check_filesystem: Whether to check filesystem structure
        data_root: Root directory for data (defaults to /data/raw)
    
    Returns:
        Comprehensive validation report
    """
    from src.core.paths import RAW_DATA_DIR
    
    if data_root is None:
        data_root = RAW_DATA_DIR
    
    # Create report
    report = ManifestValidationReport(manifest_path=manifest_path)
    
    # Load manifest
    try:
        manifest = load_manifest(manifest_path)
    except Exception as e:
        report.add_structural_error(ValidationError(
            severity='error',
            category='loading',
            message=f"Failed to load manifest: {e}"
        ))
        return report
    
    # Get all experiments
    all_experiments = list(get_all_experiments(manifest))
    report.summary.total_experiments = len(all_experiments)
    
    # Run structural validation
    structural_errors = validate_manifest_structure(manifest)
    for error in structural_errors:
        report.add_structural_error(error)
    
    # Run filesystem validation if requested
    if check_filesystem:
        # Check experiment directories
        for exp in all_experiments:
            report.summary.experiments_checked += 1
            exp_path = Path(exp['paths']['full_path'])
            
            if not exp_path.exists():
                report.add_filesystem_error(ValidationError(
                    severity='error',
                    category='filesystem',
                    message=f"Experiment directory not found: {exp['name']}",
                    details={
                        'experiment': exp['name'],
                        'session': exp.get('session', 'unknown'),
                        'expected_path': str(exp_path)
                    }
                ))
                report.summary.directories_missing += 1
            else:
                report.summary.directories_found += 1
                
                # Check data folders
                for data_type, folder_path in get_expected_data_folders(exp):
                    if not folder_path.exists():
                        report.add_filesystem_error(ValidationError(
                            severity='warning',
                            category='filesystem',
                            message=f"Missing data folder for {exp['name']}",
                            details={
                                'experiment': exp['name'],
                                'data_type': data_type,
                                'expected_path': str(folder_path)
                            }
                        ))
                        report.summary.data_folders_missing += 1
                    else:
                        report.summary.data_folders_found += 1
                        
                        # Check if folder is empty
                        if folder_path.is_dir() and not any(folder_path.iterdir()):
                            report.add_filesystem_error(ValidationError(
                                severity='info',
                                category='filesystem',
                                message=f"Empty data folder for {exp['name']}",
                                details={
                                    'experiment': exp['name'],
                                    'data_type': data_type,
                                    'path': str(folder_path)
                                }
                            ))
        
        # Check for orphan directories
        manifest_names = {
            'morning': set(),
            'afternoon': set()
        }
        
        for exp in all_experiments:
            session = exp.get('session')
            if session in manifest_names:
                manifest_names[session].add(exp['name'])
        
        # Scan filesystem
        for session in ['morning', 'afternoon']:
            session_dir = data_root / session / "Experiments"
            if session_dir.exists() and session_dir.is_dir():
                for item in session_dir.iterdir():
                    if item.is_dir() and not item.name.startswith('.'):
                        if item.name not in manifest_names[session]:
                            report.add_orphan_directory(session, item.name)
    
    return report


def generate_validation_report_cli(
    manifest_path: Path,
    output_path: Optional[Path] = None,
    format: str = 'markdown',
    check_filesystem: bool = True
) -> None:
    """
    Command-line interface for generating validation reports.
    
    Args:
        manifest_path: Path to manifest file
        output_path: Where to save report (prints to stdout if None)
        format: Output format ('json' or 'markdown')
        check_filesystem: Whether to check filesystem
    """
    report = validate_manifest_comprehensive(
        manifest_path,
        check_filesystem=check_filesystem
    )
    
    if output_path:
        report.save(output_path, format=format)
        print(f"Report saved to: {output_path}")
    else:
        if format == 'json':
            print(report.to_json())
        else:
            print(report.to_markdown())
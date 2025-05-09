#!/usr/bin/env python3
"""
Repository Structure Generator

This script analyzes the directory structure of the repository and generates
a formatted output showing the organization of files and directories.
"""

import os
import argparse
from collections import defaultdict
import json
from pathlib import Path

def count_files_by_extension(path):
    """Count files by their extension within a directory (recursive)."""
    extension_counts = defaultdict(int)
    
    for root, _, files in os.walk(path):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext:
                extension_counts[ext.lower()] += 1
            else:
                extension_counts['(no extension)'] += 1
    
    return extension_counts

def get_experiment_stats(experiment_dir):
    """Get statistics about experiment directories and data files."""
    stats = {
        'experiments': 0,
        'test_cases': 0,
        'data_files': 0,
        'gps_files': 0,
        'imu_files': 0,
        'sensor_types': set(),
    }
    
    if not os.path.exists(experiment_dir):
        return stats
    
    # Count experiment types (e.g., 1a_1, 1b_1, etc.)
    experiment_types = [d for d in os.listdir(experiment_dir) 
                      if os.path.isdir(os.path.join(experiment_dir, d))]
    stats['experiments'] = len(experiment_types)
    
    # Count test cases and data files
    for root, dirs, files in os.walk(experiment_dir):
        # Identify test case directories (usually numeric prefixed)
        test_dirs = [d for d in dirs if any(c.isdigit() for c in d)]
        if test_dirs and any(parent in root for parent in experiment_types):
            stats['test_cases'] += len(test_dirs)
        
        # Count data files
        for file in files:
            if file.endswith('.csv'):
                stats['data_files'] += 1
                
                # Count GPS and IMU files
                if 'GPS' in file or 'gps' in file:
                    stats['gps_files'] += 1
                if any(s in file for s in ['accel', 'gyro', 'mag', 'angle', 'quat']):
                    stats['imu_files'] += 1
                
                # Track sensor types
                for sensor in ['Sensor_3', 'Sensor_4', 'Sensor_5', 'Sensor_wb', 'Sensor_wnb']:
                    if sensor in root:
                        stats['sensor_types'].add(sensor)
    
    stats['sensor_types'] = list(stats['sensor_types'])
    return stats

def create_tree(path, prefix='', exclude_dirs=None):
    """Create a tree representation of the directory structure."""
    if exclude_dirs is None:
        exclude_dirs = ['.git', '__pycache__', '.ipynb_checkpoints', '.claude']
    
    if os.path.basename(path) in exclude_dirs:
        return ""
    
    output = []
    
    # Get directories and files
    entries = sorted(os.listdir(path))
    dirs = [e for e in entries if os.path.isdir(os.path.join(path, e)) and e not in exclude_dirs]
    files = [e for e in entries if os.path.isfile(os.path.join(path, e))]
    
    # Add directories
    for i, d in enumerate(dirs):
        is_last_dir = (i == len(dirs) - 1 and not files)
        output.append(f"{prefix}{'└── ' if is_last_dir else '├── '}{d}/")
        output.append(create_tree(
            os.path.join(path, d),
            prefix + ('    ' if is_last_dir else '│   '),
            exclude_dirs
        ))
    
    # Add files (limit to 5 files per dir to avoid overwhelming output)
    if len(files) > 5:
        for i, f in enumerate(files[:5]):
            output.append(f"{prefix}{'└── ' if i == 4 else '├── '}{f}")
        output.append(f"{prefix}└── ... ({len(files)-5} more files)")
    else:
        for i, f in enumerate(files):
            output.append(f"{prefix}{'└── ' if i == len(files) - 1 else '├── '}{f}")
    
    return "\n".join(output)

def generate_structure_report(repo_path, max_depth=3, simplify=True):
    """Generate a complete report of the repository structure."""
    report = {
        'repo_name': os.path.basename(os.path.abspath(repo_path)),
        'file_counts': count_files_by_extension(repo_path),
        'experiment_stats': get_experiment_stats(os.path.join(repo_path, '02_Evaluation_Experiments')),
        'directory_tree': create_simplified_tree(repo_path) if simplify else create_tree(repo_path),
    }
    
    # Add source code info
    src_path = os.path.join(repo_path, 'src')
    if os.path.exists(src_path):
        report['source_files'] = os.listdir(src_path)
    
    # Add notebook info
    notebooks_path = os.path.join(repo_path, 'notebooks')
    if os.path.exists(notebooks_path):
        report['notebooks'] = os.listdir(notebooks_path)
    
    return report

def create_simplified_tree(repo_path):
    """Create a simplified tree that doesn't go too deep into repeated structures."""
    output = []
    
    # Main directories at the root
    main_dirs = [d for d in os.listdir(repo_path) 
                if os.path.isdir(os.path.join(repo_path, d)) 
                and d not in ['.git', '__pycache__', '.ipynb_checkpoints', '.claude']]
    
    main_files = [f for f in os.listdir(repo_path) if os.path.isfile(os.path.join(repo_path, f))]
    
    # Process main directories
    for d in sorted(main_dirs):
        output.append(f"├── {d}/")
        
        # Handle each main directory differently based on its content pattern
        if d == 'src':
            # Show all files in src
            src_files = os.listdir(os.path.join(repo_path, d))
            for i, f in enumerate(sorted(src_files)):
                prefix = '│   └── ' if i == len(src_files) - 1 else '│   ├── '
                output.append(f"{prefix}{f}")
                
        elif d == 'notebooks':
            # Show all notebooks
            nb_files = os.listdir(os.path.join(repo_path, d))
            for i, f in enumerate(sorted(nb_files)):
                prefix = '│   └── ' if i == len(nb_files) - 1 else '│   ├── '
                output.append(f"{prefix}{f}")
                
        elif d == '02_Evaluation_Experiments':
            # For experiments, show main categories but summarize deeper levels
            exp_path = os.path.join(repo_path, d)
            exp_dirs = sorted([subd for subd in os.listdir(exp_path) 
                        if os.path.isdir(os.path.join(exp_path, subd))])
            
            for i, exp in enumerate(exp_dirs):
                exp_prefix = '│   └── ' if i == len(exp_dirs) - 1 else '│   ├── '
                output.append(f"{exp_prefix}{exp}/")
                
                # For each experiment type, count test cases but don't list all
                test_path = os.path.join(exp_path, exp)
                test_dirs = [td for td in os.listdir(test_path) 
                           if os.path.isdir(os.path.join(test_path, td))]
                
                if test_dirs:
                    next_prefix = '│       ' if i < len(exp_dirs) - 1 else '        '
                    output.append(f"{next_prefix}└── {len(test_dirs)} test cases with sensor and GPS data")
        
        elif d == 'GPS testing':
            # Show GPS testing files
            gps_files = os.listdir(os.path.join(repo_path, d))
            for i, f in enumerate(sorted(gps_files)):
                prefix = '│   └── ' if i == len(gps_files) - 1 else '│   ├── '
                output.append(f"{prefix}{f}")
        
        else:
            # Generic directory handling
            subpath = os.path.join(repo_path, d)
            subdirs = [sd for sd in os.listdir(subpath) if os.path.isdir(os.path.join(subpath, sd))]
            subfiles = [sf for sf in os.listdir(subpath) if os.path.isfile(os.path.join(subpath, sf))]
            
            if subdirs:
                output.append(f"│   └── {len(subdirs)} subdirectories, {len(subfiles)} files")
            elif subfiles:
                for i, f in enumerate(sorted(subfiles[:5])):
                    prefix = '│   └── ' if i == min(4, len(subfiles) - 1) else '│   ├── '
                    output.append(f"{prefix}{f}")
                if len(subfiles) > 5:
                    output.append(f"│       └── ... ({len(subfiles)-5} more files)")
    
    # Add main files at the end
    for i, f in enumerate(sorted(main_files)):
        prefix = '└── ' if i == len(main_files) - 1 else '├── '
        output.append(f"{prefix}{f}")
    
    return "\n".join(output)

def format_report(report, output_format='text'):
    """Format the report according to the specified output format."""
    if output_format == 'json':
        return json.dumps(report, indent=2)
    
    # Text report
    lines = []
    lines.append(f"# Repository Structure: {report['repo_name']}")
    lines.append("")
    
    # File statistics
    lines.append("## File Statistics")
    total_files = sum(report['file_counts'].values())
    lines.append(f"Total files: {total_files}")
    lines.append("Files by extension:")
    for ext, count in sorted(report['file_counts'].items(), key=lambda x: x[1], reverse=True):
        lines.append(f"  {ext}: {count}")
    lines.append("")
    
    # Experiment statistics
    stats = report['experiment_stats']
    lines.append("## Experiment Statistics")
    lines.append(f"Experiment types: {stats['experiments']}")
    lines.append(f"Test cases: {stats['test_cases']}")
    lines.append(f"Data files: {stats['data_files']}")
    lines.append(f"GPS files: {stats['gps_files']}")
    lines.append(f"IMU files: {stats['imu_files']}")
    lines.append(f"Sensor types: {', '.join(stats['sensor_types'])}")
    lines.append("")
    
    # Source files
    if 'source_files' in report:
        lines.append("## Source Files")
        for src in sorted(report['source_files']):
            lines.append(f"  - {src}")
        lines.append("")
    
    # Notebooks
    if 'notebooks' in report:
        lines.append("## Notebooks")
        for nb in sorted(report['notebooks']):
            lines.append(f"  - {nb}")
        lines.append("")
    
    # Directory tree
    lines.append("## Directory Structure")
    lines.append("```")
    lines.append(report['directory_tree'])
    lines.append("```")
    
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description='Generate a repository structure report')
    parser.add_argument('--path', default='.', help='Path to the repository')
    parser.add_argument('--format', choices=['text', 'json'], default='text', 
                        help='Output format (text or json)')
    parser.add_argument('--output', help='Output file (if not specified, prints to stdout)')
    parser.add_argument('--full', action='store_true', 
                        help='Show full directory tree instead of simplified version')
    
    args = parser.parse_args()
    
    report = generate_structure_report(args.path, simplify=not args.full)
    formatted_report = format_report(report, args.format)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(formatted_report)
        print(f"Report written to {args.output}")
    else:
        print(formatted_report)

if __name__ == '__main__':
    main()
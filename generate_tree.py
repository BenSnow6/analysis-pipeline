#!/usr/bin/env python3
"""
Generate a tree structure of the repository for AI analysis.
Shows high-level structure for /data directory but full details elsewhere.
"""

import os
from pathlib import Path
from typing import List, Set

def should_skip(path: Path, skip_patterns: Set[str]) -> bool:
    """Check if path should be skipped."""
    parts = path.parts
    
    # Skip hidden files and directories
    if any(part.startswith('.') for part in parts):
        return True
    
    # Skip specific directories
    if any(pattern in str(path) for pattern in skip_patterns):
        return True
    
    # Skip __pycache__ directories
    if '__pycache__' in parts:
        return True
    
    return False

def should_show_as_summary(path: Path) -> bool:
    """Check if directory should be shown as a single summary line."""
    name = path.name.lower()
    # Virtual environments and node modules should just be noted as existing
    if name in ['venv', '.venv', 'env', '.env', 'virtualenv', 'node_modules']:
        return True
    return False

def should_summarize_data(path: Path, root: Path) -> bool:
    """Check if we should summarize this data directory."""
    rel_path = path.relative_to(root)
    parts = rel_path.parts
    
    # Only summarize deep experiment data
    if len(parts) >= 2 and parts[0] == 'data':
        if parts[1] in ['raw', 'processed']:
            # Show structure up to experiment level
            if len(parts) > 4:
                return True
            # For processed data, show one more level
            if parts[1] == 'processed' and len(parts) > 5:
                return True
    
    return False

def get_tree_structure(root_path: Path, prefix: str = "", skip_patterns: Set[str] = None) -> List[str]:
    """Generate tree structure as list of strings."""
    if skip_patterns is None:
        skip_patterns = {'.git', '__pycache__', '.pytest_cache'}
    
    lines = []
    items = sorted(root_path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
    
    for i, item in enumerate(items):
        if should_skip(item, skip_patterns):
            continue
        
        is_last = i == len(items) - 1
        current_prefix = "└── " if is_last else "├── "
        next_prefix = "    " if is_last else "│   "
        
        if item.is_file():
            # Always show files
            lines.append(f"{prefix}{current_prefix}{item.name}")
        else:
            # Directory
            if should_show_as_summary(item):
                # Show virtual environments and node_modules as a single line
                lines.append(f"{prefix}{current_prefix}{item.name}/ [virtual environment]")
            else:
                lines.append(f"{prefix}{current_prefix}{item.name}/")
                
                # Check if we should summarize this directory
                if should_summarize_data(item, root_path.parent):
                    # Count subdirectories
                    subdirs = [d for d in item.iterdir() if d.is_dir() and not should_skip(d, skip_patterns)]
                    files = [f for f in item.iterdir() if f.is_file() and not should_skip(f, skip_patterns)]
                    
                    if subdirs:
                        lines.append(f"{prefix}{next_prefix}├── [{len(subdirs)} subdirectories]")
                    if files:
                        # Show file types
                        extensions = set(f.suffix for f in files if f.suffix)
                        file_summary = f"[{len(files)} files"
                        if extensions:
                            file_summary += f": {', '.join(sorted(extensions))}"
                        file_summary += "]"
                        lines.append(f"{prefix}{next_prefix}└── {file_summary}")
                else:
                    # Recurse into directory
                    sub_lines = get_tree_structure(item, prefix + next_prefix, skip_patterns)
                    lines.extend(sub_lines)
    
    return lines

def main():
    """Generate and save tree structure."""
    repo_root = Path(__file__).parent
    
    print("Generating repository tree structure...")
    print(f"Repository root: {repo_root}")
    print()
    
    # Generate tree
    tree_lines = [f"{repo_root.name}/"]
    tree_lines.extend(get_tree_structure(repo_root, "", skip_patterns={'.git', '__pycache__', '.pytest_cache'}))
    
    # Save to file
    output_file = repo_root / "repository_tree.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(tree_lines))
    
    # Also print to console
    print('\n'.join(tree_lines))
    print()
    print(f"Tree structure saved to: {output_file}")
    
    # Print summary statistics
    total_lines = len(tree_lines)
    print(f"\nTotal lines in tree: {total_lines}")

if __name__ == "__main__":
    main()
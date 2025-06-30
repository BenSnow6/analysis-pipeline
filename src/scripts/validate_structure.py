#!/usr/bin/env python3
"""Validate repository structure after migration."""

from pathlib import Path
import subprocess
import sys
import os

def check_git_status():
    """Check for uncommitted changes."""
    try:
        result = subprocess.run(['git', 'status', '--porcelain'], 
                              capture_output=True, text=True, check=True)
        if result.stdout.strip():
            return f"WARNING: Uncommitted changes detected:\n{result.stdout}"
    except subprocess.CalledProcessError as e:
        return f"ERROR: Could not check git status: {e}"
    return None

def check_imports():
    """Verify key imports still work."""
    failed_imports = []
    
    imports_to_check = [
        "from src.analysis.rpm import preprocess",
        "from src.analysis.rpm import spectral",
        "from src.analysis.rpm import fusion",
        "from src.core import io",
        "from src.core.paths import DATA_DIR",
    ]
    
    for import_stmt in imports_to_check:
        try:
            exec(import_stmt)
        except ImportError as e:
            failed_imports.append(f"{import_stmt}: {e}")
    
    if failed_imports:
        return "ERROR: Failed imports:\n" + "\n".join(failed_imports)
    return None

def check_directory_structure():
    """Verify expected directory structure exists."""
    expected_dirs = [
        "src/analysis/rpm",
        "src/core",
        "src/scripts",
        "docs/results/rpm_estimation",
        "docs/development/rpm",
        "docs/migration",
        "data/processed",
        "config",
    ]
    
    missing_dirs = []
    for dir_path in expected_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        return "ERROR: Missing directories:\n" + "\n".join(missing_dirs)
    return None

def check_rpm_module():
    """Verify RPM module structure is correct."""
    rpm_path = Path("src/analysis/rpm")
    
    # Should have Python files
    py_files = list(rpm_path.glob("*.py"))
    if len(py_files) < 5:
        return f"WARNING: Only {len(py_files)} Python files in RPM module"
    
    # Should NOT have docs or markdown files
    md_files = list(rpm_path.glob("*.md"))
    if md_files:
        return f"ERROR: Markdown files still in src/analysis/rpm: {[f.name for f in md_files]}"
    
    # Should NOT have docs directory
    if (rpm_path / "docs").exists():
        return "ERROR: docs/ directory still exists in src/analysis/rpm"
    
    return None

def check_docs_consolidation():
    """Verify documentation has been consolidated."""
    docs_rpm = Path("docs/results/rpm_estimation")
    
    # Should have README
    if not (docs_rpm / "README.md").exists():
        return "WARNING: No README.md in docs/results/rpm_estimation"
    
    # Should have work packages
    wp_dirs = list(docs_rpm.glob("wp*"))
    if len(wp_dirs) < 5:
        return f"WARNING: Only {len(wp_dirs)} work package directories found"
    
    return None

def check_disk_space():
    """Check available disk space."""
    stat = os.statvfs('.')
    free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
    if free_gb < 5:
        return f"WARNING: Low disk space: {free_gb:.1f}GB free"
    return None

def main():
    """Run all validation checks."""
    print("Repository Structure Validation")
    print("=" * 50)
    
    checks = [
        ("Git Status", check_git_status),
        ("Directory Structure", check_directory_structure),
        ("Python Imports", check_imports),
        ("RPM Module Structure", check_rpm_module),
        ("Documentation Consolidation", check_docs_consolidation),
        ("Disk Space", check_disk_space),
    ]
    
    issues = []
    for check_name, check_func in checks:
        print(f"\nChecking {check_name}...", end=" ")
        result = check_func()
        if result:
            print("ISSUE FOUND")
            print(result)
            issues.append((check_name, result))
        else:
            print("OK")
    
    print("\n" + "=" * 50)
    if issues:
        print(f"Validation completed with {len(issues)} issue(s)")
        return 1
    else:
        print("All validation checks passed!")
        return 0

if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""Migrate RPM documentation to centralized location."""

from pathlib import Path
import shutil
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def migrate_rpm_docs():
    """Consolidate RPM documentation from src to docs."""
    
    # Define paths
    src_rpm = Path("src/analysis/rpm")
    docs_rpm = Path("docs/results/rpm_estimation")
    docs_dev_rpm = Path("docs/development/rpm")
    
    # Ensure destination directories exist
    docs_rpm.mkdir(parents=True, exist_ok=True)
    docs_dev_rpm.mkdir(parents=True, exist_ok=True)
    
    migrations = []
    
    # 1. Move README.md to docs/results/rpm_estimation/
    src_readme = src_rpm / "README.md"
    dst_readme = docs_rpm / "README.md"
    if src_readme.exists():
        if dst_readme.exists():
            logger.warning(f"Destination {dst_readme} already exists. Creating backup.")
            backup_path = dst_readme.with_suffix('.md.backup')
            shutil.copy2(str(dst_readme), str(backup_path))
        migrations.append((src_readme, dst_readme))
    
    # 2. Check for docs/work_packages directory
    src_wp_dir = src_rpm / "docs" / "work_packages"
    if src_wp_dir.exists():
        for wp_dir in src_wp_dir.glob("wp*"):
            dst_wp = docs_rpm / wp_dir.name
            if dst_wp.exists():
                logger.warning(f"Work package {dst_wp} already exists. Will merge contents.")
                # Move individual files instead of directory
                for file in wp_dir.glob("*"):
                    dst_file = dst_wp / file.name
                    if dst_file.exists():
                        backup_file = dst_file.with_suffix(file.suffix + '.backup')
                        shutil.copy2(str(dst_file), str(backup_file))
                        logger.info(f"Created backup: {backup_file}")
                    migrations.append((file, dst_file))
            else:
                migrations.append((wp_dir, dst_wp))
    
    # 3. Check for other markdown files
    for md_file in src_rpm.glob("*.md"):
        if md_file.name == "README.md":
            continue  # Already handled
        
        # Determine destination based on content
        if "development" in md_file.name.lower() or "checklist" in md_file.name.lower():
            dst_file = docs_dev_rpm / md_file.name
        else:
            dst_file = docs_rpm / md_file.name
        
        if dst_file.exists():
            backup_file = dst_file.with_suffix('.md.backup')
            shutil.copy2(str(dst_file), str(backup_file))
            logger.info(f"Created backup: {backup_file}")
        
        migrations.append((md_file, dst_file))
    
    # 4. Check for results directory
    src_results = src_rpm / "results"
    if src_results.exists() and src_results.is_dir():
        # Results should go to docs/results/rpm_estimation/
        for item in src_results.rglob("*"):
            if item.is_file():
                relative_path = item.relative_to(src_results)
                dst_path = docs_rpm / "results" / relative_path
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                migrations.append((item, dst_path))
    
    # Perform migrations
    if not migrations:
        logger.info("No files to migrate.")
        return
    
    logger.info(f"Found {len(migrations)} files to migrate:")
    for src, dst in migrations:
        logger.info(f"  {src} → {dst}")
    
    # Auto-proceed in non-interactive mode
    if not sys.stdin.isatty():
        logger.info("Running in non-interactive mode, proceeding with migration...")
    else:
        response = input("\nProceed with migration? [y/N]: ")
        if response.lower() != 'y':
            logger.info("Migration cancelled.")
            return
    
    # Execute migrations
    for src, dst in migrations:
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.move(str(src), str(dst))
            else:
                shutil.move(str(src), str(dst))
            logger.info(f"Moved: {src} → {dst}")
        except Exception as e:
            logger.error(f"Failed to move {src}: {e}")
    
    # Clean up empty directories
    for dir_path in [src_rpm / "docs", src_rpm / "results"]:
        if dir_path.exists() and not any(dir_path.iterdir()):
            dir_path.rmdir()
            logger.info(f"Removed empty directory: {dir_path}")
    
    logger.info("Migration complete!")

if __name__ == "__main__":
    migrate_rpm_docs()
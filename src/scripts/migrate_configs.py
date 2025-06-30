#!/usr/bin/env python3
"""Migrate configuration files from src to config directory."""

from pathlib import Path
import shutil
import logging
import filecmp

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_config_files():
    """Find all configuration files in src."""
    config_files = []
    
    # Find YAML and JSON config files
    for pattern in ["*.yaml", "*.yml", "*.json"]:
        for file_path in Path("src").rglob(pattern):
            # Skip __pycache__ and test files
            if "__pycache__" in str(file_path) or "test" in file_path.name:
                continue
            
            # Skip non-config JSON files
            if file_path.suffix == ".json" and "report" in file_path.name:
                continue
                
            config_files.append(file_path)
    
    return sorted(config_files)

def determine_destination(file_path: Path) -> Path:
    """Determine the correct destination for a config file."""
    file_name = file_path.name
    
    # Processing configs
    if file_name in ["alignment_config.yaml", "orientation_config.yaml", 
                     "rpm_config.yaml", "timestamp_config.yaml"]:
        return Path("config/processing") / file_name
    
    # Sensor configs
    elif "sensor" in file_name.lower():
        return Path("config/sensors") / file_name
    
    # Experiment configs
    elif "experiment" in file_name.lower() or "manifest" in file_name.lower():
        return Path("config/experiments") / file_name
    
    # Reports and summaries (not really configs)
    elif "report" in file_name.lower() or "summary" in file_name.lower():
        return None  # Don't migrate these
    
    # Default to config root
    else:
        return Path("config") / file_name

def main():
    """Migrate all configuration files."""
    
    config_files = find_config_files()
    
    if not config_files:
        logger.info("No configuration files found to migrate.")
        return
    
    logger.info(f"Found {len(config_files)} potential config files:")
    
    migrations = []
    duplicates = []
    skipped = []
    
    for src_path in config_files:
        dst_path = determine_destination(src_path)
        
        if dst_path is None:
            logger.info(f"  SKIP: {src_path} (not a config file)")
            skipped.append(src_path)
            continue
        
        # Check if destination already exists
        if dst_path.exists():
            # Check if files are identical
            if filecmp.cmp(str(src_path), str(dst_path), shallow=False):
                logger.info(f"  DUPLICATE: {src_path} → {dst_path} (identical)")
                duplicates.append(src_path)
            else:
                logger.warning(f"  CONFLICT: {src_path} → {dst_path} (files differ!)")
                migrations.append((src_path, dst_path, True))  # Mark as conflict
        else:
            logger.info(f"  MIGRATE: {src_path} → {dst_path}")
            migrations.append((src_path, dst_path, False))
    
    # Summary
    logger.info(f"\nSummary:")
    logger.info(f"  - {len(duplicates)} duplicate files (can be removed)")
    logger.info(f"  - {len([m for m in migrations if not m[2]])} files to migrate")
    logger.info(f"  - {len([m for m in migrations if m[2]])} conflicts to resolve")
    logger.info(f"  - {len(skipped)} files skipped")
    
    if duplicates:
        logger.info("\nDuplicate files to remove:")
        for dup in duplicates:
            logger.info(f"  rm {dup}")
    
    if migrations:
        conflicts = [m for m in migrations if m[2]]
        if conflicts:
            logger.warning("\nCONFLICTS found! Please resolve manually:")
            for src, dst, _ in conflicts:
                logger.warning(f"  {src} differs from {dst}")
            return
    
    # Ask for confirmation
    import sys
    if duplicates or migrations:
        if not sys.stdin.isatty():
            logger.info("\nRunning in non-interactive mode, proceeding...")
            proceed = True
        else:
            response = input("\nProceed with cleanup? [y/N]: ")
            proceed = response.lower() == 'y'
        
        if proceed:
            # Remove duplicates
            for dup_path in duplicates:
                dup_path.unlink()
                logger.info(f"Removed duplicate: {dup_path}")
            
            # Perform migrations
            for src_path, dst_path, is_conflict in migrations:
                if not is_conflict:
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src_path), str(dst_path))
                    logger.info(f"Moved: {src_path} → {dst_path}")
            
            logger.info("\nConfig cleanup complete!")
            logger.info("\nDon't forget to update any imports that reference these config files!")

if __name__ == "__main__":
    main()
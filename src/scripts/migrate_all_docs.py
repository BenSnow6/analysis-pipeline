#!/usr/bin/env python3
"""Migrate ALL documentation and images from src to docs."""

from pathlib import Path
import shutil
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def migrate_module_docs(module_name: str):
    """Migrate docs and images from a specific analysis module."""
    
    src_module = Path(f"src/analysis/{module_name}")
    docs_results = Path(f"docs/results/{module_name}")
    docs_dev = Path(f"docs/development/{module_name}")
    
    # Ensure destination directories exist
    docs_results.mkdir(parents=True, exist_ok=True)
    docs_dev.mkdir(parents=True, exist_ok=True)
    
    migrations = []
    
    # Find all documentation and image files
    doc_patterns = ["*.md", "*.png", "*.jpg", "*.jpeg", "*.gif", "*.svg"]
    
    for pattern in doc_patterns:
        for file_path in src_module.glob(pattern):
            # Skip __pycache__ directories
            if "__pycache__" in str(file_path):
                continue
                
            # Determine destination based on file name
            if file_path.name == "README.md":
                dst_path = docs_results / file_path.name
            elif "checklist" in file_path.name.lower() or "development" in file_path.name.lower():
                dst_path = docs_dev / file_path.name
            elif file_path.suffix in ['.png', '.jpg', '.jpeg', '.gif', '.svg']:
                # Images go to results
                dst_path = docs_results / file_path.name
            else:
                # Other markdown files go to results
                dst_path = docs_results / file_path.name
            
            migrations.append((file_path, dst_path))
    
    return migrations

def main():
    """Migrate all documentation from src to docs."""
    
    # Analysis modules to check
    modules = ["alignment", "orientation", "timestamp", "rpm"]
    
    all_migrations = []
    
    for module in modules:
        module_path = Path(f"src/analysis/{module}")
        if module_path.exists():
            migrations = migrate_module_docs(module)
            if migrations:
                logger.info(f"\nFound {len(migrations)} files to migrate in {module}:")
                for src, dst in migrations:
                    logger.info(f"  {src.name} → {dst.relative_to('docs')}")
                all_migrations.extend(migrations)
    
    if not all_migrations:
        logger.info("No documentation files found to migrate.")
        return
    
    logger.info(f"\nTotal files to migrate: {len(all_migrations)}")
    
    # Auto-proceed in non-interactive mode
    import sys
    if not sys.stdin.isatty():
        logger.info("Running in non-interactive mode, proceeding with migration...")
    else:
        response = input("\nProceed with migration? [y/N]: ")
        if response.lower() != 'y':
            logger.info("Migration cancelled.")
            return
    
    # Execute migrations
    for src, dst in all_migrations:
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            
            # Check if destination exists
            if dst.exists():
                backup_path = dst.with_suffix(dst.suffix + '.backup')
                shutil.copy2(str(dst), str(backup_path))
                logger.warning(f"Backed up existing file: {backup_path}")
            
            shutil.move(str(src), str(dst))
            logger.info(f"Moved: {src} → {dst}")
            
        except Exception as e:
            logger.error(f"Failed to move {src}: {e}")
    
    logger.info("\nMigration complete!")
    logger.info("\nDon't forget to:")
    logger.info("1. Update any references to these files in the code")
    logger.info("2. Commit the changes")
    logger.info("3. Update CLAUDE.md if needed")

if __name__ == "__main__":
    main()
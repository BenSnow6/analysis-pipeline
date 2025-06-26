# Post-Migration Checklist

This checklist ensures the repository restructuring has been completed successfully.

## Pre-Migration Backup
- [ ] Created git branch for backup: `git checkout -b structure-migration-backup`
- [ ] Committed current state: `git add -A && git commit -m "Backup before structure migration"`

## Migration Execution

### Phase 1: Infrastructure
- [ ] Created `/src/scripts/migrate_rpm_docs.py`
- [ ] Created `/src/scripts/validate_structure.py`
- [ ] Created `/REPOSITORY_STRUCTURE.md` at root
- [ ] Created `/docs/migration/MIGRATION_CHECKLIST.md` (this file)

### Phase 2: RPM Documentation Consolidation
- [ ] Ran migration script: `python src/scripts/migrate_rpm_docs.py`
- [ ] Verified files moved:
  - [ ] `/src/analysis/rpm/README.md` → `/docs/results/rpm_estimation/README.md`
  - [ ] `/src/analysis/rpm/docs/` → `/docs/results/rpm_estimation/wp*/`
  - [ ] Development docs → `/docs/development/rpm/`
- [ ] Removed empty directories in `/src/analysis/rpm/`
- [ ] No `.md` files remain in `/src/analysis/rpm/`

### Phase 3: Clean Up Old Data Structure
- [ ] Committed deletion of `/02_Evaluation_Experiments/` (5,758 files)
- [ ] Verified git status shows deletions staged
- [ ] No broken symlinks remain

## Post-Migration Validation

### Immediate Checks
- [ ] Run validation script: `python src/scripts/validate_structure.py`
- [ ] All validation checks pass

### Import Verification
- [ ] Python imports work:
  ```python
  from src.analysis.rpm import preprocess, spectral, fusion
  from src.core import io, DATA_DIR
  from src.core.paths import PROCESSED_DATA_DIR
  ```

### Test Suite
- [ ] Run all tests: `pytest tests/`
- [ ] Key tests pass:
  - [ ] Alignment tests
  - [ ] Orientation tests  
  - [ ] RPM tests
  - [ ] Core utility tests

### Application Testing
- [ ] Dashboard loads: `python src/scripts/dashboard_app.py`
- [ ] No import errors in dashboard
- [ ] Can load experiment data through dashboard

### Documentation
- [ ] All RPM docs accessible in `/docs/results/rpm_estimation/`
- [ ] Work package structure preserved
- [ ] No duplicate documentation

## Update References

### Code Updates
- [ ] Search for old paths in Python files: `grep -r "02_Evaluation_Experiments" src/`
- [ ] Update any hardcoded paths to use path helpers
- [ ] Update any references to old doc locations

### Config Updates
- [ ] Check `/config/` files for old path references
- [ ] Update `pipeline.yaml` if needed
- [ ] Verify experiment mappings still correct

### Documentation Updates
- [ ] Update READMEs that reference old structure
- [ ] Update any notebooks with old paths
- [ ] Update CLAUDE.md if structure details changed

## Git Finalization

### Commit Strategy
- [ ] Commit deletions separately:
  ```bash
  git add -A "02_Evaluation_Experiments"
  git commit -m "refactor: Remove old data structure (moved to /data/raw/)"
  ```
- [ ] Commit documentation consolidation:
  ```bash
  git add -A src/analysis/rpm/ docs/results/rpm_estimation/
  git commit -m "refactor: Consolidate RPM documentation in /docs/results/"
  ```
- [ ] Commit infrastructure files:
  ```bash
  git add REPOSITORY_STRUCTURE.md src/scripts/migrate_*.py docs/migration/
  git commit -m "docs: Add migration scripts and repository structure guide"
  ```

### Tagging
- [ ] Create post-migration tag:
  ```bash
  git tag -a post-migration-v1.0 -m "Repository structure reorganized"
  ```

## Final Verification
- [ ] Repository follows structure in `REPOSITORY_STRUCTURE.md`
- [ ] Can perform a fresh clone and run all tests
- [ ] No broken functionality reported
- [ ] Team members notified of structure changes

## Rollback Plan
If issues are found:
1. `git checkout structure-migration-backup`
2. Investigate issues
3. Fix migration scripts
4. Re-attempt migration

## Notes
- Migration completed on: [DATE]
- Issues encountered: [LIST ANY ISSUES]
- Additional changes made: [LIST ANY EXTRA CHANGES]
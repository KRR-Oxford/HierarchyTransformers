# Changelog :newspaper:

<!-- Added for new features.
Changed for changes in existing functionality.
Deprecated for soon-to-be removed features.
Removed for now removed features.
Fixed for any bug fixes.
Security in case of vulnerabilities. -->

## Unreleased

Significant development to align with `sentence-transformers>=3.0.0`.

### Added

- [ ] Add `hierarchy_transformers.datasets` to supercede `hierarchy_transformers.utils.data` and `h ierarchy_transformers.utils.construct`.
- [ ] Upload HiT datasets to HuggingFace.
- [X] Add pytest modules for testing.
- [X] Project page at https://krr-oxford.github.io/HierarchyTransformers/
- [X] Model versioning (add random and hard negatives as a choice of `revision`) on HuggingFace.
- [X] Add `hierarchy_transformers.models.arithmetic` for projection functions.

### Changed

- [X] Move several utility functions to `hierarchy_transformers.models`.
- [X] Rewrite the `hierarchy_transformers.models.hit` to align with `sentence-transformers>=3.0.0`.

### Removed

- [X] Remove `hierarchy_transformers.models.utils`.

## v0.0.3 (2024-05-09)

Initial release (should work with `sentence-transformers<3.0.0` ) and bug fix. 

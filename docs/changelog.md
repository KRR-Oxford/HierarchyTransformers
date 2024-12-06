# Changelog :newspaper:

<!-- Added for new features.
Changed for changes in existing functionality.
Deprecated for soon-to-be removed features.
Removed for now removed features.
Fixed for any bug fixes.
Security in case of vulnerabilities. -->

## Unreleased

Significant development to align with `sentence-transformers>=3.4.0`.

### Added

- [X] [Feature] Add pytest modules for testing.
- [X] [Web] Set up [project page](https://krr-oxford.github.io/HierarchyTransformers/).
- [X] [Web] Upload HiT datasets on [HuggingFace](https://huggingface.co/Hierarchy-Transformers).
- [X] [Web] Re-organise models by setting `v1-random-negatives` and `v1-hard-negatives` revisions on [HuggingFace](https://huggingface.co/Hierarchy-Transformers).

### Changed

- [X] [Feature,Layout] Rewrite and reorganise `hierarchy_transformers.models`, `hierarchy_transformers.losses`, and `hierarchy_transformers.evaluation` to to align with `sentence-transformers>=3.4.0.dev`.
- [X] [Feature,Layout] Rewrite dataset processing and loading functions and reorganise everything into `hierarchy_transformers.datasets`.

### Removed

- [X] [Layout] Remove `hierarchy_transformers.models.utils`.

## v0.0.3 (2024-05-09)

Initial release (should work with `sentence-transformers<3.0.0` ) and bug fix. 

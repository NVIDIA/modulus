<!-- markdownlint-disable MD024 -->
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [unreleased]

### Added

- Added a CHANGELOG.md
- Added build support for internal DGL
- 4D Fourier Neural Operator model
- Ahmed body dataset

### Changed

- DGL install changed from pypi to source
- Updated SFNO to add support for super resolution, flexible checkpoining, etc.

### Deprecated

### Removed

### Fixed

- Fixed issue with torch-harmonics version locking
- Fixed the Modulus editable install
- Fixed bug in AMP in static capture

### Security

- Fixed security issues with subprocess and urllib in `filesystem.py`

### Dependencies

- Updated the base container to latest PyTorch base container which is based on torch 2.0
- Container now supports CUDA 12, Python 3.10

## [0.1.0] - 2023-05-08

### Added

- Initial public release.

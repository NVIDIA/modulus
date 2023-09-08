<!-- markdownlint-disable MD024 -->
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0a0]

### Added

- Added distributed utilities to create process groups and orthogonal process groups.
- Added distributed AFNO model implementation.
- Added distributed utilities for communication of buffers of varying size per rank.
- Added distributed utilities for message passing across multiple GPUs.

### Changed

- Updating file system cache location to modulus folder

### Deprecated

### Removed

### Fixed

- Fixed modulus uninstall in CI docker image

### Security

### Dependencies

- Updated the base container to latest PyTorch 23.07.
- Update DGL version.

## [0.2.1] - 2023-08-08

### Added

### Changed

### Deprecated

### Removed

### Fixed

- Added a workaround fix for the CUDA graphs error in multi-node runs

### Security

- Update `certifi` package version

### Dependencies

## [0.2.0] - 2023-08-01

### Added

- Added a CHANGELOG.md
- Added build support for internal DGL
- 4D Fourier Neural Operator model
- Ahmed body dataset
- Unified Climate Datapipe

### Changed

- DGL install changed from pypi to source
- Updated SFNO to add support for super resolution, flexible checkpoining, etc.

### Deprecated

### Removed

### Fixed

- Fixed issue with torch-harmonics version locking
- Fixed the Modulus editable install
- Fixed AMP bug in static capture

### Security

- Fixed security issues with subprocess and urllib in `filesystem.py`

### Dependencies

- Updated the base container to latest PyTorch base container which is based on torch 2.0
- Container now supports CUDA 12, Python 3.10

## [0.1.0] - 2023-05-08

### Added

- Initial public release.

<!-- markdownlint-disable MD024 -->
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0a0] - 2023-11-xx

### Added

### Changed

### Deprecated

### Removed

### Fixed

### Security

### Dependencies

## [0.3.0] - 2023-09-21

### Added

- Added distributed utilities to create process groups and orthogonal process groups.
- Added distributed AFNO model implementation.
- Added distributed utilities for communication of buffers of varying size per rank.
- Added instructions for docker build on ARM architecture.
- Added batching support and fix the input time step for the DLWP wrapper.

### Changed

- Updating file system cache location to modulus folder

### Fixed

- Fixed modulus uninstall in CI docker image

### Security

- Handle the tar ball extracts in a safer way.

### Dependencies

- Updated the base container to latest PyTorch 23.07.
- Update DGL version.
- Updated require installs for python wheel
- Added optional dependency list for python wheel

## [0.2.1] - 2023-08-08

### Fixed

- Added a workaround fix for the CUDA graphs error in multi-node runs

### Security

- Update `certifi` package version

## [0.2.0] - 2023-08-07

### Added

- Added a CHANGELOG.md
- Added build support for internal DGL
- 4D Fourier Neural Operator model
- Ahmed body dataset
- Unified Climate Datapipe

### Changed

- DGL install changed from pypi to source
- Updated SFNO to add support for super resolution, flexible checkpoining, etc.

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

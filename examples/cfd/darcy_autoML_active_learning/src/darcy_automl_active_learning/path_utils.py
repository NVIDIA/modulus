# File: src/path_utils.py

import os
from pathlib import Path
from typing import Optional
import logging
from .env_utils import is_running_in_docker

import logging
logger = logging.getLogger(__name__)

def get_absolute_path(base_path: str, subpath: Optional[str] = None) -> str:
    """
    Given a base path and an optional subpath/filename, returns an absolute path.

    If base_path is already absolute, we trust it.
    If it's relative, we resolve it relative to the project root or cwd 
    (you can define the policy below).
    
    Args:
        base_path (str):
            The root directory or base path from your config,
            e.g. "/workspace/examples/cfd/darcy_autoML_active_learning/data"
        subpath (str, optional):
            A sub-directory or filename to join onto base_path.
            e.g. "00_Generate_Data/data_desc.json"

    Returns:
        Absolute path (str).
    """
    # Convert to a Path object
    base = Path(base_path)

    # If subpath is provided, join it
    if subpath:
        full_path = base / subpath
    else:
        full_path = base

    # Now resolve to absolute
    # If the user intentionally put an absolute path in the config,
    # Path(...) / subpath will remain absolute. This .resolve() 
    # ensures we remove any "." or "..".
    return str(full_path.resolve())


import os
from pathlib import Path

def get_repo_root():
    """
    Determines the repository root directory based on the environment.

    Priority:
      1) PROJECT_ROOT environment variable (if valid directory).
         - If the path starts with '/root', remove the '/root' prefix.
      2) Fallback based on current working directory (no longer going up one level).

    Returns:
        Path: The path to the repository root.
    """
    repo_root_env = os.environ.get("PROJECT_ROOT")  # e.g., "/root/project/modulus-dls-api" in Docker
    if repo_root_env:
        repo_root = Path(repo_root_env).resolve()
        if repo_root.is_absolute() and len(repo_root.parts) > 1 and repo_root.parts[1] == "root":
            # Reconstruct the path without '/root'
            adjusted_repo_root = Path(*repo_root.parts[2:]).resolve()
            # Prepend '/' to keep it absolute
            adjusted_repo_root = Path("/").joinpath(adjusted_repo_root)
            repo_root = adjusted_repo_root
        return repo_root
    else:
        current_path = Path.cwd().resolve()
        # Use the current path itself, rather than going up a level
        repo_root = current_path
        return repo_root


def get_required_paths(repo_root: Path):
    """
    Generates all required paths based on the repository root.

    Args:
        repo_root (Path): The path to the repository root.

    Returns:
        Paths: An instance of the Paths data class containing all required paths.
    """
    darcy_project_root = repo_root / "examples" / "cfd" / "darcy_autoML_active_learning"
    config_file = darcy_project_root / "config" / "config.yaml"
    data_dir = darcy_project_root / "data"
    results_dir = darcy_project_root / "results"

    # Optional: Validate paths
    if not darcy_project_root.is_dir():
        logging.warning(f"Darcy project root '{darcy_project_root}' does not exist.")
    if not config_file.is_file():
        logging.warning(f"Config file '{config_file}' does not exist.")
    if not data_dir.is_dir():
        logging.warning(f"Data directory '{data_dir}' does not exist.")
    if not results_dir.is_dir():
        logging.warning(f"Results directory '{results_dir}' does not exist.")

    return repo_root, darcy_project_root, config_file, data_dir, results_dir

def identify_scenario():
    """
    Determines which scenario we are in, based on:
      - Are we in Docker? (is_running_in_docker())
      - Is PROJECT_ROOT set?

    For now, we ONLY handle Scenario A1:
      - A1 = Docker, workspace = `modulus-dls-api/`, PROJECT_ROOT is set.

    Any other scenario raises NotImplementedError.
    """
    # Check if we're in Docker
    in_docker = is_running_in_docker()

    # Check for PROJECT_ROOT
    project_root_env = os.environ.get("PROJECT_ROOT", None)

    if in_docker:
        if project_root_env:
            return "A1"
        else:
            raise NotImplementedError("Docker scenario with no PROJECT_ROOT is not yet implemented.")
    else:
        raise NotImplementedError("Local (non-Docker) scenario is not yet implemented.")

def get_paths_for_A1():
    """
    Scenario A1: Docker, workspace = `modulus-dls-api/`, PROJECT_ROOT is set.

    For simplicity, we assume:
      - The user wants 'repo_root' to be '.'
      - All subpaths are relative from '.'

    Returns:
      A tuple: (repo_root, darcy_project_root, config_file, data_dir, results_dir)
    """
    repo_root = Path(".")

    darcy_project_root = repo_root / "examples" / "cfd" / "darcy_autoML_active_learning"
    config_file = darcy_project_root / "config" / "config.yaml"
    data_dir = darcy_project_root / "data"
    results_dir = darcy_project_root / "results"

    # Log all these paths at DEBUG level
    logger.debug(f"Scenario A1 - repo_root: {repo_root}")
    logger.debug(f"Scenario A1 - darcy_project_root: {darcy_project_root}")
    logger.debug(f"Scenario A1 - config_file: {config_file}")
    logger.debug(f"Scenario A1 - data_dir: {data_dir}")
    logger.debug(f"Scenario A1 - results_dir: {results_dir}")

    return (repo_root, darcy_project_root, config_file, data_dir, results_dir)

def get_paths():
    """
    Public entry point for obtaining all required paths.

    1. Identify scenario (A1, A2, B1, etc.)
    2. Delegate to the appropriate function.

    For now, we only handle A1.
    """
    scenario = identify_scenario()
    
    if scenario == "A1":
        return get_paths_for_A1()
    else:
        raise NotImplementedError(f"Scenario '{scenario}' is not yet implemented.")

if __name__ == "__main__":
    # Set up logging to show DEBUG messages for this module.
    logging.basicConfig(level=logging.DEBUG)

    logger.info("Running path_utils as a script. Logging level set to DEBUG for demonstration.")

    # For demonstration, let's call get_paths() directly
    try:
        paths = get_paths()
        logger.info("Paths returned successfully:")
        for p in paths:
            logger.info(f"  {p}")
    except NotImplementedError as exc:
        logger.error(f"Not implemented: {exc}")
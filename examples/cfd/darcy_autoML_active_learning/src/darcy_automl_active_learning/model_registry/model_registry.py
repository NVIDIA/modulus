# File: src/darcy_automl_active_learning/model_registry/model_registry.py

import os
import json
from typing import Dict, Any, Optional, List

class ModelRegistry:
    """
    The ModelRegistry manages a collection of model descriptors,
    each specifying important metadata (base_class, accepted_formats,
    default_hyperparams, HPC constraints, etc.).
    
    If no 'descriptors_file' is specified, the registry automatically
    scans a default folder (DEFAULT_DESCRIPTOR_DIR) for .json files,
    each expected to contain exactly one model descriptor. The
    loaded descriptors can then be queried or retrieved by name.
    """

    DEFAULT_DESCRIPTOR_DIR = os.path.join(
        os.path.dirname(__file__),  # current folder: model_registry
        "descriptors"               # subfolder
    )

    def __init__(self, descriptors_file: Optional[str] = None):
        """
        If descriptors_file is provided, loads that single JSON file. 
        Otherwise, scans DEFAULT_DESCRIPTOR_DIR for all *.json files 
        and loads each as a model descriptor.

        :param descriptors_file: Path to a single JSON file containing model descriptor(s), 
                                 or None to load from /model_registry/descriptors/ by default.
        """
        self._descriptors: Dict[str, Dict[str, Any]] = {}

        if descriptors_file:
            # If user explicitly provided a single file
            self.load_descriptors(descriptors_file)
        else:
            # Otherwise, load from the default descriptors folder
            if os.path.isdir(self.DEFAULT_DESCRIPTOR_DIR):
                self.load_all_descriptors_in_folder(self.DEFAULT_DESCRIPTOR_DIR)
            else:
                # Optionally raise or just warn
                print(f"[ModelRegistry] WARNING: No descriptors_file provided and default folder "
                      f"'{self.DEFAULT_DESCRIPTOR_DIR}' not found. No models loaded.")

    def load_descriptors(self, file_path: str) -> None:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"[ModelRegistry] No file found at {file_path}")
        
        # *** Add encoding="utf-8" here ***
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "model_name" in data:
            model_key = data["model_name"]
            self._descriptors[model_key] = data
            print(f"[ModelRegistry] Loaded descriptor for '{model_key}' from {file_path}")
        else:
            raise ValueError(f"[ModelRegistry] JSON at {file_path} missing 'model_name'.")

    def load_all_descriptors_in_folder(self, folder_path: str) -> None:
        """
        Scans a folder for .json files, calling load_descriptors(...) on each.
        
        :param folder_path: path to a directory containing .json descriptor files
        """
        json_files = [
            f for f in os.listdir(folder_path) 
            if f.endswith(".json") and os.path.isfile(os.path.join(folder_path, f))
        ]

        if not json_files:
            print(f"[ModelRegistry] WARNING: No .json files found in {folder_path}. "
                  "No models loaded.")
            return

        for json_fname in sorted(json_files):
            file_path = os.path.join(folder_path, json_fname)
            try:
                self.load_descriptors(file_path)
            except Exception as e:
                # Optionally continue loading others, or re-raise.
                print(f"[ModelRegistry] ERROR: Could not load {json_fname}: {e}")

    def get_descriptor(self, model_name: str) -> Dict[str, Any]:
        """
        Retrieve the descriptor dictionary for a given model name.
        
        :param model_name: The model name key (e.g., "FNO", "AFNO", "DiffusionNet").
        :return: The descriptor dictionary (e.g., with keys: "description", "base_class", etc.).
        :raises KeyError: If the model_name is not found in the registry.
        """
        if model_name not in self._descriptors:
            raise KeyError(f"[ModelRegistry] Model '{model_name}' not found in registry.")
        return self._descriptors[model_name]

    def get_all_descriptors(self) -> Dict[str, Dict[str, Any]]:
        """
        Return a dictionary of all loaded model descriptors.
        Keys are model names, values are the descriptor dictionaries.

        :return: { "FNO": {...}, "AFNO": {...}, ... }
        """
        return self._descriptors

    def register_model_descriptor(self, descriptor: Dict[str, Any]) -> None:
        """
        Allows adding a new model descriptor at runtime.

        :param descriptor: A dictionary with at least a "model_name" key.
        """
        if "model_name" not in descriptor:
            raise ValueError("[ModelRegistry] Descriptor must contain 'model_name' field.")
        model_key = descriptor["model_name"]
        self._descriptors[model_key] = descriptor
        print(f"[ModelRegistry] Registered new descriptor for model '{model_key}'.")

    def model_exists(self, model_name: str) -> bool:
        """
        Quick check to see if a given model_name is in the registry.

        :param model_name: e.g., "FNO", "AFNO", "GraphCast", etc.
        :return: True if the registry contains it, else False.
        """
        return model_name in self._descriptors

    def list_models(self) -> List[str]:
        """
        Return a list of all model names in the registry.
        """
        return list(self._descriptors.keys())

    def load_descriptors(self, file_path: str) -> None:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"[ModelRegistry] No file found at {file_path}")
        
        # *** Add encoding="utf-8" here ***
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "model_name" in data:
            model_key = data["model_name"]
            self._descriptors[model_key] = data
            print(f"[ModelRegistry] Loaded descriptor for '{model_key}' from {file_path}")
        else:
            raise ValueError(f"[ModelRegistry] JSON at {file_path} missing 'model_name'.")

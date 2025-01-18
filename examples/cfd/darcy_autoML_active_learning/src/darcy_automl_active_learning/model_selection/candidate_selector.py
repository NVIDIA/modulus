# src/darcy_automl_active_learning/model_selection/candidate_selector.py

import os
import json
from typing import List, Tuple
from darcy_automl_active_learning.model_registry.model_registry import ModelRegistry
from .selection_strategies import BaseSelectionStrategy

class CandidateModelSelector:
    """
    CandidateModelSelector orchestrates:
      1) Validation of a data descriptor
      2) AutoML-based candidate model selection (per chosen strategy)
      3) Retrieval of required data structures for each model
    """

    def __init__(self, model_registry: ModelRegistry, selection_strategy: BaseSelectionStrategy):
        """
        Args:
            model_registry : Instance of ModelRegistry
            selection_strategy : A strategy implementing how to pick candidate models
        """
        self.model_registry = model_registry
        self.selection_strategy = selection_strategy

    def validate_data_descriptor(self, data_desc_path: str) -> bool:
        """
        Basic placeholder for data descriptor validation.
        In a real scenario, you'd parse the JSON, check required fields, etc.
        
        Returns:
            bool : True if descriptor is valid, else False
        """
        if not os.path.isfile(data_desc_path):
            print(f"[CandidateModelSelector] Data descriptor not found: {data_desc_path}")
            return False

        with open(data_desc_path, "r") as f:
            data_desc = json.load(f)

        # Example logic: must have "data_structure" key
        if "data_structure" not in data_desc:
            print("[CandidateModelSelector] 'data_structure' key missing in descriptor.")
            return False

        # Additional checks if desired...
        return True

    def automl_candidate_model_selection(self, data_desc_path: str) -> List[Tuple[str, str]]:
        """
        Invokes the selection strategy to pick suitable models.

        Args:
            data_desc_path (str): path to the data descriptor JSON

        Returns:
            List[Tuple[str, str]]: e.g. [("FNO", "candidate0"), ("AFNO", "candidate1")]
        """
        with open(data_desc_path, "r") as f:
            data_desc = json.load(f)

        # Let the strategy do the work
        selected_candidates = self.selection_strategy.select_candidates(
            data_desc=data_desc,
            model_registry=self.model_registry
        )

        return selected_candidates

    def get_required_data_structure(self, model_name: str):
        """
        Retrieve the data structure requirements from the model descriptor
        (the 'accepted_formats' or something similar).

        Args:
            model_name (str): Name of the model, must exist in the registry

        Returns:
            dict or list: The portion of the descriptor describing required input format
        """
        descriptor = self.model_registry.get_descriptor(model_name)

        # Typically, you'd parse descriptor["accepted_formats"] or similar
        return descriptor.get("accepted_formats", [])

    def save_candidate_models(self, selected_candidates: List[Tuple[str, str]], output_folder: str) -> str:
        """
        Saves the chosen models (along with their candidate keys) to a JSON file.
        
        Args:
            selected_candidates: e.g. [("FNO","candidate0"), ("AFNO","candidate1")]
            output_folder: location to store the JSON file

        Returns:
            str : path to the saved JSON
        """
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, "chosen_candidates.json")

        # Convert to a JSON-friendly structure
        # e.g. [ ["FNO","candidate0"], ["DiffusionNet","candidate1"] ]
        with open(output_path, "w") as f:
            json.dump(selected_candidates, f, indent=2)

        print(f"[CandidateModelSelector] Saved {len(selected_candidates)} candidates to: {output_path}")
        return output_path
    
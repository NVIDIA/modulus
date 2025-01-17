# File: src/AutoMLCandidateModelSelection.py

import json
import os
from typing import List, Tuple, Dict, Any

from .data_desc_logic import load_data_descriptor, check_data_model_compatibility


def automl_candidate_model_selection(
    data_desc_path: str,
    model_descriptors: List[Dict[str, Any]],
) -> List[Tuple[str, str]]:
    """
    Loads a PDE dataset descriptor from disk, then checks each model descriptor
    against that dataset for compatibility. If compatible, the model is added
    to a list of candidate models.

    Args:
        data_desc_path (str):
            Path to the JSON file describing the PDE dataset
            (e.g., dimension, geometry_type, uniform, channels, etc.).
        model_descriptors (List[Dict[str, Any]]):
            A list of dictionaries, each describing a model's accepted_formats.
            Example model descriptor structure:
                {
                  "model_name": "FNO",
                  "accepted_formats": [
                    {
                      "dimension": [2, 3],
                      "geometry_type": "grid",
                      "uniform": True,
                      "channels_min": 1
                    },
                    ...
                  ]
                }

    Returns:
        List[Tuple[str, str]]:
            A list of (model_name, candidate_id) tuples for each compatible model.
            For example: [("FNO", "candidate0"), ("AFNO", "candidate1")].
    """
    # 1) Load the dataset descriptor
    data_desc = load_data_descriptor(data_desc_path)

    # 2) We'll store chosen models in a list
    chosen_candidates = []
    candidate_counter = 0

    for model_desc in model_descriptors:
        model_name = model_desc.get("model_name", "UnknownModel")

        # 3) Check compatibility with the PDE data
        is_compatible = check_data_model_compatibility(data_desc, model_desc)

        if is_compatible:
            cand_id = f"candidate{candidate_counter}"
            chosen_candidates.append((model_name, cand_id))
            candidate_counter += 1

    return chosen_candidates


def save_candidate_models(
    candidates: List[Tuple[str, str]],
    output_folder: str,
    filename: str = "candidate_models.json",
) -> str:
    """
    Saves the selected candidate models to a JSON file.

    Args:
        candidates (List[Tuple[str, str]]):
            A list of (model_name, candidate_id) pairs produced by
            automl_candidate_model_selection(...).
        output_folder (str):
            Path to the folder where the JSON file should be written.
        filename (str, optional):
            Name of the output JSON file. Default is "candidate_models.json".

    Returns:
        str:
            The full path to the JSON file that was written.
    """
    # 1) Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # 2) Construct the output file path
    json_path = os.path.join(output_folder, filename)

    # 3) Save the candidates list as JSON
    with open(json_path, "w") as f:
        json.dump(candidates, f, indent=2)

    return json_path

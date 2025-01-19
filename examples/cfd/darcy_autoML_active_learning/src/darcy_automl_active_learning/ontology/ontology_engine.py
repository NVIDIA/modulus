import os
from typing import Dict, Any

class OntologyEngine:
    """
    A mock 'OntologyEngine' that returns hard-coded transformation plans
    for two candidates: 'candidate0' (FNOWithDropout) and 'candidate1' (AFNO).

    We explicitly embed `data_dir_path` inside each transform operation's params,
    along with optional subfolder_source/dest keys to differentiate candidate0/candidate1.
    """

    def __init__(self):
        pass

    def suggest_transformations(
        self,
        source_data_desc: Dict[str, Any],
        target_data_requirements: Dict[str, Any],
        data_dir_path: str = "data"
    ) -> Dict[str, Any]:
        """
        Returns a dictionary of transformation plans for two candidates: candidate0 and candidate1.

        Args:
            source_data_desc:         E.g., data_desc["data_structure"] from the PDE descriptor
            target_data_requirements: E.g., from candidate_selector.get_required_data_structure(model_name)
            data_dir_path:            Base path to your 'data' folder (PosixPath or string).

        Returns:
            A Python dict representing transformation plans for both candidates. 
            Each transform_ops entry includes `data_dir_path` in its params, and 
            subfolder_source/dest for the relevant stages.
        """

        # Convert data_dir_path to a string to avoid JSON serialization errors
        data_dir_path = str(data_dir_path)

        # --------------------------------------------------
        # Candidate0 => FNOWithDropout
        # --------------------------------------------------
        plan_for_fno = {
            "model_name": "FNOWithDropout",
            "stages": [
                {
                    # Stage 1: no subfolders, shared raw data
                    "stage_name": "01_01_01_LoadRawData",
                    "transform_ops": [
                        {
                            "method": "copy_only",
                            "params": {
                                "source_folder": "00_Generate_Data",
                                "dest_folder": "01_01_LoadRawData",
                                "data_dir_path": data_dir_path
                            }
                        }
                    ]
                },
                {
                    # Stage 3: subfolders for candidate0
                    "stage_name": "01_01_03_TransformRawData",
                    "transform_ops": [
                        {
                            "method": "copy_only",
                            "params": {
                                "source_folder": "01_01_LoadRawData",
                                "dest_folder": "01_01_03_TransformRawData",
                                "subfolder_source": "candidate0",
                                "subfolder_dest": "candidate0",
                                "data_dir_path": data_dir_path
                            }
                        }
                    ]
                },
                {
                    # Stage 4: subfolders for candidate0
                    "stage_name": "01_01_04_Preprocessing",
                    "transform_ops": [
                        {
                            "method": "copy_only",
                            "params": {
                                "source_folder": "01_01_03_TransformRawData",
                                "dest_folder": "01_01_04_Preprocessing",
                                "subfolder_source": "candidate0",
                                "subfolder_dest": "candidate0",
                                "data_dir_path": data_dir_path
                            }
                        }
                    ]
                },
                {
                    # Stage 5: subfolders for candidate0
                    "stage_name": "01_01_05_FeaturePreparation",
                    "transform_ops": [
                        {
                            "method": "copy_only",
                            "params": {
                                "source_folder": "01_01_04_Preprocessing",
                                "dest_folder": "01_01_05_FeaturePreparation",
                                "subfolder_source": "candidate0",
                                "subfolder_dest": "candidate0",
                                "data_dir_path": data_dir_path
                            }
                        }
                    ]
                }
            ]
        }

        # --------------------------------------------------
        # Candidate1 => AFNO
        # --------------------------------------------------
        plan_for_afno = {
            "model_name": "AFNO",
            "stages": [
                {
                    # Stage 1: no subfolders, shared raw data
                    "stage_name": "01_01_01_LoadRawData",
                    "transform_ops": [
                        {
                            "method": "copy_only",
                            "params": {
                                "source_folder": "00_Generate_Data",
                                "dest_folder": "01_01_LoadRawData",
                                "data_dir_path": data_dir_path
                            }
                        }
                    ]
                },
                {
                    # Stage 3: subfolders for candidate1
                    "stage_name": "01_01_03_TransformRawData",
                    "transform_ops": [
                        {
                            "method": "copy_only",
                            "params": {
                                "source_folder": "01_01_LoadRawData",
                                "dest_folder": "01_01_03_TransformRawData",
                                "subfolder_source": "candidate1",
                                "subfolder_dest": "candidate1",
                                "data_dir_path": data_dir_path
                            }
                        }
                    ]
                },
                {
                    # Stage 4: subfolders for candidate1
                    "stage_name": "01_01_04_Preprocessing",
                    "transform_ops": [
                        {
                            "method": "copy_only",
                            "params": {
                                "source_folder": "01_01_03_TransformRawData",
                                "dest_folder": "01_01_04_Preprocessing",
                                "subfolder_source": "candidate1",
                                "subfolder_dest": "candidate1",
                                "data_dir_path": data_dir_path
                            }
                        }
                    ]
                },
                {
                    # Stage 5: subfolders for candidate1
                    "stage_name": "01_01_05_FeaturePreparation",
                    "transform_ops": [
                        {
                            "method": "copy_only",
                            "params": {
                                "source_folder": "01_01_04_Preprocessing",
                                "dest_folder": "01_01_05_FeaturePreparation",
                                "subfolder_source": "candidate1",
                                "subfolder_dest": "candidate1",
                                "data_dir_path": data_dir_path
                            }
                        }
                    ]
                }
            ]
        }

        # Return both candidate plans
        return {
            "candidate0": {
                "model_name": "FNOWithDropout",
                "plan": plan_for_fno
            },
            "candidate1": {
                "model_name": "AFNO",
                "plan": plan_for_afno
            }
        }

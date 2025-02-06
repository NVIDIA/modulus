import os
from typing import Dict, Any

class OntologyEngine:
    """
    A mock 'OntologyEngine' that returns a transformation plan for exactly one candidate,
    based on model_name ("FNO", "FNOWithDropout", or "AFNO") and candidate_key (e.g. "candidate0").
    
    It embeds data_dir_path in each transform_op's params, along with optional subfolder_source
    and subfolder_dest for stages 3, 4, and 5. If model_name is unrecognized, raises NotImplementedError.
    """

    def __init__(self):
        pass

    def suggest_transformations(
        self,
        source_data_desc: Dict[str, Any],
        target_data_requirements: Dict[str, Any],
        model_name: str,
        candidate_key: str,
        data_dir_path: str = "data"
    ) -> Dict[str, Any]:
        """
        Build and return a transformation plan *for one candidate* (one model_name + candidate_key).

        Args:
            source_data_desc:          The PDE data descriptor (e.g., data_desc["data_structure"]).
            target_data_requirements:  E.g., from candidate_selector.get_required_data_structure(model_name).
            model_name:                "FNO", "FNOWithDropout", or "AFNO".
            candidate_key:             A unique identifier, e.g. "candidate0".
            data_dir_path:             Base directory path (string); embedded in each transform_op.

        Returns:
            A dictionary with shape:
            {
              "model_name": "<chosen model name>",
              "stages": [
                {
                  "stage_name": "01_01_01_LoadRawData",
                  "transform_ops": [
                    {
                      "method": "copy_only",
                      "params": {
                        "source_folder": "...",
                        "dest_folder": "...",
                        "subfolder_source": "...",    (optional)
                        "subfolder_dest": "...",      (optional)
                        "data_dir_path": "..."        (string)
                      }
                    },
                    ...
                  ]
                },
                ...
              ]
            }

        Raises:
            NotImplementedError: If model_name is not one of ("FNO", "FNOWithDropout", "AFNO").
        """

        # Convert to string (avoid JSON serialization problems if it's a Path)
        data_dir_path = str(data_dir_path)

        # Hard-coded plan for "FNO"
        plan_for_fno = {
            "model_name": "FNO",
            "stages": [
                {
                    "stage_name": "01_01_01_LoadRawData",
                    "transform_ops": [
                        {
                            "method": "copy_only",
                            "params": {
                                "source_folder": "00_Generate_Data",
                                "dest_folder": "01_01_LoadRawData",
                                "subfolder_dest": candidate_key,
                                "data_dir_path": data_dir_path
                            }
                        }
                    ]
                },
                {
                    "stage_name": "01_01_03_TransformRawData",
                    "transform_ops": [
                        {
                            "method": "copy_only",
                            "params": {
                                "source_folder": "01_01_LoadRawData",
                                "dest_folder": "01_01_03_TransformRawData",
                                "subfolder_source": candidate_key,
                                "subfolder_dest": candidate_key,
                                "data_dir_path": data_dir_path
                            }
                        }
                    ]
                },
                {
                    "stage_name": "01_01_04_Preprocessing",
                    "transform_ops": [
                        {
                            "method": "copy_only",
                            "params": {
                                "source_folder": "01_01_03_TransformRawData",
                                "dest_folder": "01_01_04_Preprocessing",
                                "subfolder_source": candidate_key,
                                "subfolder_dest": candidate_key,
                                "data_dir_path": data_dir_path
                            }
                        }
                    ]
                },
                {
                    "stage_name": "01_01_05_FeaturePreparation",
                    "transform_ops": [
                        {
                            "method": "copy_only",
                            "params": {
                                "source_folder": "01_01_04_Preprocessing",
                                "dest_folder": "01_01_05_FeaturePreparation",
                                "subfolder_source": candidate_key,
                                "subfolder_dest": candidate_key,
                                "data_dir_path": data_dir_path
                            }
                        }
                    ]
                }
            ]
        }

        # Hard-coded plan for "FNOWithDropout" (same stages, different top-level model_name)
        plan_for_fno_with_dropout = {
            "model_name": "FNOWithDropout",
            "stages": [
                {
                    "stage_name": "01_01_01_LoadRawData",
                    "transform_ops": [
                        {
                            "method": "copy_only",
                            "params": {
                                "source_folder": "00_Generate_Data",
                                "dest_folder": "01_01_LoadRawData",
                                "subfolder_dest": candidate_key,
                                "data_dir_path": data_dir_path
                            }
                        }
                    ]
                },
                {
                    "stage_name": "01_01_03_TransformRawData",
                    "transform_ops": [
                        {
                            "method": "copy_only",
                            "params": {
                                "source_folder": "01_01_LoadRawData",
                                "dest_folder": "01_01_03_TransformRawData",
                                "subfolder_source": candidate_key,
                                "subfolder_dest": candidate_key,
                                "data_dir_path": data_dir_path
                            }
                        }
                    ]
                },
                {
                    "stage_name": "01_01_04_Preprocessing",
                    "transform_ops": [
                        {
                            "method": "copy_only",
                            "params": {
                                "source_folder": "01_01_03_TransformRawData",
                                "dest_folder": "01_01_04_Preprocessing",
                                "subfolder_source": candidate_key,
                                "subfolder_dest": candidate_key,
                                "data_dir_path": data_dir_path
                            }
                        }
                    ]
                },
                {
                    "stage_name": "01_01_05_FeaturePreparation",
                    "transform_ops": [
                        {
                            "method": "copy_only",
                            "params": {
                                "source_folder": "01_01_04_Preprocessing",
                                "dest_folder": "01_01_05_FeaturePreparation",
                                "subfolder_source": candidate_key,
                                "subfolder_dest": candidate_key,
                                "data_dir_path": data_dir_path
                            }
                        }
                    ]
                }
            ]
        }

        # Hard-coded plan for "AFNO"
        plan_for_afno = {
            "model_name": "AFNO",
            "stages": [
                {
                    "stage_name": "01_01_01_LoadRawData",
                    "transform_ops": [
                        {
                            "method": "copy_only",
                            "params": {
                                "source_folder": "00_Generate_Data",
                                "dest_folder": "01_01_LoadRawData",
                                "subfolder_dest": candidate_key,
                                "data_dir_path": data_dir_path
                            }
                        }
                    ]
                },
                {
                    "stage_name": "01_01_03_TransformRawData",
                    "transform_ops": [
                        {
                            "method": "copy_only",
                            "params": {
                                "source_folder": "01_01_LoadRawData",
                                "dest_folder": "01_01_03_TransformRawData",
                                "subfolder_source": candidate_key,
                                "subfolder_dest": candidate_key,
                                "data_dir_path": data_dir_path
                            }
                        }
                    ]
                },
                {
                    "stage_name": "01_01_04_Preprocessing",
                    "transform_ops": [
                        {
                            "method": "copy_only",
                            "params": {
                                "source_folder": "01_01_03_TransformRawData",
                                "dest_folder": "01_01_04_Preprocessing",
                                "subfolder_source": candidate_key,
                                "subfolder_dest": candidate_key,
                                "data_dir_path": data_dir_path
                            }
                        }
                    ]
                },
                {
                    "stage_name": "01_01_05_FeaturePreparation",
                    "transform_ops": [
                        {
                            "method": "copy_only",
                            "params": {
                                "source_folder": "01_01_04_Preprocessing",
                                "dest_folder": "01_01_05_FeaturePreparation",
                                "subfolder_source": candidate_key,
                                "subfolder_dest": candidate_key,
                                "data_dir_path": data_dir_path
                            }
                        }
                    ]
                }
            ]
        }

        # Decide which plan to return based on model_name
        if model_name == "FNO":
            return plan_for_fno
        elif model_name == "FNOWithDropout":
            return plan_for_fno_with_dropout
        elif model_name == "AFNO":
            return plan_for_afno
        else:
            raise NotImplementedError(
                f"[OntologyEngine] Model '{model_name}' is not implemented. "
                "Available options: ['FNO', 'FNOWithDropout', 'AFNO']."
            )

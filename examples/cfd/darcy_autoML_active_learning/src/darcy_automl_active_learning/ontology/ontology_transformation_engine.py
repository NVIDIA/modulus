import os
import shutil
import glob
from pathlib import Path
from typing import Optional

class OntologyTransformationEngine:
    """
    The OntologyTransformationEngine provides a collection of methods (transformations)
    to modify PDE data sets (e.g., .pt files, mesh files) in ways that align with 
    different candidate model requirements. The 'OntologyEngine' or 'transformation plan' 
    can direct which of these methods to call for each pipeline stage and candidate.

    Example usage:
      engine = OntologyTransformationEngine()
      # Suppose a plan stage has:
      # {
      #   "method": "copy_only",
      #   "params": {
      #       "source_folder": "01_01_LoadRawData",
      #       "dest_folder": "01_01_03_TransformRawData",
      #       "subfolder_source": "candidate0",
      #       "subfolder_dest": "candidate0",
      #       "data_dir_path": "examples/cfd/darcy_autoML_active_learning/data"
      #   }
      # }
      # This will copy from  <data_dir_path>/01_01_LoadRawData/candidate0
      #             into  <data_dir_path>/01_01_03_TransformRawData/candidate0
    """

    def __init__(self):
        """
        Initialize any resources or configurations needed by the engine.
        """
        pass

    # ------------------------------------------------------------------------
    # 1) COPY_ONLY
    # ------------------------------------------------------------------------
    def copy_only(self, 
                  source_folder: str, 
                  dest_folder: str,
                  data_dir_path: Optional[str] = None,
                  subfolder_source: Optional[str] = None,
                  subfolder_dest: Optional[str] = None,
                  **kwargs) -> None:
        """
        Copies all relevant data files (e.g., .pt, .json) from source_folder to dest_folder.

        If data_dir_path is provided, both source_folder and dest_folder will be treated as
        relative to that base path. If subfolder_source or subfolder_dest is provided, those
        subfolders are appended to the final source or destination path, respectively.

        :param source_folder:     Directory containing the files to copy (relative or absolute).
        :param dest_folder:       Directory where files should be placed (relative or absolute).
        :param data_dir_path:     Optional base directory path. If provided, source/dest paths
                                  are joined relative to this path. If None, the paths are used
                                  as given.
        :param subfolder_source:  (Optional) Additional subfolder appended to source_folder.
        :param subfolder_dest:    (Optional) Additional subfolder appended to dest_folder.
        :param kwargs:            Placeholder for any unused params from the JSON plan
                                  (so we don't raise unexpected-arg errors).
        """
        # 1) Resolve base paths
        if data_dir_path is not None:
            base_path = Path(data_dir_path)

            # If source_folder is not absolute, prepend data_dir_path
            sf_path = Path(source_folder)
            if not sf_path.is_absolute():
                sf_path = base_path / sf_path

            # If dest_folder is not absolute, prepend data_dir_path
            df_path = Path(dest_folder)
            if not df_path.is_absolute():
                df_path = base_path / df_path
        else:
            # Use the folders exactly as provided
            sf_path = Path(source_folder)
            df_path = Path(dest_folder)

        # 2) Append subfolders if provided
        if subfolder_source:
            sf_path = sf_path / subfolder_source
        if subfolder_dest:
            df_path = df_path / subfolder_dest

        # 3) Create the destination directory if it doesn't exist
        os.makedirs(df_path, exist_ok=True)

        # 4) Example: copy all .pt files and data_desc.json if it exists
        patterns = ["*.pt", "data_desc.json"]
        for pattern in patterns:
            for file_path in sf_path.glob(pattern):
                fname = file_path.name
                dest_path = df_path / fname
                shutil.copy2(file_path, dest_path)

        print(f"[OntologyTransformationEngine] COPY_ONLY done: {sf_path} -> {df_path}")

    # ------------------------------------------------------------------------
    # 2) TRANSFORM_MESH_TO_GRID (placeholder)
    # ------------------------------------------------------------------------
    def transform_mesh_to_grid(self, source_folder: str,
                               dest_folder: str,
                               interpolation_method: str = "linear",
                               target_resolution: int = 64,
                               **kwargs) -> None:
        """
        Converts unstructured mesh data into a uniform grid format. Typically involves
        an interpolation step from mesh vertices/cells onto a regular lattice.
        """
        print(f"[OntologyTransformationEngine] Placeholder: transform_mesh_to_grid() with "
              f"{interpolation_method=}, {target_resolution=}, source={source_folder}, dst={dest_folder}")

    # ------------------------------------------------------------------------
    # 3) TRANSFORM_DECIMATE_MESH (placeholder)
    # ------------------------------------------------------------------------
    def transform_decimate_mesh(self, source_folder: str,
                                dest_folder: str,
                                decimation_ratio: float = 0.5,
                                **kwargs) -> None:
        """
        Reduces the number of vertices/faces in a mesh to lower resolution.
        """
        print(f"[OntologyTransformationEngine] Placeholder: transform_decimate_mesh() with "
              f"{decimation_ratio=}, source={source_folder}, dst={dest_folder}")

    # ------------------------------------------------------------------------
    # 4) TRANSFORM_REGRID_DATA (placeholder)
    # ------------------------------------------------------------------------
    def transform_regrid_data(self, source_folder: str,
                              dest_folder: str,
                              new_resolution: int,
                              **kwargs) -> None:
        """
        Changes the resolution of grid data (e.g. from 128x128 to 64x64).
        """
        print(f"[OntologyTransformationEngine] Placeholder: transform_regrid_data() with "
              f"{new_resolution=}, source={source_folder}, dst={dest_folder}")

    # ------------------------------------------------------------------------
    # 5) TRANSFORM_ADD_BOUNDARY_CHANNEL (placeholder)
    # ------------------------------------------------------------------------
    def transform_add_boundary_channel(self, source_folder: str,
                                       dest_folder: str,
                                       boundary_label: str = "boundary_mask",
                                       **kwargs) -> None:
        """
        Inserts an extra channel marking domain boundaries, inlets, outlets, etc.
        """
        print(f"[OntologyTransformationEngine] Placeholder: transform_add_boundary_channel() "
              f"with {boundary_label=}, source={source_folder}, dst={dest_folder}")

    # ------------------------------------------------------------------------
    # 6) TRANSFORM_COORDINATE_MAPPING (placeholder)
    # ------------------------------------------------------------------------
    def transform_coordinate_mapping(self, source_folder: str,
                                     dest_folder: str,
                                     mapping_type: str = "implicit uniform",
                                     **kwargs) -> None:
        """
        Adjust coordinate references or embed coordinate arrays for PDE fields.
        E.g., convert from (i, j) indices to explicit (x, y), or from Cartesian to polar coords.
        """
        print(f"[OntologyTransformationEngine] Placeholder: transform_coordinate_mapping() "
              f"with {mapping_type=}, source={source_folder}, dst={dest_folder}")

    # ------------------------------------------------------------------------
    # 7) TRANSFORM_NORMALIZE_TENSORS (placeholder)
    # ------------------------------------------------------------------------
    def transform_normalize_tensors(self, source_folder: str,
                                    dest_folder: str,
                                    normalization_type: str = "zscore",
                                    **kwargs) -> None:
        """
        Scales or normalizes PDE tensor fields (e.g., zero-mean, unit-variance).
        """
        print(f"[OntologyTransformationEngine] Placeholder: transform_normalize_tensors() "
              f"with {normalization_type=}, source={source_folder}, dst={dest_folder}")

    # ------------------------------------------------------------------------
    # 8) TRANSFORM_TIME_SUBSAMPLING (placeholder)
    # ------------------------------------------------------------------------
    def transform_time_subsampling(self, source_folder: str,
                                   dest_folder: str,
                                   step: int = 2,
                                   **kwargs) -> None:
        """
        Subsamples time frames from a transient dataset. E.g. keep every 2nd or 5th time step.
        """
        print(f"[OntologyTransformationEngine] Placeholder: transform_time_subsampling() "
              f"with step={step}, source={source_folder}, dst={dest_folder}")

    # ------------------------------------------------------------------------
    # 9) Additional transformations (placeholders)
    # ------------------------------------------------------------------------
    def transform_remove_outliers(self, source_folder: str,
                                  dest_folder: str,
                                  z_threshold: float = 3.0,
                                  **kwargs) -> None:
        """
        Removes or clips outlier values in PDE fields if they exceed a certain
        statistical threshold (e.g. z-score > 3).
        """
        print(f"[OntologyTransformationEngine] Placeholder: transform_remove_outliers() "
              f"with z_threshold={z_threshold}, source={source_folder}, dst={dest_folder}")

    def transform_detect_replace_nans(self, source_folder: str,
                                      dest_folder: str,
                                      replacement_value: float = 0.0,
                                      **kwargs) -> None:
        """
        Detects NaNs or infinite values and replaces them with a given default.
        """
        print(f"[OntologyTransformationEngine] Placeholder: transform_detect_replace_nans() "
              f"with replacement_value={replacement_value}, source={source_folder}, dst={dest_folder}")

    def transform_log_stats(self, source_folder: str, 
                            dest_folder: str,
                            **kwargs) -> None:
        """
        Logs basic statistics (mean, std, min, max) per PDE channel for QA/QC. 
        Could write to a local text file or log to console.
        """
        print(f"[OntologyTransformationEngine] Placeholder: transform_log_stats(), "
              f"source={source_folder}, dst={dest_folder}")

    def transform_multi_physics_combine(self, source_folder: str,
                                        dest_folder: str,
                                        fields_to_combine=None,
                                        **kwargs) -> None:
        """
        Merges data from multiple PDE fields (e.g., fluid + thermal) into a single file 
        or a new set of channels if needed.
        """
        print(f"[OntologyTransformationEngine] Placeholder: transform_multi_physics_combine() "
              f"with fields_to_combine={fields_to_combine}, source={source_folder}, dst={dest_folder}")

"""
src/data_desc_logic.py

This module provides PDE data descriptor loading and model compatibility checks.
It defines:
  1) A constant list of minimal required fields for the data descriptor.
  2) A function `load_data_descriptor(desc_path)` that loads a descriptor from a
     JSON file and verifies these required fields under "data_structure".
  3) A function `check_data_model_compatibility(data_desc, model_desc)` that
     checks whether the dataset descriptor is compatible with at least one of
     the model's accepted data formats (as specified in its descriptor).

No testing or demo code is included here. Usage examples and tests
should reside in separate modules or notebooks.

Typical usage:
    from data_desc_logic import (
        DATA_DESCRIPTOR_REQUIRED_FIELDS,
        load_data_descriptor,
        check_data_model_compatibility
    )
"""

import os
import json

# 1) Minimal required fields for data descriptor
DATA_DESCRIPTOR_REQUIRED_FIELDS = [
    "dimension",
    "geometry_type",
    "uniform",
    "representation",
    "is_transient",
    "boundary",
    "cell_type",
    "decimation",
    "channels",
    # decimation_level is optional
    # coordinate_mapping is optional
]

def load_data_descriptor(desc_path: str) -> dict:
    """
    Load a PDE dataset descriptor from a JSON file and do a basic check
    for required fields.

    The descriptor is typically something like:
    {
      "descriptor_name": "Darcy2D_Uniform_1Ch",
      "data_structure": {
          "dimension": 2,
          "geometry_type": "grid",
          "uniform": true,
          "representation": {...},
          "is_transient": false,
          "boundary": false,
          "cell_type": null,
          "decimation": false,
          "channels": 1
          # Optional: "decimation_level", "coordinate_mapping", etc.
      }
    }

    Parameters
    ----------
    desc_path : str
        Path to the JSON descriptor file.

    Returns
    -------
    dict
        A Python dictionary with the loaded descriptor. It must have a
        "data_structure" sub-dict containing the minimal required fields.

    Raises
    ------
    FileNotFoundError
        If no file is found at the given path.
    ValueError
        If the descriptor is missing the "data_structure" key or
        if one of the minimal required fields is missing in that sub-dict.

    Notes
    -----
    - This function does not check geometry or channel compatibility with
      any particular model. For that, see `check_data_model_compatibility`.
    - The validated dictionary is returned so you can pass it to other
      pipeline steps.
    """
    if not os.path.isfile(desc_path):
        raise FileNotFoundError(f"No descriptor file found at {desc_path}")

    with open(desc_path, "r") as f:
        data_desc = json.load(f)

    # The top-level dict must have a 'data_structure' sub-dict
    if "data_structure" not in data_desc:
        raise ValueError("Missing top-level key 'data_structure' in the JSON descriptor.")

    ds = data_desc["data_structure"]
    for field in DATA_DESCRIPTOR_REQUIRED_FIELDS:
        if field not in ds:
            raise ValueError(
                f"Data descriptor is missing required field '{field}' in 'data_structure'."
            )

    return data_desc


def check_data_model_compatibility(data_desc: dict, model_desc: dict) -> bool:
    """
    Checks whether a given dataset descriptor is compatible with at least one
    of the accepted data formats specified by the model descriptor.

    The dataset descriptor is assumed to have been loaded and validated
    by `load_data_descriptor`. The model descriptor is a Python dictionary
    that typically includes a "model_name" and an "accepted_formats" list,
    each format being a dictionary that describes permissible data structure
    attributes (dimension, geometry_type, uniform, channels_min, etc.).

    If the data descriptor satisfies all requirements of at least one
    accepted format in the model descriptor, this function returns True.
    Otherwise, it returns False.

    Parameters
    ----------
    data_desc : dict
        A Python dictionary representing the dataset descriptor, as
        loaded by `load_data_descriptor(...)`. Must contain a
        "data_structure" sub-dictionary with fields like "dimension",
        "geometry_type", etc.

    model_desc : dict
        A Python dictionary describing the model's accepted data formats
        under a key "accepted_formats". For example::

            model_desc = {
                "model_name": "FNO",
                "accepted_formats": [
                    {
                        "dimension": [2, 3],
                        "geometry_type": "grid",
                        "uniform": True,
                        "channels_min": 1
                    }
                    # ... possibly more accepted formats
                ]
            }

    Returns
    -------
    bool
        True if the dataset descriptor is compatible with at least one
        of the model's accepted data formats; False otherwise.

    Raises
    ------
    KeyError
        If the model descriptor lacks an "accepted_formats" key, or if
        expected sub-keys are missing within those formats.

    Notes
    -----
    - Each format in `model_desc["accepted_formats"]` is compared against
      the dataset descriptor's "data_structure" fields:
        * dimension        -> must be in the accepted list (e.g. [2, 3])
        * geometry_type    -> must match exactly
        * uniform          -> must match exactly
        * channels_min     -> ensures data_struct["channels"] >= channels_min
      Additional constraints can be added as needed.
    - This function returns True immediately upon finding the first format
      that matches all constraints. Otherwise, it returns False.
    """
    ds = data_desc["data_structure"]
    if "accepted_formats" not in model_desc:
        raise KeyError(
            "Model descriptor must contain an 'accepted_formats' key defining supported formats."
        )

    accepted_formats = model_desc["accepted_formats"]
    if not isinstance(accepted_formats, list):
        raise ValueError(
            f"'accepted_formats' must be a list in model descriptor; got {type(accepted_formats)}"
        )

    # Loop over each accepted format
    for fmt in accepted_formats:
        # dimension check
        if "dimension" in fmt:
            valid_dims = fmt["dimension"]
            if isinstance(valid_dims, list):
                if ds["dimension"] not in valid_dims:
                    continue
            else:
                # If dimension is not a list, we assume exact match required
                if ds["dimension"] != valid_dims:
                    continue

        # geometry_type check
        if "geometry_type" in fmt:
            if ds["geometry_type"] != fmt["geometry_type"]:
                continue

        # uniform check
        if "uniform" in fmt:
            if ds["uniform"] != fmt["uniform"]:
                continue

        # channels_min check
        channels_min = fmt.get("channels_min")
        if channels_min is not None:
            if ds["channels"] < channels_min:
                continue

        # If we get here, all constraints match for this format
        return True

    # None of the accepted formats matched
    return False

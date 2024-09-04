# Utils for unified training recipe

import os
import fsspec
import zarr
import numpy as np

def get_filesystem(
    type: str, # "file" or "s3"
    key: str = None,
    endpoint_url: str = None,
    region_name: str = None,
):
    """
    Get filesystem object based on the type

    Parameters
    ----------
    type : str
        Type of filesystem. Currently supports "file" and "s3"
    key : str
        Key for s3
    endpoint_url : str
        Endpoint url for s3
    region_name : str
        Region name for s3

    Returns
    -------
    fs : fsspec.filesystem
        Filesystem object
    """

    if type == "file":
        fs = fsspec.filesystem("file")
    elif type == "s3":
        fs = fsspec.filesystem(
            "s3",
            key=key,
            secret=os.environ["AWS_SECRET_ACCESS_KEY"],
            client_kwargs={
                "endpoint_url": endpoint_url,
                "region_name": region_name,
            },
        )
    else:
        raise ValueError(f"Unknown type: {type}")

    return fs

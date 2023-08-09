# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List
import fsspec
import fsspec.implementations.cached
import s3fs
import builtins
import urllib.request
import os
import hashlib
import requests

import logging

logger = logging.getLogger(__name__)

try:
    LOCAL_CACHE = os.environ["LOCAL_CACHE"]
except KeyError:
    LOCAL_CACHE = os.environ["HOME"] + "/.cache"


def _cache_fs(fs):
    return fsspec.implementations.cached.CachingFileSystem(
        fs=fs, cache_storage=LOCAL_CACHE
    )


def _get_fs(path):
    if path.startswith("s3://"):
        return s3fs.S3FileSystem(client_kwargs=dict(endpoint_url="https://pbss.s8k.io"))
    else:
        return fsspec.filesystem("file")


def _download_cached(path: str, recursive: bool = False) -> str:
    sha = hashlib.sha256(path.encode())
    filename = sha.hexdigest()
    cache_path = os.path.join(LOCAL_CACHE, filename)

    url = urllib.parse.urlparse(path)

    # TODO watch for race condition here
    if not os.path.exists(cache_path):
        logger.debug("Downloading %s to cache: %s", path, cache_path)
        if path.startswith("s3://"):
            fs = _get_fs(path)
            fs.get(path, cache_path, recursive=recursive)
        elif url.scheme == "http":
            # urllib.request.urlretrieve(path, cache_path)
            # TODO: Check if this supports directory fetches
            response = requests.get(path, stream=True, timeout=5)
            with open(cache_path, "wb") as output:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        output.write(chunk)
        elif url.scheme == "file":
            path = os.path.join(url.netloc, url.path)
            return path
        else:
            return path

    else:
        logger.debug("Opening from cache: %s", cache_path)

    return cache_path


class Package:
    """A package

    Represents a potentially remote directory tree
    """

    def __init__(self, root: str, seperator: str):
        self.root = root
        self.seperator = seperator

    def get(self, path: str, recursive: bool = False) -> str:
        """Get a local path to the item at ``path``

        ``path`` might be a remote file, in which case it is downloaded to a
        local cache at $LOCAL_CACHE or $HOME/.cache/modulus first.
        """
        return _download_cached(self._fullpath(path), recursive=recursive)

    def _fullpath(self, path):
        return self.root + self.seperator + path

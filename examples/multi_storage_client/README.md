# Training from Object Storage using Multi-Storage Client 

## What is Multi-Storage Client (MSC)?

[Multi-Storage Client](https://github.com/NVIDIA/multi-storage-client) is a Python
library that provides a unified interface for accessing various object stores and
file systems. It makes it easy for ML workloads to use object stores by providing
a familiar file-like interface without sacrificing performance. The library adds
new functionality, such as caching, client-side observability, and leverages the native
SDKs specific to each object store for optimal performance.

## Getting Started

### Installation

```bash
pip install -r requirements.txt
```
Or install different extra dependencies based on your object storage backend:
```bash
# POSIX file systems.
pip install multi-storage-client

# NVIDIA AIStore.
pip install "multi-storage-client[aistore]"

# Azure Blob Storage.
pip install "multi-storage-client[azure-storage-blob]"

# AWS S3 and S3-compatible object stores.
pip install "multi-storage-client[boto3]"

# Google Cloud Storage (GCS).
pip install "multi-storage-client[google-cloud-storage]"

# Oracle Cloud Infrastructure (OCI) Object Storage.
pip install "multi-storage-client[oci]"
```

### Configuration File

The MSC configuration file defines profiles which include storage provider configurations.
An example MSC configuration file can be found at [msc_config.yaml](./msc_config.yaml).
In this example, the data is stored in the `cwb-diffusions` bucket in a S3-compatible
object store and credentials are inferred from the environment variables `S3_KEY` and `S3_SECRET`.

## Update Code Path with MSC

For Modulus’s use cases, where Zarr is commonly used in training workflows,
migrating to MSC is a straightforward process involving only configuration changes.
For example, in the [Corrdiff](../generative/corrdiff/) training example, data
currently accessed from the file system can be updated to MSC by modifying the
input path from `/code/2023-01-24-cwb-4years.zarr` to `msc://cwb-diffusions/2023-01-24-cwb-4years.zarr`,
with the MSC configuration file defined in [msc_config.yaml](./msc_config.yaml).
This assumes the data stored in the local file has been moved to a S3 bucket `cwb-diffusions`.

### Current code path (Training from File System):

```bash
input_path = "/code/2023-01-24-cwb-4years.zarr"
zarr.open_consolidated(input_path)
```

### Updated code path (Training from Object Store using MSC):

```bash
input_path = "msc://cwb-diffusions/2023-01-24-cwb-4years.zarr"
zarr.open_consolidated(input_path)
```

## Additional Information

- [Multi-Storage Client Documentation](https://nvidia.github.io/multi-storage-client/)

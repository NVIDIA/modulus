from typing import Callable, Iterable, Optional

import numpy as np
from torch import Tensor
from torch.utils.data import Dataset


class DictDataset(Dataset):
    def __init__(self, data_dict: dict):
        self.data_dict = data_dict
        for k, v in data_dict.items():
            assert len(v) == len(
                data_dict[list(data_dict.keys())[0]]
            ), "All data must have the same length"

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.data_dict.items()}

    def __len__(self):
        return len(self.data_dict[list(self.data_dict.keys())[0]])


class DictDatasetWithConstant(DictDataset):
    def __init__(self, data_dict: dict, constant_dict: dict):
        super().__init__(data_dict)
        self.constant_dict = constant_dict

    def __getitem__(self, index):
        return_dict = {k: v[index] for k, v in self.data_dict.items()}
        return_dict.update(self.constant_dict)
        return return_dict


class MappingDatasetWrapper(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        mapping: dict,
        every_n_data: Optional[int] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        r"""
        Args:
            dataset (Dataset): The dataset to wrap
            mapping (dict): The mapping from the dataset to the new dataset. If the key is comma separated, it will be interpreted as dataset_key, subkey.
            every_n_data (Optional[int], optional): If not None, only return every n data. Defaults to None.
            transform (Optional[Callable], optional): A callable to transform the data. Defaults to None.
        """
        self.dataset = dataset
        self.mapping = mapping
        self.every_n_data = every_n_data
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]

        return_dict = {}
        for k, v in self.mapping.items():
            # If comma doesn't exist, just use the key
            if "," not in k:
                if k not in item:
                    raise KeyError(f"Key {k} not found in dataset")
                return_value = item[k]
            else:
                # Split the key into dataset_key, subkey
                dataset_key, subkey = k.split(",")
                # Get the value from the dataset
                if dataset_key not in item:
                    raise KeyError(f"Key {dataset_key} not found in dataset")
                if subkey not in item[dataset_key]:
                    raise KeyError(f"Key {subkey} not found in dataset[{dataset_key}]")

                return_value = item[dataset_key][subkey]

            if self.every_n_data is not None and (
                isinstance(return_value, Iterable)
                or isinstance(return_value, np.ndarray)
                or isinstance(return_value, Tensor)
            ):
                return_value = return_value[:: self.every_n_data]
            return_dict[v] = return_value

        if self.transform is not None:
            return_dict = self.transform(return_dict)

        return return_dict

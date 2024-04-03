import tarfile
import uuid
from collections import defaultdict
from multiprocessing import Pool, set_start_method
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

# TODO(akamenev): migration
# import fire
import numpy as np
import pandas as pd
import pyvista as pv
import torch
import webdataset as wds
from torch.utils.data import DataLoader, Dataset

try:
    import ensightreader
except ImportError:
    print("Could not import ensightreader. Please install it from `pip install ensight-reader`")

# Add parent's parent directory to path
if __name__ == "__main__":
    import sys

    # Get current path
    current_path = Path(__file__).resolve().parent
    # Add parent's parent directory to path
    sys.path.append(str(current_path.parent.parent))

from src.data.base_datamodule import BaseDataModule
from src.data.mesh_utils import Normalizer, compute_drag_coefficient, convert_to_pyvista
from src.data.webdataset_base import NumpyPreprocess, Webdataset

# DrivAer dataset
# Air density = 1.205 kg/m^3
# Stream velocity = 38.8889 m/s
DRIVAER_AIR_DENSITY = 1.205
DRIVAER_STREAM_VELOCITY = 38.8889

# DrivAer pressure mean and std
DRIVAER_PRESSURE_MEAN = -150.13066236223494
DRIVAER_PRESSURE_STD = 229.1046667362158


class DrivAerDataset(Dataset):
    """DrivAer dataset.

    Data sets:
    Data set A: DrivAer without spoiler, 400 simulations (280 train, 60 validation, 60 test)
    Data set B: DrivAer with spoiler, 600 simulations (420 train, 90 validation, 90 test)

    1.       Train on A, test on  A
    2.       Train on B, test on B
    3.       Train on A + N samples of B (N is {0, 10, 50, 200, Max}), test on A and B
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        phase: Literal["train", "val", "test"] = "train",
        has_spoiler: bool = False,
        variables: Optional[list] = [
            "time_avg_pressure",
            # "time_avg_velocity", # data is invalid
            "time_avg_wall_shear_stress",
        ],
    ):
        if isinstance(data_path, str):
            data_path = Path(data_path)
        data_path = data_path.expanduser()
        self.data_path = data_path  # Path that contains data_set_A and data_set_B
        assert isinstance(has_spoiler, bool), f"has_spoiler should be a boolean, got {has_spoiler}"
        assert phase in [
            "train",
            "val",
            "test",
        ], f"phase should be one of ['train', 'val', 'test'], got {phase}"
        assert self.data_path.exists(), f"Path {self.data_path} does not exist"
        assert self.data_path.is_dir(), f"Path {self.data_path} is not a directory"
        assert (
            self.data_path / "data_set_A"
        ).exists(), f"Path {self.data_path} does not contain data_set_A"
        self.has_spoiler = has_spoiler

        if has_spoiler:
            self.data_path = self.data_path / "data_set_B"
            self.TEST_INDS = np.array(range(510, 600))
            self.VAL_INDS = np.array(range(420, 510))
            self.TRAIN_INDS = np.array(range(420))
        else:
            self.data_path = self.data_path / "data_set_A"
            self.TEST_INDS = np.array(range(340, 400))
            self.VAL_INDS = np.array(range(280, 340))
            self.TRAIN_INDS = np.array(range(280))

        # Load parameters
        parameters = pd.read_csv(self.data_path / "ParameterFile.txt", delim_whitespace=True)
        self.phase = phase
        if phase == "train":
            self.indices = self.TRAIN_INDS
        elif phase == "val":
            self.indices = self.VAL_INDS
        elif phase == "test":
            self.indices = self.TEST_INDS
        self.parameters = parameters.iloc[self.indices]
        self.variables = variables

    @property
    def _attribute(self, variable, name):
        return self.data[variable].attrs[name]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]
        case = ensightreader.read_case(
            self.data_path / "snapshots" / f"EnSight{index}" / f"EnSight{index}.case"
        )
        geofile = case.get_geometry_model()
        ids = geofile.get_part_ids()  # list of part ids
        # remove id 49, which is the internalMesh for without spoiler
        if self.has_spoiler:
            ids.remove(50)
        else:
            ids.remove(49)

        pv_parts = []
        variable_data = defaultdict(list)
        for part_id in ids:
            # print(f"Reading part {part_id} / {len(ids)} for case {index}")
            part = geofile.get_part_by_id(part_id)
            element_blocks = part.element_blocks
            with geofile.open() as fp_geo:
                part_coordinates = part.read_nodes(fp_geo)

            # Read element data
            faces = []
            with geofile.open() as fp_geo:
                for block in part.element_blocks:
                    faces_block = convert_to_pyvista(block, fp_geo)
                    faces.append(faces_block)
            faces = np.concatenate(faces)
            part_mesh = pv.PolyData(part_coordinates, faces)
            pv_parts.append(part_mesh)

            # Get variables
            for variable_name in self.variables:
                variable = case.get_variable(variable_name)
                blocks = []
                for element_block in element_blocks:
                    with variable.mmap() as mm_var:
                        data = variable.read_element_data(
                            mm_var, part.part_id, element_block.element_type
                        )
                        if data is None:
                            print(
                                f"Variable {variable_name} is None in element block {element_block}"
                            )

                        blocks.append(data)
                data = np.concatenate(blocks)
                # scalar variables are transformed to N,1 arrays
                if len(data.shape) == 1:
                    data = data[:, np.newaxis]
                variable_data[variable_name].append(data)

            # Check if the data is consistent
            for k, v in variable_data.items():
                # The last item is the current part
                assert len(v[-1]) == part_mesh.n_faces_strict, f"Length of {k} is not consistent"

        # Combine parts into one mesh
        mesh = pv.MultiBlock(pv_parts).combine(merge_points=True).extract_surface().clean()

        # Concatenate the variable_data
        for k, v in variable_data.items():
            variable_data[k] = np.concatenate(v).squeeze()
            assert len(variable_data[k]) == mesh.n_faces_strict, f"Length of {k} is not consistent"

        # Estimate normals
        mesh.compute_normals(
            cell_normals=True, point_normals=True, flip_normals=True, inplace=True
        )

        # Extract cell centers and areas
        cell_centers = np.array(mesh.cell_centers().points)
        cell_normals = np.array(mesh.cell_normals)
        cell_sizes = mesh.compute_cell_sizes(length=False, area=True, volume=False)
        cell_sizes = np.array(cell_sizes.cell_data["Area"])

        # Normalize cell normals
        cell_normals = cell_normals / np.linalg.norm(cell_normals, axis=1)[:, np.newaxis]

        # Get the idx'th row from pandas dataframe
        curr_params = self.parameters.iloc[idx]

        # Add the parameters to the dictionary
        return {
            "mesh_nodes": np.array(mesh.points),
            "cell_centers": cell_centers,
            "cell_areas": cell_sizes,
            "cell_normals": cell_normals,
            **curr_params.to_dict(),
            **variable_data,
        }


class DrivAerToWebdataset:
    def __init__(
        self,
        dataset: DrivAerDataset,
        output_path: Union[str, Path],
        tarfile_name: str = "data.tar",
    ):
        self.dataset = dataset
        self.tarfile_name = tarfile_name
        self.output_path = Path(output_path)
        self.temp_path = self.output_path / str(uuid.uuid4())[:8]
        self.temp_path.mkdir(exist_ok=True)
        print(
            f"Saving DrivAerWebdataset to {self.temp_path} for {self.dataset.data_path} phase: {self.dataset.phase}, has_spoiler: {self.dataset.has_spoiler}"
        )
        # Create a text file in the temp_path and print dataset information
        with open(self.temp_path / "info.txt", "w") as f:
            f.write(f"Dataset: {self.dataset.data_path}\n")
            f.write(f"Phase: {self.dataset.phase}\n")
            f.write(f"Has spoiler: {self.dataset.has_spoiler}\n")
            f.write(f"Number of items: {len(self.dataset)}\n")

    def _save_item(self, idx: int):
        print(f"Saving item {idx}/{len(self.dataset)}")
        item = self.dataset[idx]
        # Save to 0 padded index
        np.savez_compressed(
            self.temp_path / f"{idx:06d}.npz",
            **item,
        )

    def save(self, num_processes: int = 1):
        # Save each item in as a numpy file in parallel
        assert num_processes > 0, f"num_processes should be greater than 0, got {num_processes}"
        if num_processes < 2:
            for idx in range(len(self.dataset)):
                print(f"Saving item {idx}/{len(self.dataset)}")
                self._save_item(idx)
        elif num_processes > 1:
            with Pool(num_processes) as p:
                p.map(self._save_item, range(len(self.dataset)))

        # Compress the dataset to a tar file
        print("Compressing the dataset to a tar file")
        self.output_path = self.output_path.expanduser()
        self.output_path.mkdir(exist_ok=True)
        tar_path = self.output_path / self.tarfile_name
        with tarfile.open(tar_path, "w") as tar:
            for file in self.temp_path.glob("*.npz"):
                tar.add(file, arcname=file.name)


class _DrivAerWebdatasetPreprocess(NumpyPreprocess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.air_coeff = 2 / (DRIVAER_AIR_DENSITY * DRIVAER_STREAM_VELOCITY**2)
        # DrivAer Mean: -150.13066236223494, std: 229.1046667362158
        self.normalizer = Normalizer(
            kwargs.get("mean", DRIVAER_PRESSURE_MEAN),
            kwargs.get("std", DRIVAER_PRESSURE_STD),
        )
        every_n_data = kwargs.get("every_n_data", 1)
        if every_n_data is None:
            every_n_data = 1
        self.every_n_data = every_n_data

    def __call__(self, sample):
        np_dict = super().__call__(sample)

        if self.every_n_data > 1:
            # Downsample the data
            for k, v in np_dict.items():
                if (isinstance(v, np.ndarray) and v.size > 1) or (
                    isinstance(v, torch.Tensor) and v.numel() > 1
                ):
                    np_dict[k] = v[:: self.every_n_data]

        if "Snapshot" in np_dict:
            # array('EnSightXXX', dtype='<U10')
            # Convert it to an integer
            np_dict["Snapshot"] = int(np.char.replace(np_dict["Snapshot"], "EnSight", ""))

        # Compute drag coefficient using area, normal, pressure and wall shear stress
        drag_coef = compute_drag_coefficient(
            np_dict["cell_normals"],
            np_dict["cell_areas"],
            self.air_coeff / np_dict["proj_area_x"],
            np_dict["time_avg_pressure"],
            np_dict["time_avg_wall_shear_stress"],
        )
        # np_dict["c_d"] is computed on a finer mesh and the newly computed drag is on a coarser mesh so they are not equal
        np_dict["c_d_computed"] = drag_coef
        np_dict["time_avg_pressure_whitened"] = self.normalizer.encode(
            np_dict["time_avg_pressure"]
        )
        return np_dict


class DrivAerWebdataset(Webdataset):
    def __init__(self, paths: str | List[str], preprocessor_kwargs: Optional[dict] = None) -> None:
        if isinstance(paths, str):
            paths = [paths]
        if preprocessor_kwargs is None:
            preprocessor_kwargs = {}
        preprocessor_kwargs["np_ext"] = "npz"
        super().__init__(paths, _DrivAerWebdatasetPreprocess, preprocessor_kwargs)


class DrivAerDataModule(BaseDataModule):
    def __init__(
        self,
        data_path: Union[Path, str],
        subsets_postfix: Optional[List[str]] = ["spoiler", "nospoiler"],
        every_n_data: Optional[int] = None,  # downsample
    ) -> None:
        """
        Args:
            data_path (Union[Path, str]): Path that contains train and test directories
            subsets_postfix (Optional[List[str]], optional): Postfixes for the subsets. Defaults to ["spoiler", "nospoiler"].

        """
        super().__init__()

        if isinstance(data_path, str):
            data_path = Path(data_path)
        assert (
            data_path.exists() and data_path.is_dir()
        ), f"{data_path} must exist and should be a directory"
        self.data_dir = data_path
        self.subsets_postfix = subsets_postfix
        self.every_n_data = every_n_data
        self.setup()

    def setup(self, stage: Optional[str] = None):
        subsets_postfix = self.subsets_postfix
        every_n_data = self.every_n_data
        self._train_dataset = DrivAerWebdataset(
            [str(self.data_dir / f"train_{subset}.tar") for subset in subsets_postfix],
            preprocessor_kwargs={"every_n_data": every_n_data},
        )
        self._val_dataset = DrivAerWebdataset(
            [str(self.data_dir / f"val_{subset}.tar") for subset in subsets_postfix],
            preprocessor_kwargs={"every_n_data": every_n_data},
        )
        self._test_dataset = DrivAerWebdataset(
            [str(self.data_dir / f"test_{subset}.tar") for subset in subsets_postfix],
            preprocessor_kwargs={"every_n_data": every_n_data},
        )
        self.normalizer = Normalizer(DRIVAER_PRESSURE_MEAN, DRIVAER_PRESSURE_STD)
        self.air_coeff = 2 / (DRIVAER_AIR_DENSITY * DRIVAER_STREAM_VELOCITY**2)

    def encode(self, x):
        return self.normalizer.encode(x)

    def decode(self, x):
        return self.normalizer.decode(x)

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def val_dataset(self):
        return self._val_dataset

    @property
    def test_dataset(self):
        return self._test_dataset

    def train_dataloader(self, **kwargs) -> wds.WebLoader:
        collate_fn = getattr(self, "collate_fn", None)
        # Remove shuffle from kwargs
        kwargs.pop("shuffle", None)
        buffer_size = kwargs.pop("buffer_size", 100)
        return DataLoader(self.train_dataset.shuffle(buffer_size), collate_fn=collate_fn, **kwargs)

    def val_dataloader(self, **kwargs) -> DataLoader:
        collate_fn = getattr(self, "collate_fn", None)
        return DataLoader(self.val_dataset, collate_fn=collate_fn, **kwargs)

    def test_dataloader(self, **kwargs) -> DataLoader:
        collate_fn = getattr(self, "collate_fn", None)
        return DataLoader(self.test_dataset, collate_fn=collate_fn, **kwargs)


class DrivAerNoSpoilerDataModule(DrivAerDataModule):
    def __init__(self, data_path: Union[Path, str], every_n_data: Optional[int] = None) -> None:
        super().__init__(data_path, ["nospoiler"], every_n_data)


class DrivAerSpoilerDataModule(DrivAerDataModule):
    def __init__(self, data_path: Union[Path, str], every_n_data: Optional[int] = None) -> None:
        super().__init__(data_path, ["spoiler"], every_n_data)


def convert_to_webdataset(
    data_path: str,
    out_path: Optional[str] = "~/datasets/drivaer_webdataset",
    has_spoiler: Optional[bool] = False,
    phase: Optional[Literal["train", "val", "test"]] = "train",
    num_processes: Optional[int] = 8,
):
    set_start_method("spawn", force=True)
    # Add the parent directory to the path
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

    # Create separate paths for train/text, with and without spoiler
    print(f"Saving {phase} {'with' if has_spoiler else 'without'} spoiler")
    dataset = DrivAerDataset(data_path, phase=phase, has_spoiler=has_spoiler)
    output_path = Path(out_path).expanduser()
    output_path.mkdir(exist_ok=True)
    # Save to numpy
    DrivAerToWebdataset(
        dataset,
        output_path,
        tarfile_name=f"{phase}_{'spoiler' if has_spoiler else 'nospoiler'}.tar",
    ).save(num_processes=num_processes)


def compute_pressure_stats(*data_paths: List[str]):
    dataset = DrivAerWebdataset(data_paths)
    num_points = []
    means = []
    vars = []

    # visualize the progress bar using tqdm
    import tqdm

    for item in tqdm.tqdm(dataset):
        p = item["time_avg_pressure"]
        num_points.append(p.shape[0])
        means.append(p.mean().item())
        vars.append(p.var().item())
    # Compute normalization function
    mean = np.mean(means)
    std = np.sqrt(np.mean(vars))
    print(f"Mean: {mean}, std: {std}")
    # Save the parameters as a text file
    with open("pressure_normalization.txt", "w") as f:
        f.write(f"Mean: {mean}, std: {std}")


def test_datamodule(
    data_dir: str,
    subset_postfix: Optional[List[str]] = ["spoiler", "nospoiler"],
    every_n_data: Optional[int] = 1,
):
    datamodule = DrivAerDataModule(
        data_dir, subsets_postfix=subset_postfix, every_n_data=every_n_data
    )
    for i, batch in enumerate(datamodule.val_dataloader()):
        print(i, batch["cell_centers"].shape, batch["time_avg_pressure_whitened"].shape)


if __name__ == "__main__":
    __spec__ = None
    # fire.Fire(convert_to_webdataset)
    # fire.Fire(compute_pressure_stats)
    # TODO(akamenev): migration
    # fire.Fire(test_datamodule)

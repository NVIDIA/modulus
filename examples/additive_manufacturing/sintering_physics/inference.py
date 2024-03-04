# ignore_header_test
# ruff: noqa: E402

# © Copyright 2023 HP Development Company, L.P.
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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


import os, json
from absl import app
import time
from tqdm import tqdm
import numpy as np
import tree

try:
    import tensorflow as tf
except ImportError:
    raise ImportError(
        "Mesh Graph Net Datapipe requires the Tensorflow library. Install the "
        + "package at: https://www.tensorflow.org/install"
    )

physical_devices = tf.config.list_physical_devices("GPU")
try:
    for device_ in physical_devices:
        tf.config.experimental.set_memory_growth(device_, True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

import torch
from torch.utils.tensorboard import SummaryWriter

from modulus.models.vfgn.graph_network_modules import LearnedSimulator
from modulus.datapipes.vfgn import utils
from modulus.datapipes.vfgn.utils import _read_metadata
from modulus.models.vfgn.graph_dataset import GraphDataset

from modulus.distributed.manager import DistributedManager
from modulus.launch.logging import (
    LaunchLogger,
    PythonLogger,
    initialize_wandb,
    initialize_mlflow,
    RankZeroLoggingWrapper,
)

from constants import Constants

C = Constants()


class Stats:
    """
    Represents statistical attributes, specifically mean and standard deviation.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def to(self, device):
        """Transfers the mean and standard deviation to a specified device."""
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self


device = "cpu"

INPUT_SEQUENCE_LENGTH = 5  # calculate the last 5 velocities. [options: 5, 10]
PREDICT_LENGTH = 1  # [options: 5]

NUM_PARTICLE_TYPES = 3
KINEMATIC_PARTICLE_ID = 0  # refers to anchor point
METAL_PARTICLE_ID = 2  # refers to normal particles
ANCHOR_PLANE_PARTICLE_ID = 1  # refers to anchor plane


def infer_stage(
    model,
    features,
    global_context,
    current_positions,
    num_steps,
    ground_truth_positions,
    updated_predictions,
    sequence_length,
    metadata_1=None,
    metadata_2=None,
    renorm=False,
    rank_zero_logger=None,
):
    """
    Performs inference over a specified number of steps using a given model
    and updates predictions.
    """
    len_predicted = len(updated_predictions)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rank_zero_logger.info(f"global_context shape: {global_context.shape}")

    for step in range(num_steps):
        if global_context is None:
            global_context_step = None
        else:
            read_step_context = global_context[: step + INPUT_SEQUENCE_LENGTH]
            if read_step_context.shape[0] <= (sequence_length - 1):
                zero_pad = torch.zeros(
                    [sequence_length - read_step_context.shape[0] - 1, 1],
                    dtype=features["step_context"].dtype,
                ).to(device)
                # global_context_step = torch.concat([read_step_context, zero_pad], 0)
                global_context_step = torch.cat([read_step_context, zero_pad], 0)
            else:
                global_context_step = read_step_context[-(sequence_length - 1) :]
            global_context_step = torch.reshape(global_context_step, [1, -1])
            rank_zero_logger.info(f"global_context_step shape: {global_context_step.shape}")

        predict_positions = model.inference(
            position_sequence=current_positions.to(device),
            n_particles_per_example=features["n_particles_per_example"].to(device),
            n_edges_per_example=features["n_edges_per_example"].to(device),
            senders=features["senders"].to(device),
            receivers=features["receivers"].to(device),
            predict_length=PREDICT_LENGTH,
            particle_types=features["particle_type"].to(device),
            global_context=global_context_step.to(device),
        )

        kinematic_mask = (
            utils.get_kinematic_mask(features["particle_type"]).to(torch.bool).to(device)
        )
        positions_ground_truth = ground_truth_positions[:, step + len_predicted]

        predict_positions = predict_positions[:, 0].squeeze(1)
        kinematic_mask = torch.repeat_interleave(
            kinematic_mask, repeats=predict_positions.shape[-1]
        )
        kinematic_mask = torch.reshape(
            kinematic_mask, [-1, predict_positions.shape[-1]]
        )

        next_position = torch.where(
            kinematic_mask, positions_ground_truth, predict_positions
        )
        rank_zero_logger.info(f"ground truth position: {step + len_predicted}")

        updated_predictions.append(next_position)

        if C.rollout_refine is False:
            # False: rollout the predictions
            current_positions = torch.cat(
                [current_positions[:, 1:], next_position.unsqueeze(1)], axis=1
            )
        else:
            # True: single-step prediction for all steps
            current_positions = torch.cat(
                [current_positions[:, 1:], positions_ground_truth.unsqueeze(1)], axis=1
            )

        # renorm
        if renorm and step == (num_steps - 1):
            current_positions = (
                (current_positions * metadata_1["pos_std"] + metadata_1["pos_mean"])
                - metadata_2["pos_mean"]
            ) / metadata_2["pos_std"]

    return current_positions, updated_predictions


def load_stage_model(model, model_path, features, global_context_step, sequence_length):
    """
    Loads a model from a specified path, configures it for inference, and returns
    the prepared model.
    """
    device = "cpu"
    global_context_step = global_context_step[:, :sequence_length]

    model.inference(
        position_sequence=features["position"][:, 0:INPUT_SEQUENCE_LENGTH].to(device),
        n_particles_per_example=features["n_particles_per_example"].to(device),
        n_edges_per_example=features["n_edges_per_example"].to(device),
        senders=features["senders"].to(device),
        receivers=features["receivers"].to(device),
        predict_length=PREDICT_LENGTH,
        particle_types=features["particle_type"].to(device),
        global_context=global_context_step.to(device),
    )
    model.load_state_dict(torch.load(model_path), strict=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # config optimizer
    model.setMessagePassingDevices(["cuda:0"])
    model = model.to(device)
    model.eval()

    return model


cast = lambda v: np.array(v, dtype=np.float64)


def Inference(rank_zero_logger, dist):
    """
    Conducts a test process on a graph dataset using a multi-stage model approach
    for predictions.
    """
    rank_zero_logger.info(f"\n\n.......... Start calling model inference with defined data path ........\n\n")

    # config inference dataset
    dataset = GraphDataset(
        mode="rollout",
        split=C.eval_split
    )
    rank_zero_logger.info(f"Initialized inference dataset with mode {dataset.mode}, dataset size {dataset.size}...")

    metadat_1 = _read_metadata(os.path.join(C.data_path, C.meta1))
    rank_zero_logger.info(f"normalization_stats from sinter stage-1: {metadat_1}")

    metadat_2 = _read_metadata(os.path.join(C.data_path, C.meta2))

    acceleration_stats = Stats(
        torch.DoubleTensor(cast(metadat_1["acc_mean"])),
        torch.DoubleTensor(utils._combine_std(cast(metadat_1["acc_std"]), C.noise_std)),
    )
    velocity_stats = Stats(
        torch.DoubleTensor(cast(metadat_1["vel_mean"])),
        torch.DoubleTensor(utils._combine_std(cast(metadat_1["vel_std"]), C.noise_std)),
    )
    context_stats = Stats(
        torch.DoubleTensor(cast(metadat_1["context_mean"])),
        torch.DoubleTensor(
            utils._combine_std(cast(metadat_1["context_std"]), C.noise_std)
        ),
    )
    sequence_length_s1 = int(metadat_1["sequence_length"])

    normalization_stats = {
        "acceleration": acceleration_stats,
        "velocity": velocity_stats,
        "context": context_stats,
    }

    acceleration_stats_2 = Stats(
        torch.DoubleTensor(cast(metadat_2["acc_mean"])),
        torch.DoubleTensor(utils._combine_std(cast(metadat_2["acc_std"]), C.noise_std)),
    )
    velocity_stats_2 = Stats(
        torch.DoubleTensor(cast(metadat_2["vel_mean"])),
        torch.DoubleTensor(utils._combine_std(cast(metadat_2["vel_std"]), C.noise_std)),
    )
    context_stats_2 = Stats(
        torch.DoubleTensor(cast(metadat_2["context_mean"])),
        torch.DoubleTensor(
            utils._combine_std(cast(metadat_2["context_std"]), C.noise_std)
        ),
    )
    sequence_length_s2 = int(metadat_2["sequence_length"])
    normalization_stats_s2 = {
        "acceleration": acceleration_stats_2,
        "velocity": velocity_stats_2,
        "context": context_stats_2,
    }
    rank_zero_logger.info(f"normalization_stats from sinter stage-2: {metadat_2}")

    model_s1 = LearnedSimulator(
        num_dimensions=metadat_1["dim"] * PREDICT_LENGTH,
        num_seq=INPUT_SEQUENCE_LENGTH,
        boundaries=torch.DoubleTensor(metadat_1["bounds"]),
        num_particle_types=NUM_PARTICLE_TYPES,
        particle_type_embedding_size=16,
        normalization_stats=normalization_stats,
    )
    rank_zero_logger.info(f"initialized model#1 with LearnedSimulator")

    model_s2 = LearnedSimulator(
        num_dimensions=metadat_1["dim"] * PREDICT_LENGTH,
        num_seq=INPUT_SEQUENCE_LENGTH,
        boundaries=torch.DoubleTensor(metadat_1["bounds"]),
        num_particle_types=NUM_PARTICLE_TYPES,
        particle_type_embedding_size=16,
        normalization_stats=normalization_stats_s2,
    )
    rank_zero_logger.info(f"initialized model#2 with LearnedSimulator")

    loaded = False
    example_index = 0
    device = "cpu"

    # index for the step 100 model, 2 stages
    #     1. start_index, end_index = 0, 1321
    #     2. start_index, end_index = 1000, 3393
    START_INDEX_S1, END_INDEX_S1 = 0 // 100, 1321 // 100
    START_INDEX_S2, END_INDEX_S2 = 1321 // 100, 3393 // 100

    with torch.no_grad():
        for features, targets in tqdm(dataset):
            if loaded is False:
                global_context = features["step_context"].to(device)
                if global_context is None:
                    global_context_step = None
                else:
                    # global_context_step = global_context[
                    #     INPUT_SEQUENCE_LENGTH - 1].unsqueeze(-1)
                    global_context_step = global_context[:-1]
                    global_context_step = torch.reshape(global_context_step, [1, -1])

                ##### Load model from ckpts
                model_s1 = load_stage_model(
                    model_s1,
                    C.model_path_s1,
                    features,
                    global_context_step,
                    sequence_length_s1,
                )
                rank_zero_logger.info(f"Loaded model#1 from ckpt path {C.model_path_s1}")

                model_s2 = load_stage_model(
                    model_s2,
                    C.model_path_s2,
                    features,
                    global_context_step,
                    sequence_length_s2,
                )
                rank_zero_logger.info(f"Loaded model#1 from ckpt path {C.model_path_s2}")

                loaded = True

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            rank_zero_logger.info(f"device: {device}")

            ###### start prediction ######
            initial_positions = features["position"][:, 0:INPUT_SEQUENCE_LENGTH].to(
                device
            )
            ground_truth_positions = features["position"][
                :, INPUT_SEQUENCE_LENGTH:END_INDEX_S2
            ].to(device)
            global_context = features["step_context"].to(device)

            current_positions = initial_positions
            updated_predictions = []
            start_time = time.time()
            rank_zero_logger.info(f"start time: {start_time}")

            ############ stage-1  ############
            rank_zero_logger.info(f"device: {device}")
            num_steps_s1 = END_INDEX_S1 - START_INDEX_S1 - INPUT_SEQUENCE_LENGTH
            rank_zero_logger.info(f"start infer {num_steps_s1} steps ........ \n")
            current_positions, updated_predictions = infer_stage(
                model_s1,
                features,
                global_context[: (sequence_length_s1 + 1)],
                current_positions,
                num_steps_s1,
                ground_truth_positions,
                updated_predictions,
                sequence_length_s1 + 1,
                metadat_1,
                metadat_2,
                True,
                rank_zero_logger,
            )
            rank_zero_logger.info(f"updated_predictions step length from stage-1: {len(updated_predictions)}")

            ############ stage-2  ############
            num_steps_s2 = END_INDEX_S2 - START_INDEX_S2
            rank_zero_logger.info(f"start infer {num_steps_s2} steps ........ \n")
            current_positions, updated_predictions = infer_stage(
                model_s2,
                features,
                global_context[(sequence_length_s1 - INPUT_SEQUENCE_LENGTH) :],
                current_positions,
                num_steps_s2,
                ground_truth_positions,
                updated_predictions,
                sequence_length_s2 + 1,
                metadat_1,
                metadat_2,
                True,
                rank_zero_logger,
            )
            rank_zero_logger.info(f"updated_predictions step length from stage-2: {len(updated_predictions)}")

            ############ stage-transit  ############
            ## to be implemented

            end_time = time.time()
            rank_zero_logger.info(f"prediction time: {end_time - start_time}")

            # Store in pkl
            updated_predictions = torch.stack(updated_predictions)

            rank_zero_logger.info(
                f"\n\n finished running all stages, initial_positions shape: {initial_positions.shape},\n"
                f" ground_truth_positions shape: {ground_truth_positions.shape}, "
                f"\n updated_predictions shape: {updated_predictions.shape}"
            )

            rollout_op = {
                "initial_positions": tf.transpose(utils.torch2tf(initial_positions), [1, 0, 2]),
                "predicted_rollout": utils.torch2tf(updated_predictions),
                "ground_truth_rollout": tf.transpose(utils.torch2tf(ground_truth_positions), [1, 0, 2]),
                "particle_types": utils.torch2tf(features["particle_type"]),
                "global_context": utils.torch2tf(global_context),
            }

            squared_error = (
                rollout_op["predicted_rollout"] - rollout_op["ground_truth_rollout"]
            ) ** 2

            # Add a leading axis, since Estimator's predict method insists that all
            # tensors have a shared leading batch axis fo the same dims.
            rollout_op = tree.map_structure(lambda x: x.numpy(), rollout_op)

            rollout_op["metadat_1"] = metadat_1
            rollout_op["metadat_2"] = metadat_2
            filename = f"rollout_{C.eval_split}_{example_index}.pkl"
            filename = os.path.join(C.output_path, filename)
            if not os.path.exists(C.output_path):
                os.makedirs(C.output_path)
            with open(filename, "wb") as file:
                json.dump(rollout_op, file)
            example_index += 1

            rank_zero_logger.info(f"prediction time: {time.time()-start_time}\n")


def main(_):
    """
    Triggers the inference phase based on the configuration.
    """
    # initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)  # Rank 0 logger

    Inference(rank_zero_logger, dist)


if __name__ == "__main__":
    # tf.disable_v2_behavior()
    LaunchLogger.initialize()  # Modulus launch logger
    logger = PythonLogger("main")  # General python logger
    logger.file_logging()

    app.run(main)

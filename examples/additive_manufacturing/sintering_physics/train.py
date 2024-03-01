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
import random
import time
from tqdm import tqdm
import numpy as np
import math
import ast

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
from modulus.models.vfgn.graph_network import LearnedSimulator as LearnedSimulatorBeta
from modulus.utils.vfgn import utils
from modulus.utils.vfgn.utils import _read_metadata
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
    Represents statistical attributes with methods for device transfer.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def to(self, device):
        """
        Transfers the mean and standard deviation to a specified device.
        """
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


cast = lambda v: np.array(v, dtype=np.float64)


def Train(rank_zero_logger, dist):
    """
    Trains a graph-based model, evaluating and saving its performance periodically.
    """
    # config dataset
    dataset = GraphDataset(
        size=C.num_steps,
        data_path=C.data_path,
        batch_size=C.batch_size,
        prefetch_buffer_size=C.prefetch_buffer_size,
    )
    testDataset = GraphDataset(
        size=C.num_steps, split="test", data_path=C.data_path, batch_size=C.batch_size
    )

    # config model
    metadata = _read_metadata(C.data_path)
    acceleration_stats = Stats(
        torch.DoubleTensor(cast(metadata["acc_mean"])),
        torch.DoubleTensor(utils._combine_std(cast(metadata["acc_std"]), C.noise_std)),
    )
    velocity_stats = Stats(
        torch.DoubleTensor(cast(metadata["vel_mean"])),
        torch.DoubleTensor(utils._combine_std(cast(metadata["vel_std"]), C.noise_std)),
    )
    context_stats = Stats(
        torch.DoubleTensor(cast(metadata["context_mean"])),
        torch.DoubleTensor(
            utils._combine_std(cast(metadata["context_std"]), C.noise_std)
        ),
    )

    normalization_stats = {
        "acceleration": acceleration_stats,
        "velocity": velocity_stats,
        "context": context_stats,
    }
    model = LearnedSimulator(
        num_dimensions=metadata["dim"] * PREDICT_LENGTH,
        num_seq=INPUT_SEQUENCE_LENGTH,
        boundaries=torch.DoubleTensor(metadata["bounds"]),
        num_particle_types=NUM_PARTICLE_TYPES,
        particle_type_embedding_size=16,
        normalization_stats=normalization_stats,
    )

    writer = SummaryWriter(log_dir=C.model_path_vfgn)

    optimizer = None
    device = "cpu"
    step = 0
    running_loss = 0.0
    best_loss = 1000.0

    rank_zero_logger.info("Training started...")

    for features, targets in tqdm(dataset):

        inputs = features["position"]
        particle_types = features["particle_type"]

        sampled_noise = model.get_random_walk_noise_for_position_sequence(
            inputs, noise_std_last_step=C.noise_std
        )
        if C.loss.startswith("anchor"):
            rank_zero_logger.info("compute noise_mask...")
            print("compute noise_mask")
            # if C.loss == 'anchor':

            non_kinematic_mask = utils.get_metal_mask(features["particle_type"])
            noise_mask = (
                non_kinematic_mask.to(sampled_noise.dtype).unsqueeze(-1).unsqueeze(-1)
            )

            anchor_plane_mask = utils.get_anchor_z_mask(features["particle_type"])
            noise_anchor_plane_mask = (
                anchor_plane_mask.to(sampled_noise.dtype).unsqueeze(-1).unsqueeze(-1)
            )
            zero_mask = torch.zeros(
                noise_anchor_plane_mask.shape, dtype=noise_anchor_plane_mask.dtype
            )
            noise_anchor_plane_mask = torch.cat(
                [noise_anchor_plane_mask, noise_anchor_plane_mask, zero_mask], axis=-1
            )

            noise_mask = torch.repeat_interleave(noise_mask, repeats=3, dim=-1)
            noise_mask += noise_anchor_plane_mask

        else:
            non_kinematic_mask = torch.logical_not(
                utils.get_kinematic_mask(particle_types).bool()
            )
            noise_mask = (
                non_kinematic_mask.to(sampled_noise.dtype).unsqueeze(-1).unsqueeze(-1)
            )

        sampled_noise *= noise_mask

        pred_target = model(
            next_positions=targets.to(device),
            position_sequence=inputs.to(device),
            position_sequence_noise=sampled_noise.to(device),
            n_particles_per_example=features["n_particles_per_example"].to(device),
            n_edges_per_example=features["n_edges_per_example"].to(device),
            senders=features["senders"].to(device),
            receivers=features["receivers"].to(device),
            predict_length=PREDICT_LENGTH,
            particle_types=features["particle_type"].to(device),
            global_context=features.get("step_context").to(device),
        )

        if optimizer is None:
            # first data need to inference the feature size
            device = torch.device(C.device if torch.cuda.is_available() else "cpu")
            rank_zero_logger.info(
                f"*******************device: {device} ****************"
            )
            # print("*******************device: {} ****************".format(device))
            # config optimizer
            message_passing_devices = ast.literal_eval(C.message_passing_devices)
            model.setMessagePassingDevices(message_passing_devices)
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
            if C.fp16:
                # double check if amp installed
                try:
                    from apex import amp

                    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
                except ImportError as e:
                    print("Apex package not available -> ", e)
                    exit()

            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=0.1, verbose=True
            )
            decay_steps = int(5e6)
            # input feature size is dynamic, so need to forward model in CPU before loading into GPU
            # first step is forwarded in CPU, so skip the first step
            continue

        pred_acceleration, target_acceleration = pred_target
        # Calculate the L2 loss and mask out loss on kinematic particles
        loss = (pred_acceleration - target_acceleration) ** 2

        decay_fators_1 = torch.DoubleTensor(
            [math.pow(C.loss_decay_factor, i) for i in range(PREDICT_LENGTH)]
        ).to(device)
        decay_fators_3 = torch.repeat_interleave(decay_fators_1, repeats=3)

        loss = loss * decay_fators_3  # torch.Size([num_nodes, input_dim])
        loss = torch.sum(loss, dim=-1)  # torch.Size([num_nodes])

        if C.loss.startswith("anchor"):
            rank_zero_logger.info("processing anchor loss\n\n")
            # print("processing anchor loss\n\n")
            # omit anchor point in loss
            non_kinematic_mask = (
                torch.logical_not(utils.get_kinematic_mask(particle_types))
                .to(torch.bool)
                .to(device)
            )
            num_non_kinematic = torch.sum(non_kinematic_mask)

            loss = torch.where(
                non_kinematic_mask,
                loss,
                torch.zeros(loss.shape, dtype=inputs.dtype).to(device),
            )
            loss = torch.sum(loss) / torch.sum(num_non_kinematic)

            # compute the loss in z-axis of anchor plane points
            loss_plane = pred_acceleration[..., 2] ** 2
            decay_fator = torch.DoubleTensor(
                [math.pow(LOSS_DECAY_FACTOR, i) for i in range(1)]
            ).to(device)
            loss_plane = loss_plane * decay_fator

            anchor_plane_mask = anchor_plane_mask.to(torch.bool).to(device)
            num_anchor_plane = torch.sum(anchor_plane_mask)

            loss_plane = torch.where(
                anchor_plane_mask,
                loss_plane,
                torch.zeros(loss_plane.shape, dtype=inputs.dtype).to(device),
            )
            loss_plane = torch.sum(loss_plane) / torch.sum(num_anchor_plane)
            rank_zero_logger.info(f"loss: {loss}, loss_plane: {loss_plane}")

            loss = loss + C.l_plane * loss_plane

            if C.loss == "anchor_me":
                loss_l1 = torch.nn.functional.l1_loss(
                    pred_acceleration * decay_fators_3,
                    target_acceleration * decay_fators_3,
                )

                loss = loss + C.l_me * loss_l1

        elif C.loss.startswith("weighted"):
            loss = utils.weighted_square_error(
                pred_acceleration, target_acceleration, device
            )

            if C.loss == "weighted_anchor":
                loss_plane = pred_acceleration[..., 2] ** 2

                anchor_plane_mask = anchor_plane_mask.to(torch.bool).to(device)
                num_anchor_plane = torch.sum(anchor_plane_mask)
                loss_plane = torch.where(
                    anchor_plane_mask,
                    loss_plane,
                    torch.zeros(loss_plane.shape, dtype=inputs.dtype).to(device),
                )

                loss_plane = torch.sum(loss_plane) / torch.sum(num_anchor_plane)
                rank_zero_logger.info(f"loss: {loss}, loss_plane: {loss_plane}")
                loss = loss + C.l_plane * loss_plane

        elif C.loss == "correlation":
            """
            Compute the correlation of random neighboring point pairs
            to be optimized:
            - todo: get random surface point id list
            - todo: fix the pid list for each build
            """
            rank_zero_logger.info("processing correlation loss\n\n")

            loss_corr_factor = 1
            k = 100  # OR 1/ 100 * particle num, whichever smaller

            pid_list = [pid for pid in range(target_acceleration.shape[0])]
            random_pids = random.choices(pid_list, k=k)

            loss_corr = 0
            for idx_i in range(len(random_pids)):
                for idx_j in range(idx_i, len(random_pids)):
                    i, j = random_pids[idx_i], random_pids[idx_j]

                    corr_gt = torch.nn.functional.cosine_similarity(
                        target_acceleration[i], target_acceleration[j], dim=0
                    )
                    corr_pred = torch.nn.functional.cosine_similarity(
                        pred_acceleration[i], pred_acceleration[j], dim=0
                    )

                    loss_corr_ = (corr_gt - corr_pred) ** 2
                    loss_corr += loss_corr_

            loss_corr /= k**2

            non_kinematic_mask = non_kinematic_mask.to(torch.bool).to(device)
            num_non_kinematic = torch.sum(non_kinematic_mask)
            loss = torch.where(
                non_kinematic_mask,
                loss,
                torch.zeros(loss.shape, dtype=loss.dtype).to(device),
            )
            loss = torch.sum(loss) / torch.sum(num_non_kinematic)

            loss = loss + (loss_corr_factor * loss_corr)

        elif C.loss == "me":
            # adding the L1 loss component with weight defined by "C.l_me"
            rank_zero_logger.info("processing ME loss\n\n")
            loss_l1 = torch.nn.functional.l1_loss(
                pred_acceleration, target_acceleration
            )
            loss_l1 = loss_l1 * decay_fators_3
            loss_l1 = torch.sum(loss_l1, dim=-1)

            non_kinematic_mask = non_kinematic_mask.to(torch.bool).to(device)
            num_non_kinematic = torch.sum(non_kinematic_mask)
            loss = torch.where(
                non_kinematic_mask,
                loss,
                torch.zeros(loss.shape, dtype=loss.dtype).to(device),
            )
            loss = torch.sum(loss) / torch.sum(num_non_kinematic)

            loss = loss + C.l_me * loss_l1

        else:
            # standard loss with applying mask
            non_kinematic_mask = non_kinematic_mask.to(torch.bool).to(device)
            num_non_kinematic = torch.sum(non_kinematic_mask)
            loss = torch.where(
                non_kinematic_mask,
                loss,
                torch.zeros(loss.shape, dtype=loss.dtype).to(device),
            )
            loss = torch.sum(loss) / torch.sum(num_non_kinematic)

        rank_zero_logger.info(f"loss: {loss}")
        # back propogation
        optimizer.zero_grad()
        if C.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        running_loss += loss.item()

        step += 1

        if step % decay_steps == 0:
            scheduler.step()

        if step % 10 == 0:
            mean_loss = round(running_loss / 10, 5)
            writer.add_scalar("loss", mean_loss, step)
            writer.flush()

            running_loss = 0.0

        if step % 50 == 0:
            model.eval()
            with torch.no_grad():
                test_loss = 0.0
                position_loss = 0.0
                for j in range(C.eval_steps):
                    features, targets = next(testDataset)
                    # test inference features.get('step_context') shape:  torch.Size([2, 5])

                    predicted_positions = model.inference(
                        position_sequence=features["position"].to(device),
                        n_particles_per_example=features["n_particles_per_example"].to(
                            device
                        ),
                        n_edges_per_example=features["n_edges_per_example"].to(device),
                        senders=features["senders"].to(device),
                        receivers=features["receivers"].to(device),
                        predict_length=PREDICT_LENGTH,
                        particle_types=features["particle_type"].to(device),
                        global_context=features.get("step_context").to(device),
                    )
                    inputs = features["position"]
                    sampled_noise = torch.zeros(inputs.shape, dtype=inputs.dtype)
                    # sampled_noise = model.get_random_walk_noise_for_position_sequence(inputs, noise_std_last_step=FLAGS.noise_std)

                    pred_target = model(
                        next_positions=targets.to(device),
                        position_sequence=inputs.to(device),
                        position_sequence_noise=sampled_noise.to(device),
                        n_particles_per_example=features["n_particles_per_example"].to(
                            device
                        ),
                        n_edges_per_example=features["n_edges_per_example"].to(device),
                        senders=features["senders"].to(device),
                        receivers=features["receivers"].to(device),
                        predict_length=PREDICT_LENGTH,
                        particle_types=features["particle_type"].to(device),
                        global_context=features.get("step_context").to(device),
                    )

                    test_mse = torch.nn.functional.mse_loss(*pred_target)
                    p_mse = torch.nn.functional.mse_loss(
                        predicted_positions, targets.to(device)
                    )
                    test_loss += test_mse.item()
                    position_loss += p_mse.item()

                writer.add_scalar("loss_mse", test_loss, step)
                writer.add_scalar("position_mse", position_loss, step)
                writer.flush()

                if test_loss < best_loss:
                    torch.save(
                        model.state_dict(),
                        os.path.join(
                            C.model_path_vfgn,
                            "model_loss-{:.2E}_step-{}.pt".format(test_loss, step),
                        ),
                    )
                    best_loss = test_loss
            model.train()

    writer.close()


def Test(rank_zero_logger, dist):
    """
    Executes the testing phase for a graph-based model, generating and
    storing predictions.
    """
    # config test dataset
    dataset = GraphDataset(
        # size=C.num_steps,
        mode="rollout",
        split=C.eval_split,
        data_path=C.data_path,
        batch_size=C.batch_size,
    )

    metadata = _read_metadata(C.data_path)
    acceleration_stats = Stats(
        torch.DoubleTensor(cast(metadata["acc_mean"])),
        torch.DoubleTensor(utils._combine_std(cast(metadata["acc_std"]), C.noise_std)),
    )
    velocity_stats = Stats(
        torch.DoubleTensor(cast(metadata["vel_mean"])),
        torch.DoubleTensor(utils._combine_std(cast(metadata["vel_std"]), C.noise_std)),
    )
    context_stats = Stats(
        torch.DoubleTensor(cast(metadata["context_mean"])),
        torch.DoubleTensor(
            utils._combine_std(cast(metadata["context_std"]), C.noise_std)
        ),
    )

    normalization_stats = {
        "acceleration": acceleration_stats,
        "velocity": velocity_stats,
        "context": context_stats,
    }

    if not C.version_modulus:
        model = LearnedSimulatorBeta(
            num_dimensions=metadata["dim"] * PREDICT_LENGTH,
            num_seq=INPUT_SEQUENCE_LENGTH,
            boundaries=torch.DoubleTensor(metadata["bounds"]),
            num_particle_types=NUM_PARTICLE_TYPES,
            particle_type_embedding_size=16,
            normalization_stats=normalization_stats,
        )
    else:
        model = LearnedSimulator(
            num_dimensions=metadata["dim"] * PREDICT_LENGTH,
            num_seq=INPUT_SEQUENCE_LENGTH,
            boundaries=torch.DoubleTensor(metadata["bounds"]),
            num_particle_types=NUM_PARTICLE_TYPES,
            particle_type_embedding_size=16,
            normalization_stats=normalization_stats,
        )

    loaded = False
    example_index = 0
    device = "cpu"
    with torch.no_grad():
        for features, targets in tqdm(dataset):
            if loaded is False:
                # input feature size is dynamic, so need to forward model in CPU before loading into GPU
                global_context = features["step_context"].to(device)
                if global_context is None:
                    global_context_step = None
                else:
                    global_context_step = global_context[:-1]
                    global_context_step = torch.reshape(global_context_step, [1, -1])

                model.inference(
                    position_sequence=features["position"][
                        :, 0:INPUT_SEQUENCE_LENGTH
                    ].to(device),
                    n_particles_per_example=features["n_particles_per_example"].to(
                        device
                    ),
                    n_edges_per_example=features["n_edges_per_example"].to(device),
                    senders=features["senders"].to(device),
                    receivers=features["receivers"].to(device),
                    predict_length=PREDICT_LENGTH,
                    particle_types=features["particle_type"].to(device),
                    global_context=global_context_step.to(device),
                )
                model.load_state_dict(torch.load(C.model_path_vfgn))
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                rank_zero_logger.info(f"device: {device}")
                # config optimizer
                # todo: check msg passing device
                model.setMessagePassingDevices(["cuda:0"])
                model = model.to(device)
                model.eval()
                loaded = True

            initial_positions = features["position"][:, :INPUT_SEQUENCE_LENGTH].to(
                device
            )
            ground_truth_positions = features["position"][:, INPUT_SEQUENCE_LENGTH:].to(
                device
            )
            global_context = features["step_context"].to(device)
            rank_zero_logger.info(
                f"\n initial_positions shape: {initial_positions.shape}"
            )
            rank_zero_logger.info(
                f"\n ground_truth_positions shape: {ground_truth_positions.shape}"
            )

            num_steps = ground_truth_positions.shape[1]

            current_positions = initial_positions
            updated_predictions = []

            start_time = time.time()
            rank_zero_logger.info(f"start time: {start_time}\n")

            for step in range(num_steps):
                rank_zero_logger.info(f"start predictiong step: {step}")
                if global_context is None:
                    global_context_step = None
                else:
                    read_step_context = global_context[: step + INPUT_SEQUENCE_LENGTH]
                    zero_pad = torch.zeros(
                        [global_context.shape[0] - read_step_context.shape[0] - 1, 1],
                        dtype=features["step_context"].dtype,
                    ).to(device)

                    global_context_step = torch.cat([read_step_context, zero_pad], 0)
                    global_context_step = torch.reshape(global_context_step, [1, -1])

                predict_positions = model.inference(
                    position_sequence=current_positions.to(device),
                    n_particles_per_example=features["n_particles_per_example"].to(
                        device
                    ),
                    n_edges_per_example=features["n_edges_per_example"].to(device),
                    senders=features["senders"].to(device),
                    receivers=features["receivers"].to(device),
                    predict_length=PREDICT_LENGTH,
                    particle_types=features["particle_type"].to(device),
                    global_context=global_context_step.to(device),
                )

                kinematic_mask = (
                    utils.get_kinematic_mask(features["particle_type"])
                    .to(torch.bool)
                    .to(device)
                )
                positions_ground_truth = ground_truth_positions[:, step]

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

                updated_predictions.append(next_position)
                if C.rollout_refine is False:
                    # False: rollout the predictions
                    current_positions = torch.cat(
                        [current_positions[:, 1:], next_position.unsqueeze(1)], axis=1
                    )
                else:
                    # True: single-step prediction for all steps
                    current_positions = torch.cat(
                        [current_positions[:, 1:], positions_ground_truth.unsqueeze(1)],
                        axis=1,
                    )

            updated_predictions = torch.stack(updated_predictions)
            rank_zero_logger.info(
                f"\n updated_predictions shape: {updated_predictions.shape}"
            )
            rank_zero_logger.info(
                f"\n ground_truth_positions shape: {ground_truth_positions.shape}"
            )

            initial_positions_list = initial_positions.cpu().numpy().tolist()
            updated_predictions_list = updated_predictions.cpu().numpy().tolist()
            ground_truth_positions_list = ground_truth_positions.cpu().numpy().tolist()
            particle_types_list = features["particle_type"].cpu().numpy().tolist()
            global_context_list = global_context.cpu().numpy().tolist()

            rollout_op = {
                "initial_positions": initial_positions_list,
                "predicted_rollout": updated_predictions_list,
                "ground_truth_rollout": ground_truth_positions_list,
                "particle_types": particle_types_list,
                "global_context": global_context_list,
            }

            # Add a leading axis, since Estimator's predict method insists that all
            # tensors have a shared leading batch axis fo the same dims.
            # rollout_op = tree.map_structure(lambda x: x.numpy(), rollout_op)

            rollout_op["metadata"] = metadata
            filename = f"rollout_{C.eval_split}_{example_index}.json"
            filename = os.path.join(C.output_path, filename)
            if not os.path.exists(C.output_path):
                os.makedirs(C.output_path)
            with open(filename, "w") as file_object:
                json.dump(rollout_op, file_object)

            example_index += 1
            rank_zero_logger.info(f"prediction time: {time.time()-start_time}\n")


def main(_):
    """
    Triggers the train or test phase based on the configuration.
    """
    # initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    # save constants to JSON file
    # todo: test the disk.rank init and save
    # if dist.rank == 0:
    #     print('check main', C.ckpt_path)
    #     os.makedirs(C.ckpt_path, exist_ok=True)
    #     with open(
    #             os.path.join(C.ckpt_path, C.ckpt_name.replace(".pt", ".json")), "w"
    #     ) as json_file:
    #         json_file.write(C.json(indent=4))

    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)  # Rank 0 logger

    if C.mode == "train":
        Train(rank_zero_logger, dist)
    else:
        Test(rank_zero_logger, dist)


if __name__ == "__main__":
    # tf.disable_v2_behavior()
    LaunchLogger.initialize()  # Modulus launch logger
    logger = PythonLogger("main")  # General python logger
    logger.file_logging()

    app.run(main)

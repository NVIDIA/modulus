# © Copyright 2023 HP Development Company, L.P.
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


import collections
import functools
import json
import os
import tree
from absl import app
from absl import flags
# import reading_utils
import reading_utils as reading_utils

import tensorflow as tf
import random
import time

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
# from graph_network_v2 import LearnedSimulator
from graph_network import LearnedSimulator

import numpy as np
import math
import pickle
from apex import amp
import ast

flags.DEFINE_enum(
    'mode', 'train', ['train', 'eval', 'eval_rollout'],
    help='Train model, one step evaluation or rollout evaluation.')
flags.DEFINE_enum('eval_split', 'test', ['train', 'valid', 'test'],
                  help='Split to use when running evaluation.')
flags.DEFINE_string('data_path', '../../../../data/large-time-step/step_100', help='The dataset directory.')
flags.DEFINE_string('meta1', 'step100_s1', help='The dataset directory.')    # recalculated meta: meta1, meta2, meta3
flags.DEFINE_string('meta2', 'step100_s2', help='The dataset directory.')

flags.DEFINE_integer('batch_size', 2, help='The batch size.')
flags.DEFINE_integer('num_steps', int(2e7), help='Number of steps of training.')
flags.DEFINE_integer('eval_steps', 1, help='Number of steps of evaluation.')
flags.DEFINE_float('noise_std', 6.7e-4, help='The std deviation of the noise.')

flags.DEFINE_string('model_path_s1', 'models/step100_s1_weighted/model_loss-1.95E-06_step-154700.pt',
                    help=('The path for saving checkpoints of the model. '
                          'Defaults to a temporary directory.'))
flags.DEFINE_string('model_path_s2', 'models/step100_s2_lr5_weighted/model_loss-7.16E-05_step-135850.pt',
                    help='path to stage-2 model')

flags.DEFINE_string('output_path', None,
                    help='The path for saving outputs (e.g. rollouts).')

flags.DEFINE_enum('loss', 'standard', ['standard', 'anchor', 'me', 'correlation', 'anchor_me'],
                  help='loss type.')

flags.DEFINE_float('l_plane', 10, help='The scale factor of anchor plane loss.')
flags.DEFINE_float('l_me', 1, help='The scale factor of me loss.')

flags.DEFINE_integer('prefetch_buffer_size', 100, help="")
flags.DEFINE_string('device', 'cuda:0',
                    help='The device to training.')

flags.DEFINE_string('message_passing_devices',"['cuda:0']",help="The devices for message passing")
flags.DEFINE_bool('fp16',False,help='Training with mixed precision.')
flags.DEFINE_bool('rollout_refine',True, help='Use ground truth value as input in every steps')

FLAGS = flags.FLAGS


class Stats:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self


device = "cpu"

INPUT_SEQUENCE_LENGTH = 5#10  # So we can calculate the last 5 velocities.
PREDICT_LENGTH = 1#5
LOSS_DECAY_FACTOR = 0.6

NUM_PARTICLE_TYPES = 3
KINEMATIC_PARTICLE_ID = 0   # anchor point
METAL_PARTICLE_ID = 2  # normal particles
ANCHOR_PLANE_PARTICLE_ID = 1    # anchor plane


def get_kinematic_mask(particle_types):
    """Returns a boolean mask, set to true for kinematic (obstacle) particles."""
    # return tf.equal(particle_types, KINEMATIC_PARTICLE_ID)
    # return size: num_particles_in_batch

    return particle_types == torch.ones(particle_types.shape) * KINEMATIC_PARTICLE_ID


def get_metal_mask(particle_types):
    # get free particles
    return particle_types == torch.ones(particle_types.shape) * METAL_PARTICLE_ID


def get_anchor_z_mask(particle_types):
    # get anchor plane particles
    return particle_types == torch.ones(particle_types.shape) * ANCHOR_PLANE_PARTICLE_ID


def cos_theta(p1, p2):
    return (torch.dot(p1, p2)) / ((torch.sqrt(torch.dot(p1, p1))) * (math.sqrt(torch.dot(p2, p2))))


def prepare_inputs(tensor_dict):
    """Prepares a single stack of inputs by calculating inputs and targets.

    Computes n_particles_per_example, which is a tensor that contains information
    about how to partition the axis - i.e. which nodes belong to which graph.

    Adds a batch axis to `n_particles_per_example` and `step_context` so they can
    later be batched using `batch_concat`. This batch will be the same as if the
    elements had been batched via stacking.

    Note that all other tensors have a variable size particle axis,
    and in this case they will simply be concatenated along that
    axis.



    Args:
      tensor_dict: A dict of tensors containing positions, and step context (
      if available).

    Returns:
      A tuple of input features and target positions.

    """
    # Position is encoded as [sequence_length, num_particles, dim] but the model
    # expects [num_particles, sequence_length, dim].
    predict_length = PREDICT_LENGTH

    pos = tensor_dict['position']

    pos = tf.transpose(pos, perm=[1, 0, 2])

    # The target position is the final step of the stack of positions.
    target_position = pos[:, -predict_length:]

    # Remove the target from the input.
    tensor_dict['position'] = pos[:, :-predict_length]

    # Compute the number of particles per example.
    num_particles = tf.shape(pos)[0]
    # Add an extra dimension for stacking via concat.
    tensor_dict['n_particles_per_example'] = num_particles[tf.newaxis]

    num_edges = tf.shape(tensor_dict['senders'])[0]
    tensor_dict['n_edges_per_example'] = num_edges[tf.newaxis]

    if 'step_context' in tensor_dict:
        # Take the input global context. We have a stack of global contexts,
        # and we take the penultimate since the final is the target.
        tensor_dict['step_context'] = tensor_dict['step_context'][-predict_length - 1]
        # Add an extra dimension for stacking via concat.
        tensor_dict['step_context'] = tensor_dict['step_context'][tf.newaxis]

    return tensor_dict, target_position


def prepare_rollout_inputs(context, features):
    """Prepares an inputs trajectory for rollout."""
    out_dict = {**context}
    # Position is encoded as [sequence_length, num_particles, dim] but the model
    # expects [num_particles, sequence_length, dim].
    pos = tf.transpose(features['position'], [1, 0, 2])
    # The target position is the final step of the stack of positions.
    target_position = pos[:, -1]
    # Remove the target from the input.
    out_dict['position'] = pos[:, :-1]
    # Compute the number of nodes

    out_dict['n_particles_per_example'] = [tf.shape(pos)[0]]
    out_dict['n_edges_per_example'] = [tf.shape(context['senders'])[0]]
    if 'step_context' in features:
         out_dict['step_context'] = tf.dtypes.cast(features['step_context'], tf.float64)

    out_dict['is_trajectory'] = tf.constant([True], tf.bool)
    return out_dict, target_position


def batch_concat(dataset, batch_size):
    """We implement batching as concatenating on the leading axis."""

    # We create a dataset of datasets of length batch_size.
    windowed_ds = dataset.window(batch_size)

    # The plan is then to reduce every nested dataset by concatenating. We can
    # do this using tf.data.Dataset.reduce. This requires an initial state, and
    # then incrementally reduces by running through the dataset

    # Get initial state. In this case this will be empty tensors of the
    # correct shape.
    initial_state = tree.map_structure(
        lambda spec: tf.zeros(  # pylint: disable=g-long-lambda
            shape=[0] + spec.shape.as_list()[1:], dtype=spec.dtype),
        dataset.element_spec)

    # We run through the nest and concatenate each entry with the previous state.
    def reduce_window(initial_state, ds):
        return ds.reduce(initial_state, lambda x, y: tf.concat([x, y], axis=0))

    return windowed_ds.map(
        lambda *x: tree.map_structure(reduce_window, initial_state, x))


def _read_metadata(data_path):
    with open(os.path.join(data_path, 'metadata.json'), 'rt') as fp:
        print(os.path.join(data_path, 'metadata.json'))
        return json.loads(fp.read())


def get_input_fn(data_path, batch_size, mode, split):
    """Gets the learning simulation input function for tf.estimator.Estimator.

    Args:
      data_path: the path to the dataset directory.
      batch_size: the number of graphs in a batch.
      mode: either 'one_step_train', 'one_step' or 'rollout'
      split: either 'train', 'valid' or 'test.

    Returns:
      The input function for the learning simulation model.
    """

    def input_fn():
        """Input function for learning simulation."""
        # Loads the metadata of the dataset.
        metadata = _read_metadata(data_path)
        # Create a tf.data.Dataset from the TFRecord.
        ds = tf.data.TFRecordDataset([os.path.join(data_path, f'{split}.tfrecord')])
        ds = ds.map(functools.partial(
            reading_utils.parse_serialized_simulation_example, metadata=metadata))
        # ds = ds.repeat()
        # ds = ds.shuffle(128)
        if mode.startswith('one_step'):
            # Splits an entire trajectory into chunks of 7 steps.
            # Previous 5 velocities, current velocity and target.
            split_with_window = functools.partial(
                reading_utils.split_trajectory,
                window_length=INPUT_SEQUENCE_LENGTH, predict_length=PREDICT_LENGTH)
            ds = ds.flat_map(split_with_window)
            # Splits a chunk into input steps and target steps
            ds = ds.map(prepare_inputs)
            # If in train mode, repeat dataset forever and shuffle.
            if mode == 'one_step_train':
                ds.prefetch(buffer_size=FLAGS.prefetch_buffer_size)
                ds = ds.repeat()
                ds = ds.shuffle(512)

        # Custom batching on the leading axis.
            ds = batch_concat(ds, batch_size)
        elif mode == 'rollout':
            # Rollout evaluation only available for batch size 1
            assert batch_size == 1
            ds = ds.map(prepare_rollout_inputs)
        else:
            raise ValueError(f'mode: {mode} not recognized')

        return ds

    return input_fn


def infer_stage(model, features, global_context, current_positions,
                num_steps, ground_truth_positions, updated_predictions, sequence_length,
                metadata_1=None, metadata_2=None, renorm=False):
    len_predicted = len(updated_predictions)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("infer_stage device: ", device)
    print("infer_stage, global_context shape: ", global_context.shape)
    print(f"start infer {num_steps} steps ........ \n")

    for step in range(num_steps):
        if global_context is None:
            global_context_step = None
        else:
            read_step_context = global_context[:step+INPUT_SEQUENCE_LENGTH]
            if read_step_context.shape[0] <= (sequence_length-1):
                zero_pad = torch.zeros([sequence_length-read_step_context.shape[0]-1, 1], dtype=features['step_context'].dtype).to(device)
                # global_context_step = torch.concat([read_step_context, zero_pad], 0)
                global_context_step = torch.cat([read_step_context, zero_pad], 0)
            else:
                global_context_step = read_step_context[-(sequence_length-1):]
            global_context_step = torch.reshape(global_context_step,[1, -1])
            print("global_context_step shape: ",global_context_step.shape, global_context_step)

        predict_positions = model.inference(
            position_sequence=current_positions.to(device),
            n_particles_per_example=features['n_particles_per_example'].to(device),
            n_edges_per_example=features['n_edges_per_example'].to(device),
            senders=features['senders'].to(device),
            receivers=features['receivers'].to(device),
            predict_length=PREDICT_LENGTH,
            particle_types=features['particle_type'].to(device),
            global_context= global_context_step.to(device)
        )

        kinematic_mask = get_kinematic_mask(features['particle_type']).to(torch.bool).to(device)
        positions_ground_truth = ground_truth_positions[:, step+len_predicted]

        predict_positions = predict_positions[:, 0].squeeze(1)
        kinematic_mask = torch.repeat_interleave(kinematic_mask,repeats=predict_positions.shape[-1])
        kinematic_mask = torch.reshape(kinematic_mask,[-1,predict_positions.shape[-1]])

        next_position = torch.where(kinematic_mask, positions_ground_truth, predict_positions)
        print("ground truth position: ", (step+len_predicted))

        updated_predictions.append(next_position)
        # print("current_positions shape: ", current_positions.shape)


        if FLAGS.rollout_refine is False:
            # False: rollout the predictions
            current_positions = torch.cat([current_positions[:, 1:], next_position.unsqueeze(1)], axis=1)
        else:
            # True: single-step prediction for all steps
            current_positions = torch.cat([current_positions[:,1:],positions_ground_truth.unsqueeze(1)], axis=1)

        # renorm
        if renorm and step == (num_steps-1):
            current_positions = ((current_positions * metadata_1['pos_std'] + metadata_1['pos_mean'])
                                 - metadata_2['pos_mean']) / metadata_2['pos_std']

    print("updated_predictions len: ", len(updated_predictions))
    print(f"finished predicting stage-: {num_steps} steps\n\n")

    return current_positions, updated_predictions


def load_stage_model(model, model_path,
                     features, global_context_step,
                     sequence_length):
    device = 'cpu'
    global_context_step = global_context_step[:, :sequence_length]
    print("load_stage_model, global_context_step: ", global_context_step.shape)

    model.inference(
        position_sequence=features['position'][:, 0:INPUT_SEQUENCE_LENGTH].to(device),
        n_particles_per_example=features['n_particles_per_example'].to(device),
        n_edges_per_example=features['n_edges_per_example'].to(device),
        senders=features['senders'].to(device),
        receivers=features['receivers'].to(device),
        predict_length=PREDICT_LENGTH,
        particle_types=features['particle_type'].to(device),
        global_context=global_context_step.to(device)
    )
    model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    # config optimizer
    model.setMessagePassingDevices(['cuda:0'])
    model = model.to(device)
    model.eval()

    print("\n loaded model ", model_path)

    return model


def _combine_std(std_x, std_y):
    return np.sqrt(std_x ** 2 + std_y ** 2)


def tf2torch(t):
    t = torch.from_numpy(t.numpy())
    return t

def torch2tf(t):
    t = tf.convert_to_tensor(t.cpu().numpy())
    return t


class GraphDataset:
    def __init__(self, size=1000, mode='one_step_train', split='train'):
        self.dataset = get_input_fn(FLAGS.data_path, FLAGS.batch_size,
                                    mode=mode, split=split)()
        self.dataset = iter(self.dataset)
        self.size = size
        self.pos = 0

    def __len__(self):
        return self.size

    def __next__(self):
        if self.pos< self.size:
            features, targets = self.dataset.get_next()
            for key in features:
                if key != "key":
                    features[key] = tf2torch(features[key])

            targets = tf2torch(targets)
            self.pos += 1
            return features, targets
        else:
            raise StopIteration

    def __iter__(self):
        return self


cast = lambda v: np.array(v, dtype=np.float64)


def Test():
    dataset = GraphDataset(mode='rollout', split=FLAGS.eval_split)

    metadat_1 = _read_metadata(os.path.join(FLAGS.data_path, FLAGS.meta1))
    print("normalization_stats: ", metadat_1)

    metadat_2 = _read_metadata(os.path.join(FLAGS.data_path, FLAGS.meta2))

    acceleration_stats = Stats(torch.DoubleTensor(cast(metadat_1['acc_mean'])),
                               torch.DoubleTensor(_combine_std(cast(metadat_1['acc_std']), FLAGS.noise_std)))
    velocity_stats = Stats(torch.DoubleTensor(cast(metadat_1['vel_mean'])),
                           torch.DoubleTensor(_combine_std(cast(metadat_1['vel_std']), FLAGS.noise_std)))
    context_stats = Stats(torch.DoubleTensor(cast(metadat_1['context_mean'])),
                          torch.DoubleTensor(_combine_std(cast(metadat_1['context_std']), FLAGS.noise_std)))
    sequence_length_s1 = int(metadat_1['sequence_length'])

    normalization_stats = {'acceleration': acceleration_stats, 'velocity': velocity_stats, 'context': context_stats}

    acceleration_stats_2 = Stats(torch.DoubleTensor(cast(metadat_2['acc_mean'])),
                               torch.DoubleTensor(_combine_std(cast(metadat_2['acc_std']), FLAGS.noise_std)))
    velocity_stats_2 = Stats(torch.DoubleTensor(cast(metadat_2['vel_mean'])),
                           torch.DoubleTensor(_combine_std(cast(metadat_2['vel_std']), FLAGS.noise_std)))
    context_stats_2 = Stats(torch.DoubleTensor(cast(metadat_2['context_mean'])),
                          torch.DoubleTensor(_combine_std(cast(metadat_2['context_std']), FLAGS.noise_std)))
    sequence_length_s2 = int(metadat_2['sequence_length'])
    normalization_stats_s2 = {'acceleration': acceleration_stats_2, 'velocity': velocity_stats_2, 'context': context_stats_2}
    print("\nnormalization_stats_s2: ", metadat_2)


    model_s1 = LearnedSimulator(num_dimensions=metadat_1['dim'] * PREDICT_LENGTH, num_seq=INPUT_SEQUENCE_LENGTH,
                             boundaries=torch.DoubleTensor(metadat_1['bounds']),
                             num_particle_types=NUM_PARTICLE_TYPES, particle_type_embedding_size=16,
                             normalization_stats=normalization_stats)

    model_s2 = LearnedSimulator(num_dimensions=metadat_1['dim'] * PREDICT_LENGTH, num_seq=INPUT_SEQUENCE_LENGTH,
                                boundaries=torch.DoubleTensor(metadat_1['bounds']),
                                num_particle_types=NUM_PARTICLE_TYPES, particle_type_embedding_size=16,
                                normalization_stats=normalization_stats_s2)


    loaded = False
    example_index =0
    device = 'cpu'

    #  # index for the step 100 model, 2 stages
    #     start_index, end_index = 0, 1321
    #     start_index, end_index = 1000, 3393
    START_INDEX_S1, END_INDEX_S1 = 0//100, 1321//100
    START_INDEX_S2, END_INDEX_S2 = 1321//100, 3393//100

    with torch.no_grad():
        for features, targets in tqdm(dataset):
            if loaded is False:
                global_context = features['step_context'].to(device)
                if global_context is None:
                    global_context_step = None
                else:
                    # global_context_step = global_context[
                    #     INPUT_SEQUENCE_LENGTH - 1].unsqueeze(-1)
                    global_context_step = global_context[:-1]
                    global_context_step = torch.reshape(global_context_step,[1, -1])

                ##### Load model from ckpts
                model_s1 = load_stage_model(model_s1, FLAGS.model_path_s1, features, global_context_step, sequence_length_s1)
                model_s2 = load_stage_model(model_s2, FLAGS.model_path_s2, features, global_context_step, sequence_length_s2)

                loaded = True

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print("device: ", device)

            ###### start prediction ######
            initial_positions = features['position'][:, 0:INPUT_SEQUENCE_LENGTH].to(device)
            ground_truth_positions = features['position'][:, INPUT_SEQUENCE_LENGTH:END_INDEX_S2].to(device)
            global_context = features['step_context'].to(device)

            current_positions = initial_positions
            updated_predictions = []
            start_time = time.time()
            print("start time: ", start_time)
            print("\n")

            ############ stage-1  ############
            print("device: ", device)
            num_steps_s1 = END_INDEX_S1 - START_INDEX_S1 - INPUT_SEQUENCE_LENGTH
            current_positions, updated_predictions = infer_stage(model_s1, features, global_context[:(sequence_length_s1+1)], current_positions,
                                                                 num_steps_s1, ground_truth_positions, updated_predictions,
                                                                 sequence_length_s1+1,
                                                                 metadat_1, metadat_2, True)
            print("updated_predictions len: ", len(updated_predictions))


            ############ stage-2  ############
            num_steps_s2 = END_INDEX_S2 - START_INDEX_S2
            current_positions, updated_predictions = infer_stage(model_s2, features, global_context[(sequence_length_s1-INPUT_SEQUENCE_LENGTH):], current_positions,
                                                                 num_steps_s2, ground_truth_positions, updated_predictions,
                                                                 sequence_length_s2+1)
            print("updated_predictions len: ", len(updated_predictions))


            ############ stage-transit  ############
            ## to be implemented

            end_time = time.time()
            print("prediction time: ", end_time-start_time)
            print("\n")


            # Store in pkl
            updated_predictions = torch.stack(updated_predictions)

            print("\n\n finished running all stages, initial, gt, predicted: ", initial_positions.shape,
                  ground_truth_positions.shape, updated_predictions.shape)

            initial_positions = torch2tf(initial_positions)
            updated_predictions = torch2tf(updated_predictions)
            ground_truth_positions = torch2tf(ground_truth_positions)
            particle_types = torch2tf(features['particle_type'])
            global_context = torch2tf(global_context)

            rollout_op = {
                'initial_positions': tf.transpose(initial_positions, [1, 0, 2]),
                'predicted_rollout': updated_predictions,
                'ground_truth_rollout': tf.transpose(ground_truth_positions, [1, 0, 2]),
                'particle_types': particle_types,
                'global_context': global_context
            }

            squared_error = (rollout_op['predicted_rollout'] -
                             rollout_op['ground_truth_rollout']) ** 2

            # Add a leading axis, since Estimator's predict method insists that all
            # tensors have a shared leading batch axis fo the same dims.
            rollout_op = tree.map_structure(lambda x: x.numpy(), rollout_op)

            rollout_op['metadat_1'] = metadat_1
            rollout_op['metadat_2'] = metadat_2
            filename = f'rollout_{FLAGS.eval_split}_{example_index}.pkl'
            filename = os.path.join(FLAGS.output_path, filename)
            if not os.path.exists(FLAGS.output_path):
                os.makedirs(FLAGS.output_path)
            with open(filename, 'wb') as file:
                pickle.dump(rollout_op, file)
            example_index+=1
            
            print(f"prediction time: {time.time()-start_time}\n")


def main(_):
    Test()

if __name__ == '__main__':
    # tf.disable_v2_behavior()
    app.run(main)

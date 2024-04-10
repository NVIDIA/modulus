import collections
import functools
import json
import os
import tree
from absl import app
from absl import flags
import reading_utils_m2 as reading_utils
import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
try:
    for device_ in physical_devices:
        tf.config.experimental.set_memory_growth(device_, True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

import random
import time

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
# from graph_network_global_m2 import LearnedSimulator      # this is the file for INPUT_SEQUENCE_LENGTH >= 3
# from graph_network_global_m2_posvel import LearnedSimulator   # this is the file for INPUT_SEQUENCE_LENGTH >= 2
from graph_network_global_m1 import LearnedSimulator
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
flags.DEFINE_string('data_path', None, help='The dataset directory.')
flags.DEFINE_integer('batch_size', 2, help='The batch size.')
flags.DEFINE_integer('num_steps', int(2e7), help='Number of steps of training.')
flags.DEFINE_integer('eval_steps', 1, help='Number of steps of evaluation.')
flags.DEFINE_float('noise_std', 6.7e-4, help='The std deviation of the noise.')
flags.DEFINE_string('model_path', None,
                    help=('The path for saving checkpoints of the model. '
                          'Defaults to a temporary directory.'))
flags.DEFINE_string('output_path', None,
                    help='The path for saving outputs (e.g. rollouts).')

flags.DEFINE_enum('loss', 'standard', ['standard', 'weighted', 'anchor', 'me',
                                       'correlation', 'anchor_me', 'weighted_anchor'],
                  help='loss type.')

# flags.DEFINE_bool('l_weighted', True, help='weighted loss compute to minimize the max node error')
flags.DEFINE_float('l_plane', 30, help='The scale factor of anchor plane loss.')
flags.DEFINE_float('l_me', 3, help='The scale factor of me loss.')

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

INPUT_SEQUENCE_LENGTH = 5 #5#10  # So we can calculate the last 5 velocities.
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


def weighted_square_error(y_pre,y, device):
    k=y_pre-y
    k=k.view(-1)
    k=torch.square(k)
    sorted, indices = torch.sort(k,descending=True)
    print("weight: ", sorted.size())
    n=sorted.size()[0]

    weights=[]
    dw=1.0/n
    for i in range(n):
        weights.append(dw)
        dw=dw*0.99
    weights=torch.FloatTensor(weights).to(device)

    out=weights*sorted

    out=torch.sum(out)
    print("weighted_square_error out: ", out)
    return out


def weighted_loss(loss_, device):
    # k=y_pre-y
    loss_=loss_.view(-1)
    # k=torch.square(k)
    sorted, indices = torch.sort(loss_,descending=True)
    n=sorted.size()[0]

    weights=[]
    dw=1.0/n
    for i in range(n):
        weights.append(dw)
        dw=dw*0.99
    weights=torch.FloatTensor(weights).to(device)

    out=weights*sorted

    out=torch.sum(out)
    print("out: ", out)
    return out


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
    print("prep inputs pos shape: ", pos.shape)
    # prep inputs pos shape:  (6, None, 3)
    # prep inputs pos shape:  (3, None, 3)

    pos = tf.transpose(pos, perm=[1, 0, 2])

    # The target position is the final step of the stack of positions.
    target_position = pos[:, -predict_length:]
    # print("target_position shape: ", target_position.shape)

    # Remove the target from the input.
    tensor_dict['position'] = pos[:, :-predict_length]
    # print("tensor_dict['position'] shape: ", tensor_dict['position'].shape)

    # Compute the number of particles per example.
    num_particles = tf.shape(pos)[0]
    # Add an extra dimension for stacking via concat.
    tensor_dict['n_particles_per_example'] = num_particles[tf.newaxis]

    num_edges = tf.shape(tensor_dict['senders'])[0]
    tensor_dict['n_edges_per_example'] = num_edges[tf.newaxis]

    if 'step_context' in tensor_dict:
        # Take the input global context. We have a stack of global contexts,
        # and we take the penultimate since the final is the target.

        # Method 0:
        # tensor_dict['step_context'] = tensor_dict['step_context'][-predict_length - 1]
        # print("take step_context at:  ", -predict_length - 1, tensor_dict['step_context'].shape)
        # # take step_context at:   -2 (1,)
        # # Add an extra dimension for stacking via concat.
        # tensor_dict['step_context'] = tensor_dict['step_context'][tf.newaxis]

        # Method 1:
        # tensor_dict['step_context'] = tensor_dict['step_context'][:-1]
        # # print("take step_context at:  ", tensor_dict['step_context'].shape, tensor_dict['step_context'])
        # tensor_dict['step_context'] = tf.reshape(tensor_dict['step_context'],[1, -1])
        # print("take step_context at:  ", tensor_dict['step_context'].shape, tensor_dict['step_context'])
        # # take step_context at:   (5, 1)

        # Method 2:
        # Tensor("strided_slice_6:0", shape=(32, 1), dtype=float64)
        tensor_dict['step_context'] = tf.reshape(tensor_dict['step_context'],[1, -1])
        print("take step_context at:  ", tensor_dict['step_context'].shape, tensor_dict['step_context'])

    print("prepare inputs, tensor_dict['step_context'] shape: ", tensor_dict['step_context'].shape)
    # take step_context at:   (1, 33) Tensor("Reshape:0", shape=(1, 33), dtype=float64)
    # prepare inputs, tensor_dict['step_context'] shape:  (1, 33)

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
    print("out_dict['position'] : ", out_dict['position'].shape)
    # Compute the number of nodes

    out_dict['n_particles_per_example'] = [tf.shape(pos)[0]]
    out_dict['n_edges_per_example'] = [tf.shape(context['senders'])[0]]
    if 'step_context' in features:
        out_dict['step_context'] = tf.dtypes.cast(features['step_context'], tf.float64)
        print("out_dict['step_context'] : ", out_dict['step_context'].shape)

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
        print("\n\ninit data set, path= ", data_path)
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


def _read_metadata(data_path):
    with open(os.path.join(data_path, 'metadata.json'), 'rt') as fp:
        return json.load(fp)


def _combine_std(std_x, std_y):
    # print("compute _combine_std")
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


def Train():
    # config dataset
    dataset = GraphDataset(size=FLAGS.num_steps)
    testDataset = GraphDataset(size=FLAGS.num_steps,split='test')

    # config model
    metadata = _read_metadata(FLAGS.data_path)
    acceleration_stats = Stats(torch.DoubleTensor(cast(metadata['acc_mean'])),
                               torch.DoubleTensor(_combine_std(cast(metadata['acc_std']), FLAGS.noise_std)))
    velocity_stats = Stats(torch.DoubleTensor(cast(metadata['vel_mean'])),
                           torch.DoubleTensor(_combine_std(cast(metadata['vel_std']), FLAGS.noise_std)))
    context_stats = Stats(torch.DoubleTensor(cast(metadata['context_mean'])),
                          torch.DoubleTensor(_combine_std(cast(metadata['context_std']), FLAGS.noise_std)))

    normalization_stats = {'acceleration': acceleration_stats, 'velocity': velocity_stats, 'context': context_stats}
    model = LearnedSimulator(num_dimensions=metadata['dim'] * PREDICT_LENGTH, num_seq=INPUT_SEQUENCE_LENGTH,
                             boundaries=torch.DoubleTensor(metadata['bounds']),
                             num_particle_types=NUM_PARTICLE_TYPES, particle_type_embedding_size=16,
                             normalization_stats=normalization_stats)

    #
    writer = SummaryWriter(log_dir=FLAGS.model_path)

    optimizer = None
    device = 'cpu'
    step = 0
    running_loss = 0.0
    best_loss = 1000.0

    for features, targets in tqdm(dataset):

        inputs = features['position']
        particle_types = features['particle_type']

        sampled_noise = model.get_random_walk_noise_for_position_sequence(inputs, noise_std_last_step=FLAGS.noise_std)
        if FLAGS.loss.startswith('anchor'):
            print("compute noise_mask")
            # if FLAGS.loss == 'anchor':

            non_kinematic_mask = get_metal_mask(features['particle_type'])
            noise_mask = non_kinematic_mask.to(sampled_noise.dtype).unsqueeze(-1).unsqueeze(-1)

            anchor_plane_mask = get_anchor_z_mask(features['particle_type'])
            noise_anchor_plane_mask = anchor_plane_mask.to(sampled_noise.dtype).unsqueeze(-1).unsqueeze(-1)
            zero_mask = torch.zeros(noise_anchor_plane_mask.shape, dtype=noise_anchor_plane_mask.dtype)
            noise_anchor_plane_mask = torch.cat([noise_anchor_plane_mask, noise_anchor_plane_mask, zero_mask], axis=-1)

            noise_mask = torch.repeat_interleave(noise_mask, repeats=3, dim=-1)
            noise_mask += noise_anchor_plane_mask

        else:
            non_kinematic_mask = torch.logical_not(get_kinematic_mask(particle_types).bool())
            noise_mask = non_kinematic_mask.to(sampled_noise.dtype).unsqueeze(-1).unsqueeze(-1)

        sampled_noise *= noise_mask

        # print("features.get('step_context') shape: ", features.get('step_context').shape)
        pred_target = model(
            next_positions=targets.to(device),
            position_sequence=inputs.to(device),
            position_sequence_noise=sampled_noise.to(device),
            n_particles_per_example=features['n_particles_per_example'].to(device),
            n_edges_per_example=features['n_edges_per_example'].to(device),
            senders=features['senders'].to(device),
            receivers=features['receivers'].to(device),
            predict_length=PREDICT_LENGTH,
            particle_types=features['particle_type'].to(device),
            global_context=features.get('step_context').to(device)
        )

        if optimizer is None:
            # first data need to inference the feature size
            device = torch.device(FLAGS.device if torch.cuda.is_available() else "cpu")
            print("*******************device: {} ****************".format(device))
            # config optimizer
            message_passing_devices=ast.literal_eval(FLAGS.message_passing_devices)
            model.setMessagePassingDevices(message_passing_devices)
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
            if FLAGS.fp16:
                model, optimizer = amp.initialize(model,optimizer,opt_level='O1')
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1, verbose=True)
            decay_steps = int(5e6)
            continue

        pred_acceleration, target_acceleration = pred_target
        print("pred_acceleration dim: ", pred_acceleration.shape)
        # Calculate the loss and mask out loss on kilrnematic particles/
        loss = (pred_acceleration - target_acceleration) ** 2  # L2 loss

        # decay_fators_1 dim:  torch.Size([5])
        decay_fators_1 = torch.DoubleTensor([math.pow(LOSS_DECAY_FACTOR, i) for i in range(PREDICT_LENGTH)]).to(device)
        # decay_fators_3 dim:  torch.Size([15])
        decay_fators_3 = torch.repeat_interleave(decay_fators_1, repeats=3)
        print("loss dtype: ", loss.dtype)   # loss dtype:  torch.float32

        loss = loss * decay_fators_3    # torch.Size([71424, 3])
        print("loss dim: ", loss.shape) # loss dim:  torch.Size([71424, 15])

        loss = torch.sum(loss, dim=-1)  # torch.Size([71424])

        ############ weighted_square_error
        # if FLAGS.l_weighted:
        #     weighted_square_error(pred_acceleration,target_acceleration, device)
        # exit()

        if FLAGS.loss.startswith('anchor'):
            # todo: need to update anchor loss
            # omit anchor point in loss
            non_kinematic_mask = torch.logical_not(get_kinematic_mask(particle_types)).to(torch.bool).to(device)
            num_non_kinematic = torch.sum(non_kinematic_mask)

            loss = torch.where(non_kinematic_mask, loss, torch.zeros(loss.shape, dtype=inputs.dtype).to(device))
            loss = torch.sum(loss) / torch.sum(num_non_kinematic)

            # Compute z-loss
            # zero_acceleration = model.get_normalized_acceleration(0, PREDICT_LENGTH)    # loss zero_acceleration:  torch.Size([15])

            # compute the loss in z aix of anchor plane points
            # loss_plane = torch.square(pred_acceleration - zero_acceleration)[..., 2::3]
            # print("loss plane: ", loss_plane.shape, torch.sum(torch.sum(loss_plane)))
            loss_plane = pred_acceleration[..., 2] ** 2
            decay_fator = torch.DoubleTensor([math.pow(LOSS_DECAY_FACTOR, i) for i in range(1)]).to(device)
            loss_plane = loss_plane * decay_fator
            # print("loss plane 2: ", loss_plane_2.shape, torch.sum(torch.sum(loss_plane_2)))

            anchor_plane_mask = anchor_plane_mask.to(torch.bool).to(device)
            num_anchor_plane = torch.sum(anchor_plane_mask)

            # loss_plane = loss_plane * decay_fators_1
            # loss_plane = torch.sum(loss_plane, dim=-1)
            # print("loss dtype: ", loss.dtype)
            # print("loss_plane dtype: ", loss_plane.dtype)
            loss_plane = torch.where(anchor_plane_mask, loss_plane,
                                     torch.zeros(loss_plane.shape, dtype=inputs.dtype).to(device))

            loss_plane = torch.sum(loss_plane) / torch.sum(num_anchor_plane)

            print(f"loss: {loss}, loss_plane: {loss_plane}")

            loss = loss + FLAGS.l_plane * loss_plane

            if FLAGS.loss == "anchor_me":
                # loss_l1 = torch.nn.functional.l1_loss(pred_acceleration, target_acceleration)
                # loss_l1 = loss_l1 * decay_fators_3
                # loss_l1 = torch.sum(loss_l1, dim=-1)

                loss_l1 = torch.nn.functional.l1_loss(pred_acceleration* decay_fators_3, target_acceleration* decay_fators_3)

                print("mse loss: ", loss)
                print("me loss: ", loss_l1)
                loss = loss + FLAGS.l_me * loss_l1

        elif FLAGS.loss.startswith("weighted"):
            loss = weighted_square_error(pred_acceleration,target_acceleration, device)

            if FLAGS.loss == "weighted_anchor":
                loss_plane = pred_acceleration[..., 2] ** 2

                anchor_plane_mask = anchor_plane_mask.to(torch.bool).to(device)
                num_anchor_plane = torch.sum(anchor_plane_mask)
                loss_plane = torch.where(anchor_plane_mask, loss_plane,
                                         torch.zeros(loss_plane.shape, dtype=inputs.dtype).to(device))

                loss_plane = torch.sum(loss_plane) / torch.sum(num_anchor_plane)

                print(f"loss: {loss}, loss_plane: {loss_plane}")
                loss = loss + FLAGS.l_plane * loss_plane

        elif FLAGS.loss == "correlation":
            # todo: simplify the equation for faster training
            print("processing correlation loss\n\n")

            # loss = (pred_acceleration - target_acceleration) ** 2  # L2 loss
            # loss = loss * decay_fators_3    # torch.Size([71424, 3]) mse loss
            loss_corr_factor = 1
            k = 100 # OR 1/ 100 * particle num, whichever smaller
            # todo: get random surface point id list
            # todo: fix the pid list for each build
            pid_list = [pid for pid in range(target_acceleration.shape[0])]
            random_pids = random.choices(pid_list, k=k)
            # print("random_pids: ", random_pids)
            # random.shuffle(pid_list)
            # random_pairs = np.array(pid_list)[:200].reshape(-1, 2)

            loss_corr = 0
            start_time = time.time()
            for idx_i in range(len(random_pids)):
                for idx_j in range(idx_i, len(random_pids)):
                    i, j = random_pids[idx_i], random_pids[idx_j]

                    corr_gt = torch.nn.functional.cosine_similarity(target_acceleration[i], target_acceleration[j], dim=0)
                    corr_pred = torch.nn.functional.cosine_similarity(pred_acceleration[i], pred_acceleration[j], dim=0)
                    # type:
                    # tensor(0.9484, device='cuda:0', grad_fn=<DivBackward0>)

                    loss_corr_ = (corr_gt - corr_pred) ** 2
                    loss_corr += loss_corr_

            # print("corr_gt_list type: ", corr_gt_list[0], type(corr_pred_list[0]))
            # corr_gt_list type:  tensor(1., device='cuda:0') <class 'torch.Tensor'>
            # loss_corr = torch.nn.functional.mse_loss(tf.convert_to_tensor(corr_gt_list), tf.convert_to_tensor(corr_pred_list))
            loss_corr /= k**2

            # loss_corr /= len(random_pids) ** 2
            non_kinematic_mask = non_kinematic_mask.to(torch.bool).to(device)
            num_non_kinematic = torch.sum(non_kinematic_mask)
            loss = torch.where(non_kinematic_mask, loss, torch.zeros(loss.shape, dtype=loss.dtype).to(device))
            loss = torch.sum(loss) / torch.sum(num_non_kinematic)

            print("loss : ", loss)
            print("loss_corr : ", loss_corr)
            loss = loss + (loss_corr_factor * loss_corr)
            # print("loss time: ", time.time()-start_time)
            # exit()

        elif FLAGS.loss == "correlation_2":
            # todo: simplify the equation for faster training
            print("processing correlation loss\n\n")

            loss_corr_factor = 1
            pid_list = [pid for pid in range(target_acceleration.shape[0])]
            while len(pid_list) <20000:
                pid_list.extend([pid for pid in range(target_acceleration.shape[0])])
            random.shuffle(pid_list)
            random_pairs = np.array(pid_list)[:20000].reshape(-1, 2)
            print("pairs: ", random_pairs.shape)

            loss_corr = 0
            start_time = time.time()
            for pair in random_pairs:
                i, j = pair[0], pair[1]

                corr_gt = torch.nn.functional.cosine_similarity(target_acceleration[i], target_acceleration[j], dim=0)
                corr_pred = torch.nn.functional.cosine_similarity(pred_acceleration[i], pred_acceleration[j], dim=0)
                # type:
                # tensor(0.9484, device='cuda:0', grad_fn=<DivBackward0>)

                loss_corr_ = (corr_gt - corr_pred) ** 2
                loss_corr += loss_corr_

            # print("corr_gt_list type: ", corr_gt_list[0], type(corr_pred_list[0]))
            # corr_gt_list type:  tensor(1., device='cuda:0') <class 'torch.Tensor'>
            # loss_corr = torch.nn.functional.mse_loss(tf.convert_to_tensor(corr_gt_list), tf.convert_to_tensor(corr_pred_list))
            loss_corr /= 100**2

            # loss_corr /= len(random_pids) ** 2
            non_kinematic_mask = non_kinematic_mask.to(torch.bool).to(device)
            num_non_kinematic = torch.sum(non_kinematic_mask)
            loss = torch.where(non_kinematic_mask, loss, torch.zeros(loss.shape, dtype=loss.dtype).to(device))
            loss = torch.sum(loss) / torch.sum(num_non_kinematic)

            print("loss : ", loss)
            print("loss_corr : ", loss_corr)
            loss = loss + (loss_corr_factor * loss_corr)
            print("loss time: ", time.time()-start_time)
            exit()

        elif FLAGS.loss == "me":
            print("processing me loss\n\n")
            loss_l1 = torch.nn.functional.l1_loss(pred_acceleration, target_acceleration)
            loss_l1 = loss_l1 * decay_fators_3
            print("loss_l1 shape: ", loss_l1.shape, loss_l1)
            loss_l1 = torch.sum(loss_l1, dim=-1)
            print("loss_l1 shape: ", loss_l1.shape, loss_l1)

            non_kinematic_mask = non_kinematic_mask.to(torch.bool).to(device)
            num_non_kinematic = torch.sum(non_kinematic_mask)
            loss = torch.where(non_kinematic_mask, loss, torch.zeros(loss.shape, dtype=loss.dtype).to(device))
            loss = torch.sum(loss) / torch.sum(num_non_kinematic)

            print("mse loss: ", loss)
            print("me loss: ", loss_l1)
            loss = loss + FLAGS.l_me * loss_l1

        else:
            non_kinematic_mask = non_kinematic_mask.to(torch.bool).to(device)
            num_non_kinematic = torch.sum(non_kinematic_mask)
            # print(loss)
            loss = torch.where(non_kinematic_mask, loss, torch.zeros(loss.shape, dtype=loss.dtype).to(device))
            # print(loss)
            loss = torch.sum(loss) / torch.sum(num_non_kinematic)
        print("loss: ", loss)
        # back propogation
        optimizer.zero_grad()
        if FLAGS.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        running_loss += loss.item()

        step += 1
        # exit()

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
                for j in range(FLAGS.eval_steps):
                    features, targets = next(testDataset)
                    # test inference features.get('step_context') shape:  torch.Size([2, 5])

                    predicted_positions = model.inference(
                        position_sequence=features['position'].to(device),
                        n_particles_per_example=features['n_particles_per_example'].to(device),
                        n_edges_per_example=features['n_edges_per_example'].to(device),
                        senders=features['senders'].to(device),
                        receivers=features['receivers'].to(device),
                        predict_length=PREDICT_LENGTH,
                        particle_types=features['particle_type'].to(device),
                        global_context=features.get('step_context').to(device)
                    )
                    inputs = features['position']
                    sampled_noise = torch.zeros(inputs.shape, dtype=inputs.dtype)
                    # sampled_noise = model.get_random_walk_noise_for_position_sequence(inputs, noise_std_last_step=FLAGS.noise_std)

                    pred_target = model(next_positions=targets.to(device),
                                        position_sequence=inputs.to(device),
                                        position_sequence_noise=sampled_noise.to(device),
                                        n_particles_per_example=features['n_particles_per_example'].to(device),
                                        n_edges_per_example=features['n_edges_per_example'].to(device),
                                        senders=features['senders'].to(device),
                                        receivers=features['receivers'].to(device),
                                        predict_length=PREDICT_LENGTH,
                                        particle_types=features['particle_type'].to(device),
                                        global_context=features.get('step_context').to(device)
                                        )

                    test_mse = torch.nn.functional.mse_loss(*pred_target)
                    p_mse = torch.nn.functional.mse_loss(predicted_positions, targets.to(device))
                    test_loss += test_mse.item()
                    position_loss += p_mse.item()

                writer.add_scalar("loss_mse", test_loss, step)
                writer.add_scalar("position_mse", position_loss, step)
                writer.flush()

                if test_loss < best_loss:
                    torch.save(model.state_dict(),
                               os.path.join(FLAGS.model_path, 'model_loss-{:.2E}_step-{}.pt'.format(test_loss, step)))
                    best_loss = test_loss
            model.train()

    writer.close()


def Test():
    dataset = GraphDataset(mode='rollout', split=FLAGS.eval_split)

    metadata = _read_metadata(FLAGS.data_path)
    acceleration_stats = Stats(torch.DoubleTensor(cast(metadata['acc_mean'])),
                               torch.DoubleTensor(_combine_std(cast(metadata['acc_std']), FLAGS.noise_std)))
    velocity_stats = Stats(torch.DoubleTensor(cast(metadata['vel_mean'])),
                           torch.DoubleTensor(_combine_std(cast(metadata['vel_std']), FLAGS.noise_std)))
    context_stats = Stats(torch.DoubleTensor(cast(metadata['context_mean'])),
                          torch.DoubleTensor(_combine_std(cast(metadata['context_std']), FLAGS.noise_std)))

    normalization_stats = {'acceleration': acceleration_stats, 'velocity': velocity_stats, 'context': context_stats}

    model = LearnedSimulator(num_dimensions=metadata['dim'] * PREDICT_LENGTH, num_seq=INPUT_SEQUENCE_LENGTH,
                             boundaries=torch.DoubleTensor(metadata['bounds']),
                             num_particle_types=NUM_PARTICLE_TYPES, particle_type_embedding_size=16,
                             normalization_stats=normalization_stats)

    loaded = False
    example_index =0
    device = 'cpu'
    with torch.no_grad():
        for features, targets in tqdm(dataset):
            if loaded is False:
                global_context = features['step_context'].to(device)
                if global_context is None:
                    global_context_step = None
                else:
                    # global_context_step = global_context[
                    #     INPUT_SEQUENCE_LENGTH - 1].unsqueeze(-1)
                    # inference, global_context_step shape:  torch.Size([34, 1]) torch.Size([1, 1])
                    # global_context_step = global_context[
                    #     0:INPUT_SEQUENCE_LENGTH]
                    global_context_step = global_context[:-1]
                    print("inference, global_context_step shape: ", global_context.shape, global_context_step.shape)
                    #      torch.Size([5, 1])
                    global_context_step = torch.reshape(global_context_step,[1, -1])
                    print("global_context_step shape: ",global_context_step.shape)


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
                model.load_state_dict(torch.load(FLAGS.model_path))
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                print("device: ", device)
                # config optimizer
                model.setMessagePassingDevices(['cuda:0'])
                model = model.to(device)
                model.eval()
                loaded = True

            offset = 0
            initial_positions = features['position'][:, (0+offset):(INPUT_SEQUENCE_LENGTH+offset)].to(device)
            ground_truth_positions = features['position'][:, (INPUT_SEQUENCE_LENGTH+offset):].to(device)
            global_context = features['step_context'].to(device)
            print("\n initial_positions shape: ", initial_positions.shape)
            print("\n ground_truth_positions shape: ", ground_truth_positions.shape)

            # num_steps = metadata['sequence_length'] - INPUT_SEQUENCE_LENGTH
            num_steps = ground_truth_positions.shape[1]

            current_positions = initial_positions
            updated_predictions = []

            start_time = time.time()
            print("start time: ", start_time)
            print("\n")

            for step in range(num_steps):
                print("start predictiong step: ", step)
                if global_context is None:
                    global_context_step = None
                else:
                    # global_context_step = global_context[
                    #     step + INPUT_SEQUENCE_LENGTH - 1].unsqueeze(-1)
                    # print("global_context_step: ", step + INPUT_SEQUENCE_LENGTH - 1)
                    # global_context_step = global_context[
                    #                       step:(step+INPUT_SEQUENCE_LENGTH)]
                    read_step_context = global_context[:step+INPUT_SEQUENCE_LENGTH]
                    zero_pad = torch.zeros([global_context.shape[0]-read_step_context.shape[0]-1, 1], dtype=features['step_context'].dtype).to(device)

                    # global_context_step = torch.concat([read_step_context, zero_pad], 0)
                    global_context_step = torch.cat([read_step_context, zero_pad], 0)
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
                positions_ground_truth = ground_truth_positions[:, step]
                # print("\n predict_positions shape: ", predict_positions.shape)
                # print("\n positions_ground_truth shape: ", positions_ground_truth.shape)
                #  predict_positions shape:  torch.Size([10132, 1, 3])
                #  positions_ground_truth shape:  torch.Size([10132, 3])

                predict_positions = predict_positions[:, 0].squeeze(1)
                kinematic_mask = torch.repeat_interleave(kinematic_mask,repeats=predict_positions.shape[-1])
                kinematic_mask = torch.reshape(kinematic_mask,[-1,predict_positions.shape[-1]])

                next_position = torch.where(kinematic_mask, positions_ground_truth, predict_positions)

                updated_predictions.append(next_position)
                if FLAGS.rollout_refine is False:
                    # False: rollout the predictions
                    current_positions = torch.cat([current_positions[:, 1:], next_position.unsqueeze(1)], axis=1)
                else:
                    # True: single-step prediction for all steps
                    current_positions = torch.cat([current_positions[:,1:],positions_ground_truth.unsqueeze(1)], axis=1)

            updated_predictions = torch.stack(updated_predictions)
            print("\n updated_predictions shape: ", updated_predictions.shape)
            print("\n ground_truth_positions shape: ", ground_truth_positions.shape)

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

            # squared_error = (rollout_op['predicted_rollout'] -
            #                  rollout_op['ground_truth_rollout']) ** 2
            # print("squared_error: ", squared_error)

            # Add a leading axis, since Estimator's predict method insists that all
            # tensors have a shared leading batch axis fo the same dims.
            rollout_op = tree.map_structure(lambda x: x.numpy(), rollout_op)

            rollout_op['metadata'] = metadata
            filename = f'rollout_{FLAGS.eval_split}_{example_index}.pkl'
            filename = os.path.join(FLAGS.output_path, filename)
            if not os.path.exists(FLAGS.output_path):
                os.makedirs(FLAGS.output_path)
            with open(filename, 'wb') as file:
                pickle.dump(rollout_op, file)
            example_index+=1
            
            print(f"prediction time: {time.time()-start_time}\n")


def main(_):
    if FLAGS.mode == "train":
        Train()
    else:
        Test()

if __name__ == '__main__':
    # tf.disable_v2_behavior()
    app.run(main)

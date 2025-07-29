# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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
# ==============================================================================

# -*- coding: utf-8 -*-


import argparse
import os


def _set_gpu_preallocation(mode: float):
    """Set JAX GPU memory preallocation fraction.

    This function configures how much of the total GPU memory JAX will preallocate
    by setting the XLA_PYTHON_CLIENT_MEM_FRACTION environment variable. By default,
    JAX preallocates 75% of GPU memory, but this can be adjusted to avoid out-of-memory
    errors.

    Args:
        mode: A float between 0.0 and 1.0 (exclusive) representing the fraction of
              GPU memory to preallocate. For example, 0.5 would allocate 50% of
              available GPU memory.

    Raises:
        AssertionError: If mode is not a float or not in the range [0.0, 1.0).

    Example:
        >>> _set_gpu_preallocation(0.5)  # Allocate 50% of GPU memory
    """
    assert isinstance(mode, float) and 0. <= mode < 1., f'GPU memory preallocation must be in [0., 1.]. But got {mode}.'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(mode)
    # os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'


def _set_gpu_device(device_ids):
    """Set visible GPU devices for CUDA and turn off JAX traceback filtering.

    This function configures which GPU devices are visible to the application by setting
    the CUDA_VISIBLE_DEVICES environment variable. It handles different input types
    for device specification.

    Args:
        device_ids: GPU device identifiers in one of these formats:
            - int: Single GPU ID (e.g., 0 for first GPU)
            - tuple/list: Multiple GPU IDs (e.g., [0,1,2] for first three GPUs)
            - str: Comma-separated GPU IDs (e.g., "0,1,2")

    Raises:
        ValueError: If device_ids is not int, tuple, list, or str.

    Note:
        Also disables JAX traceback filtering to show complete stack traces.
    """
    if isinstance(device_ids, int):
        device_ids = str(device_ids)
    elif isinstance(device_ids, (tuple, list)):
        device_ids = ','.join([str(d) for d in device_ids])
    elif isinstance(device_ids, str):
        device_ids = device_ids
    else:
        raise ValueError
    os.environ['CUDA_VISIBLE_DEVICES'] = device_ids
    os.environ['JAX_TRACEBACK_FILTERING'] = 'off'


def get_parser(gpu_pre_allocate=0.99):
    """Create and configure an argument parser for neural network training settings.

    This function initializes an ArgumentParser with default parameters for training
    neural networks, particularly focusing on GPU settings, dataset configurations,
    and network architecture parameters. It handles two initialization modes:
    - Loading parameters from a specified file
    - Using built-in defaults

    Args:
        gpu_pre_allocate (float, optional): Fraction of GPU memory to pre-allocate.
            Must be between 0.0 and 1.0. Defaults to 0.99.

    Returns:
        argparse.Namespace: Parsed command-line arguments with configured defaults.

    Note:
        This function automatically configures GPU devices and memory allocation
        based on the provided settings.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--devices', type=str, default='0', help='The GPU device ids.')
    parser.add_argument("--filepath", type=str, default=None)
    args, _ = parser.parse_known_args()

    # device management
    _set_gpu_device(args.devices)
    _set_gpu_preallocation(gpu_pre_allocate)

    if args.filepath is not None:
        with open(os.path.join(args.filepath, 'first-round-losses.txt'), 'r') as f:
            line = f.readline().replace('Namespace', 'dict')
            settings = eval(line)
            vjp_method = settings['vjp_method']
            etrace_decay = settings['etrace_decay']
            dt = settings['dt']
            epoch_round1 = settings['epoch_round1']
            epoch_round2 = settings['epoch_round2']
            flywire_version = settings['flywire_version']
            neural_activity_id = settings['neural_activity_id']
            neural_activity_max_fr = settings['neural_activity_max_fr']
            lr_round1 = settings['lr_round1']
            lr_round2 = settings['lr_round2']
            connectome_scale_factor = settings['connectome_scale_factor']
            split = settings['split']
            sim_before_train = settings['sim_before_train']
            loss = settings['loss']
            batch_size = settings['batch_size']
            n_lora_rank = settings['n_lora_rank']
            n_rnn_hidden = settings['n_rnn_hidden']

    else:
        vjp_method = 'single-step'
        etrace_decay = 0.99
        dt = 0.2
        epoch_round1 = 200
        flywire_version = '630'
        neural_activity_id = '2017-10-26_1'
        neural_activity_max_fr = 100.
        lr_round1 = 1e-2
        lr_round2 = 1e-3
        connectome_scale_factor = 0.0825 / 100
        split = 0.7
        sim_before_train = 0.1
        loss = 'mse'
        batch_size = 128
        n_lora_rank = 20
        n_rnn_hidden = 256

    # training method
    parser.add_argument("--vjp_method", type=str, default=vjp_method, choices=['multi-step', 'single-step'],
                        help="The method for computing the Jacobian-vector product (JVP).")
    parser.add_argument("--etrace_decay", type=float, default=etrace_decay,
                        help="The time constant of eligibility trace ")
    parser.add_argument("--fitting_target", type=str, default='csr', choices=['lora', 'csr'])

    # training parameters
    parser.add_argument('--dt', type=float, default=dt, help='Control the time step of the simulation.')
    parser.add_argument('--epoch', type=int, default=epoch_round1,
                        help='The number of epochs for spiking network training.')
    parser.add_argument('--flywire_version', type=str, default=flywire_version, help='The version of flywire.')
    parser.add_argument('--neural_activity_id', type=str, default=neural_activity_id, help='The id of neural activity.')
    parser.add_argument('--neural_activity_max_fr', type=float, default=neural_activity_max_fr,
                        help='The maximum firing rate of neural activity. [Hz]')
    parser.add_argument('--lr', type=float, default=lr_round1,
                        help='The learning rate for first-round spiking network training.')
    parser.add_argument('--connectome_scale_factor', type=float, default=connectome_scale_factor,
                        help='The scale factor of connectome. [mV]')
    parser.add_argument('--split', type=float, default=split, help='The split ratio of training and validation set.')
    parser.add_argument('--sim_before_train', type=float, default=sim_before_train,
                        help='The fraction of simulation time before training.')
    parser.add_argument('--loss', type=str, default=loss,
                        choices=['mse', 'mae', 'huber', 'cosine_distance', 'log_cosh'],
                        help='The loss function for training. [mse, mae, huber, cosine_distance, log_cosh]')
    parser.add_argument('--batch_size', type=int, default=batch_size, help='The batch size for training.')
    parser.add_argument('--n_lora_rank', type=int, default=n_lora_rank, help='The rank of low-rank approximation.')
    parser.add_argument('--n_rnn_hidden', type=int, default=n_rnn_hidden, help='The number of hidden units in rnn.')

    return parser.parse_args()

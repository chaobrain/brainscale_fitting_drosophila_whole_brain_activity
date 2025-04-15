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
    """GPU memory allocation.

    If preallocation is enabled, this makes JAX preallocate ``percent`` of the total GPU memory,
    instead of the default 75%. Lowering the amount preallocated can fix OOMs that occur when the JAX program starts.
    """
    assert isinstance(mode, float) and 0. <= mode < 1., f'GPU memory preallocation must be in [0., 1.]. But got {mode}.'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(mode)


def _set_gpu_device(device_ids):
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
    parser = argparse.ArgumentParser()

    parser.add_argument('--devices', type=str, default='0', help='The GPU device ids.')
    args, _ = parser.parse_known_args()
    # device management
    _set_gpu_device(args.devices)
    _set_gpu_preallocation(gpu_pre_allocate)

    # training method
    parser.add_argument("--vjp_method", type=str, default='single-step', choices=['multi-step', 'single-step'],
                        help="The method for computing the Jacobian-vector product (JVP).")
    parser.add_argument("--etrace_decay", type=float, default=0.99, help="The time constant of eligibility trace ")

    # training parameters
    parser.add_argument('--dt', type=float, default=0.2, help='Control the time step of the simulation.')
    parser.add_argument('--epoch_round1', type=int, default=200,
                        help='The number of epochs for spiking network training.')
    parser.add_argument('--epoch_round2', type=int, default=200, help='The number of epochs for rnn training.')
    parser.add_argument('--flywire_version', type=str, required=True, help='The version of flywire.')
    parser.add_argument('--neural_activity_id', type=str, required=True, help='The id of neural activity.')
    parser.add_argument('--neural_activity_max_fr', type=float, default=100.0,
                        help='The maximum firing rate of neural activity. [Hz]')
    parser.add_argument('--lr_round1', type=float, default=1e-2,
                        help='The learning rate for first-round spiking network training.')
    parser.add_argument('--lr_round2', type=float, default=1e-3,
                        help='The learning rate for second-round rnn training.')
    parser.add_argument('--connectome_scale_factor', type=float, default=0.0825 / 100,
                        help='The scale factor of connectome. [mV]')
    parser.add_argument('--bin_size', type=float, default=0.5, help='The bin size of neural activity. [Hz]')
    parser.add_argument('--split', type=float, default=0.7, help='The split ratio of training and validation set.')
    parser.add_argument('--sim_before_train', type=float, default=0.1,
                        help='The fraction of simulation time before training.')
    parser.add_argument('--loss', type=str, default='mse',
                        choices=['mse', 'mae', 'huber', 'cosine_distance', 'log_cosh'],
                        help='The loss function for training. [mse, mae, huber, cosine_distance, log_cosh]')
    parser.add_argument('--batch_size', type=int, default=128, help='The batch size for training.')
    parser.add_argument('--n_lora_rank', type=int, default=20, help='The rank of low-rank approximation.')
    parser.add_argument('--n_rnn_hidden', type=int, default=256, help='The number of hidden units in rnn.')

    return parser.parse_args()

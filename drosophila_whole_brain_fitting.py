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

"""
Drosophila Whole Brain Fitting Model (v4.2)

This module implements a comprehensive framework for modeling and predicting neural activity
in the Drosophila brain using a combination of spiking neural networks and recurrent neural
networks. The model leverages the FlyWire connectome dataset to construct biologically plausible
network architectures that replicate observed firing patterns in the fly brain.

Training Process:
----------------
The training is structured in two rounds:

1. First Round:
   - Trains a biologically plausible spiking neural network using eligibility traces
   - Maps activity from previous time steps to predict current time step activity
   - Uses a combination of sparse connectivity from the connectome and low-rank trainable
     parameters (LoRA) for computational efficiency
   - Optimizes network weights to minimize differences between predicted and observed firing rates

2. Second Round:
   - Trains a GRU-based recurrent neural network as an input encoder/decoder
   - Uses simulated data from the first-round model as training data
   - Provides a more efficient representation for generating sequences of neural activity

Input Generation:
----------------
1. Firing Rate Tokenization:
   - Continuous firing rates are tokenized into discrete bins using uniform binning
   - Each bin represents X mV of firing rate with equal spacing between tokens
   - Provides a discrete representation suitable for classification metrics

2. Input Handling:
   - Neural activities are loaded from experimental datasets
   - Firing rates are normalized and scaled appropriately
   - Noise is optionally added for robustness during training
   - Input representation includes previous time step activity to predict next step

Training Paradigm:
----------------
The core training paradigm follows next-step prediction:
- Use previous time step firing rates as input
- Predict current time step firing rates as output
- Compare predictions against observed/target firing rates
- Update network parameters using gradient-based optimization
- Apply eligibility traces for biologically plausible learning in spiking networks

Key Components:
-------------
- Population: Implements leaky integrate-and-fire neurons with FlyWire connectome
- Interaction: Handles synaptic connectivity and signal propagation between neurons
- NeuralActivity: Manages neural activity data and conversions
- DrosophilaSpikingNetwork: Integrates components into a full brain simulation
- SpikingNetworkTrainer: Handles training and optimization procedures
- DrosophilaInputEncoder: GRU-based network for efficient representation learning

The module provides utilities for loading pre-trained models, generating simulation data,
and visualizing results for thorough analysis of network performance.
"""

import os
import platform
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from args import get_parser

settings = get_parser()

if (
    platform.system() == 'Linux'
    and
    platform.platform() not in ['Linux-5.15.167.4-microsoft-standard-WSL2-x86_64-with-glibc2.35']
):
    import matplotlib
    matplotlib.use('Agg')

sys.path.append('/mnt/d/codes/projects/brainscale')
sys.path.append('/mnt/d/codes/projects/brainevent')
sys.path.append('D:/codes/projects/brainscale')
sys.path.append('D:/codes/projects/brainevent')

import brainstate
import braintools
import brainscale
import brainunit as u
import jax
import jax.numpy as jnp

from utils import (
    output,
    FilePath,
    DrosophilaSpikingNetTrainer,
    NeuralData,
    DrosophilaSpikingNetwork,
    DrosophilaInputEncoder,
    DrosophilaInputTrainer,
    DrosophilaRestingStateModel,
)

brainstate.environ.set(dt=settings.dt * u.ms)


def _train_spiking_network(
    trainer: DrosophilaSpikingNetTrainer,
    train_epoch: int,
    batch_size: int = 128,
    checkpoint_path: str = None,
):
    brainstate.nn.vmap_init_all_states(trainer.target, axis_size=batch_size, state_tag='hidden')

    if checkpoint_path is not None:
        braintools.file.msgpack_load(checkpoint_path, trainer.target.states(brainstate.ParamState))
        filepath = os.path.join(os.path.dirname(checkpoint_path), 'new')
        os.makedirs(filepath, exist_ok=True)
    else:
        filepath = trainer.filepath

    # training process
    os.makedirs(filepath, exist_ok=True)
    with open(f'{filepath}/first-round-losses.txt', 'w') as file:
        output(file, str(settings))
        max_acc = 0.
        for i_epoch in range(train_epoch):
            i_batch = 0
            all_loss = []
            all_acc = []
            for input_embed, target_neuropil_fr in trainer.data.iter_train_data(batch_size=batch_size):
                t0 = time.time()
                res = trainer._train(input_embed, target_neuropil_fr)
                loss, neuropil_fr, max_g, acc = jax.block_until_ready(res)
                t1 = time.time()

                output(
                    file,
                    f'epoch = {i_epoch}, '
                    f'batch = {i_batch}, '
                    f'loss = {loss:.5f}, '
                    f'bin acc = {acc:.5f}, '
                    f'lr = {trainer.opt.lr():.6f}, '
                    f'time = {t1 - t0:.5f}s'
                )
                output(file, f'max_g = {max_g}')

                i_batch += 1
                all_loss.append(loss)
                all_acc.append(acc)
            trainer._show_res(neuropil_fr, target_neuropil_fr, i_epoch, '', filepath=filepath)
            trainer.opt.lr.step_epoch()

            # save checkpoint
            loss = np.mean(all_loss)
            acc = np.mean(all_acc)
            output(
                file,
                f'epoch = {i_epoch}, '
                f'loss = {loss:.5f}, '
                f'bin acc = {acc:.5f}, '
                f'lr = {trainer.opt.lr():.6f}'
            )
            if acc > max_acc:
                braintools.file.msgpack_save(
                    f'{filepath}/first-round-checkpoint.msgpack',
                    trainer.target.states(brainstate.ParamState),
                )
                max_acc = acc


def first_round_train():
    """
    Train the first round of a spiking neural network to simulate Drosophila brain activity.

    This function configures and executes the first round of training for the spiking neural network,
    which aims to reproduce neuropil firing rates observed in the Drosophila brain. It either
    initializes a new training session or continues from a checkpoint with appropriate settings.

    The function handles:
    - Setting training parameters (epochs, learning rate, etc.)
    - Loading checkpoint settings if provided
    - Configuring environment settings such as time step
    - Initializing and running the SpikingNetworkTrainer

    The training process maps neural activity from previous time steps to predict activity
    at the current time step using a biologically plausible spiking neural network with
    eligibility trace-based learning.

    Returns
    -------
    str
        The filepath where the trained model checkpoint is saved.

    Notes
    -----
    - Default parameters are set for a new training session if no checkpoint is provided
    - When resuming from a checkpoint, parameters are extracted from the checkpoint filepath
    - Uses a 0.2 ms simulation time step
    """
    trainer = DrosophilaSpikingNetTrainer(
        lr=settings.lr_round1,
        etrace_decay=settings.etrace_decay,
        sim_before_train=settings.sim_before_train,
        neural_activity_id=settings.neural_activity_id,
        flywire_version=settings.flywire_version,
        max_firing_rate=settings.neural_activity_max_fr * u.Hz,
        loss_fn=settings.loss,
        scale_factor=settings.connectome_scale_factor * u.mV,
        conn_param_type=brainscale.ETraceParam if settings.etrace_decay != 0. else brainscale.NonTempParam,
        input_param_type=brainscale.ETraceParam if settings.etrace_decay != 0. else brainscale.NonTempParam,
        bin_size=settings.bin_size * u.Hz,
        vjp_method=settings.vjp_method,
        n_rank=settings.n_lora_rank,
        split=settings.split,
    )
    _train_spiking_network(
        trainer,
        train_epoch=settings.epoch_round1,
        batch_size=settings.batch_size
    )
    return trainer.filepath


def first_round_generate_training_data(filepath: str):
    """
    Generate simulated neural activity training data using a pre-trained spiking neural network.

    This function loads parameters from a trained model checkpoint, initializes a FiringRateNetwork
    with those parameters, and simulates neural activity across the full time series of the
    original dataset. For each time step, it captures the simulated neuropil firing rates
    and calculates similarity metrics between simulated and target data.

    The generated data is saved in the same directory as the checkpoint file and can be used
    for subsequent training of recurrent neural networks or other downstream models.

    Parameters
    ----------
    filepath : str
        The filepath to the directory containing the model checkpoint. The function expects
        the checkpoint to be named 'first-round-checkpoint.msgpack'.

    Notes
    -----
    - Uses a hard-coded filepath to load a specific model checkpoint
    - Initializes network states and parameters from the checkpoint
    - Simulates activity time step by time step, recording firing rates
    - Calculates similarity metrics between simulated and target firing rates
    - Progress is displayed using a tqdm progress bar
    """

    pather = FilePath.from_filepath(filepath)
    data = NeuralData(
        flywire_version=pather.flywire_version,
        neural_activity_id=pather.neural_activity_id,
        neural_activity_max_fr=pather.neural_activity_max_fr,
        bin_size=pather.bin_size,
        split=pather.split,
    )
    net = DrosophilaSpikingNetwork(
        n_neuropil=data.n_neuropil,
        flywire_version=pather.flywire_version,
        conn_param_type=pather.conn_param_type,
        input_param_type=pather.input_param_type,
        n_rank=pather.n_rank,
        scale_factor=pather.scale_factor,
    )
    braintools.file.msgpack_load(
        os.path.join(filepath, 'first-round-checkpoint.msgpack'),
        net.states(brainstate.ParamState)
    )
    brainstate.nn.init_all_states(net)

    @brainstate.compile.jit
    def one_step(i, indices):
        input_embed = data.spike_rates[i] / u.Hz
        output_neuropil_fr = data.spike_rates[i + 1]
        net.simulate(input_embed, indices[:n_sim])
        net.simulate(input_embed, indices[n_sim:])
        neuropil_fr = data.count_neuropil_fr(net.pop.spike_count.value, net.n_sample_step - n_sim).to_decimal(u.Hz)
        sim = jnp.corrcoef(u.get_mantissa(output_neuropil_fr), neuropil_fr)[0, 1]
        return neuropil_fr, sim

    n_sim = int(pather.sim_before_train * net.n_sample_step)
    simulated_neuropil_fr = []
    indices = np.arange(net.n_sample_step)
    bar = tqdm(total=data.n_train)
    all_sim = []
    for i in range(0, data.n_train):
        neuropil_fr, sim = one_step(i, indices)
        bar.update()
        bar.set_description(f'similarity = {sim:.5f}')
        indices = indices + net.n_sample_step
        simulated_neuropil_fr.append(neuropil_fr)
        all_sim.append(sim)
    bar.close()
    print(f'Mean similarity = {np.mean(np.mean(all_sim)):.5f}')
    simulated_neuropil_fr = np.asarray(simulated_neuropil_fr)  # [n_time, n_neuropil]
    np.save(os.path.join(filepath, 'simulated_neuropil_fr'), simulated_neuropil_fr)


def _train_rnn_network(trainer: DrosophilaInputTrainer):
    os.makedirs(trainer.filepath, exist_ok=True)
    with open(f'{trainer.filepath}/second-round-losses.txt', 'w') as file:
        output(file, str(settings))

        all_loss = []
        all_acc = []
        t0 = time.time()
        min_loss = np.inf
        for i_epoch in range(settings.epoch_round2):
            train_loss_epoch = []
            train_acc_epoch = []
            for i_batch in range(100):
                inputs = trainer.generate_inputs()
                loss, acc = trainer._epoch_train(inputs)
                train_acc_epoch.append(acc)
                train_loss_epoch.append(loss)
            all_loss.extend(train_loss_epoch)
            all_acc.extend(train_acc_epoch)
            acc = np.mean(train_acc_epoch)
            loss = np.mean(train_loss_epoch)
            test_mse, test_acc = trainer._epoch_test()
            output(
                file,
                f'Epoch = {i_epoch}, '
                f'Train Loss = {loss:.5f}, '
                f'Train bin acc = {acc:.5f}, '
                f'Test Loss = {test_mse:.5f}, '
                f'Test bin acc = {test_acc:.5f}, '
                f'lr = {trainer.opt.lr():.6f}, '
                f'time = {time.time() - t0:.2f} s'
            )
            trainer.opt.lr.step_epoch()
            if min_loss > test_mse:
                min_loss = test_mse
                braintools.file.msgpack_save(
                    f'{trainer.filepath}/second-round-checkpoint.msgpack',
                    trainer.net.states(brainstate.LongTermState)
                )
            t0 = time.time()

    fig, gs = braintools.visualize.get_figure(1, 2, 3, 4)
    fig.add_subplot(gs[0, 0])
    plt.plot(all_loss)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    fig.add_subplot(gs[0, 1])
    plt.plot(all_acc)
    plt.xlabel('Batch')
    plt.ylabel('Bin accuracy')
    plt.savefig(f'{trainer.filepath}/second-round-loss-acc.pdf')
    plt.close()


def second_round_train(filepath):
    """
    Train the second round of neural network to predict firing rates based on simulated data.

    This function loads simulated neural activity data generated from a previously trained
    spiking neural network and trains a recurrent neural network (DrosophilaInputEncoder)
    to predict firing rates in the Drosophila brain. The second round of training focuses
    on creating a more efficient model that can generalize from the simulated data.

    The function handles:
    - Loading parameters and simulation results from the first round
    - Setting up the recurrent neural network architecture
    - Generating training inputs with appropriate noise augmentation
    - Training the model with gradient-based optimization
    - Monitoring performance metrics (loss and bin accuracy)
    - Saving the best model checkpoint based on loss

    Notes
    -----
    - Uses a GRU-based encoder architecture (DrosophilaInputEncoder)
    - Applies noise to input data for robustness
    - Monitors bin accuracy (correctly classified firing rate bins)
    - Uses a step learning rate schedule that decays over time
    - Saves the best model checkpoint based on loss value
    """
    trainer = DrosophilaInputTrainer(
        filepath=filepath,
        batch_size=settings.batch_size,
        n_rnn_hidden=settings.n_rnn_hidden,
        lr_round2=settings.lr_round2,
        noise_scale=0.1,
    )
    _train_rnn_network(trainer)


def second_round_loading(filepath):
    args = FilePath.from_filepath(filepath)

    data = np.load(f'data/spike_rates/ito_{args.neural_activity_id}_spike_rate.npz')
    spike_rates = u.math.asarray(data['rates'][1:] * args.neural_activity_max_fr).T
    simulated_spike_rates = np.load(os.path.join(filepath, 'simulated_neuropil_fr.npy'))

    net = DrosophilaInputEncoder(
        n_in=spike_rates.shape[1],
        n_hidden=settings.n_rnn_hidden,
        n_out=spike_rates.shape[1]
    )
    braintools.file.msgpack_load(
        os.path.join(filepath, f'second-round-checkpoint.msgpack'),
        net.states(brainstate.LongTermState)
    )
    brainstate.nn.init_all_states(net)
    outputs = brainstate.compile.for_loop(net, simulated_spike_rates)

    num = 5
    times = np.arange(simulated_spike_rates.shape[0]) / (1.2 * u.Hz)
    for ii in range(0, spike_rates.shape[1], num):
        fig, gs = braintools.visualize.get_figure(num, 2, 4, 8.0)
        for i in range(num):
            # plot simulated neuropil firing rate data
            fig.add_subplot(gs[i, 0])
            data = outputs[:, i + ii]
            plt.plot(times, data)
            plt.ylim(0., data.max() * 1.05)
            # plot experimental neuropil firing rate data
            fig.add_subplot(gs[i, 1])
            data = simulated_spike_rates[:, i + ii]
            plt.plot(times, data)
            plt.ylim(0., data.max() * 1.05)
        plt.show()


def example_to_load(filepath: str):
    """
    Load and initialize a pre-trained Drosophila spiking neural network model for demonstration.

    This function demonstrates how to load a previously trained spiking neural network model
    from a checkpoint file and initialize it for use. It extracts configuration parameters
    from the checkpoint filepath, sets up the environment parameters, and initializes the
    DrosophilaSpikingNetwork with the extracted settings.

    The function is intended as an example showing how to:
    - Extract model hyperparameters from a checkpoint filepath
    - Configure the simulation environment
    - Initialize a spiking neural network with the appropriate parameters
    - Load trained weights from a checkpoint file

    Notes
    -----
    - Sets the simulation time step to 0.2 milliseconds
    - Uses a hardcoded filepath to a specific model checkpoint
    - Extracts all training parameters from the checkpoint filepath
    - Initializes but does not execute simulation (incomplete function)
    """
    drosophila = DrosophilaRestingStateModel(filepath, n_rnn_hidden=settings.n_rnn_hidden)
    drosophila.f_predict()


if __name__ == '__main__':
    pass
    if settings.filepath is None:
        _filepath_ = first_round_train()
    else:
        _filepath_ = settings.filepath
    first_round_generate_training_data(_filepath_)
    second_round_train(_filepath_)
    example_to_load(_filepath_)

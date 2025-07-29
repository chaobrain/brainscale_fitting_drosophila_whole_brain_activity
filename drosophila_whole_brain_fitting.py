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
- RecurrentNetwork: Handles synaptic connectivity and signal propagation between neurons
- NeuralActivity: Manages neural activity data and conversions
- DrosophilaSpikingNetwork: Integrates components into a full brain simulation
- SpikingNetworkTrainer: Handles training and optimization procedures
- DrosophilaInputEncoder: GRU-based network for efficient representation learning

The module provides utilities for loading pre-trained models, generating simulation data,
and visualizing results for thorough analysis of network performance.
"""

import platform

from args import get_parser

settings = get_parser()

if platform.system() == 'Linux':
    import matplotlib

    matplotlib.use('Agg')

import brainstate
import brainscale
import brainunit as u

from models import DrosophilaSpikingNetTrainer

brainstate.environ.set(dt=settings.dt * u.ms)


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
        lr=settings.lr,
        etrace_decay=settings.etrace_decay,
        sim_before_train=settings.sim_before_train,
        neural_activity_id=settings.neural_activity_id,
        flywire_version=settings.flywire_version,
        max_firing_rate=settings.neural_activity_max_fr * u.Hz,
        loss_fn=settings.loss,
        scale_factor=settings.connectome_scale_factor * u.mV,
        conn_param_type=brainscale.ETraceParam if settings.etrace_decay != 0. else brainscale.NonTempParam,
        input_param_type=brainscale.ETraceParam if settings.etrace_decay != 0. else brainscale.NonTempParam,
        vjp_method=settings.vjp_method,
        n_rank=settings.n_lora_rank,
        split=settings.split,
        fitting_target=settings.fitting_target,
    )
    trainer.f_train(train_epoch=settings.epoch, settings=settings)


if __name__ == '__main__':
    first_round_train()

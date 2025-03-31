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

1、连续数值的 firing rate tokenization

使用 uniform binning，将连续数值的 firing rate tokenize为 若干个 bin，每个 bin 代表 0.1 mV 的 firing rate。
每个 bin 对应一个 token，token 之间的距离为 0.1 mV。每个bin使用随机的 embedding 表示。

2、训练范式

使用上一个时刻的 firing rate 作为输入，当前时刻的 firing rate 作为输出。

"""

from __future__ import annotations

import datetime
import os
import platform
import sys
import time
from pathlib import Path
from typing import Callable

from tqdm import tqdm

if platform.platform() in [
    'Linux-6.8.0-48-generic-x86_64-with-glibc2.35',
    'Linux-5.15.0-131-generic-x86_64-with-glibc2.31',
    'Linux-5.15.0-84-generic-x86_64-with-glibc2.31',
]:
    import matplotlib

    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.99'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'
sys.path.append('/mnt/d/codes/projects/brainscale')

import brainevent
import brainstate
import braintools
import brainscale
import brainunit as u
import jax
import jax.numpy as jnp

from utils import g_init, v_init, count_pre_post_connections, barplot, output


class Population(brainstate.nn.Neuron):
    """
    A population of neurons with leaky integrate-and-fire dynamics.

    The dynamics of the neurons are given by the following equations::

       dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
       dg/dt = -g / tau               : volt (unless refractory)


    Args:
      v_rest: The resting potential of the neurons.
      v_reset: The reset potential of the neurons after a spike.
      v_th: The threshold potential of the neurons for spiking.
      tau_m: The membrane time constant of the neurons.
      tau_syn: The synaptic time constant of the neurons.
      tau_ref: The refractory period of the neurons.
      spk_fun: The spike function of the neurons.
      name: The name of the population.

    """

    def __init__(
        self,
        flywire_version: str | int = '783',
        v_rest: u.Quantity = 0 * u.mV,  # resting potential
        v_reset: u.Quantity = 0 * u.mV,  # reset potential after spike
        v_th: u.Quantity = 1 * u.mV,  # potential threshold for spiking
        tau_m: u.Quantity = 20 * u.ms,  # membrane time constant
        # Jürgensen et al https://doi.org/10.1088/2634-4386/ac3ba6
        tau_syn: u.Quantity = 5 * u.ms,  # synaptic time constant
        # Lazar et al https://doi.org/10.7554/eLife.62362
        tau_ref: u.Quantity | None = 2.2 * u.ms,  # refractory period
        spk_fun: Callable = brainstate.surrogate.ReluGrad(width=1.5),  # spike function
        V_init: Callable = brainstate.init.Constant(0 * u.mV),  # initial voltage
        g_init: Callable = brainstate.init.Constant(0. * u.mV),  # initial voltage
        name: str = None,
    ):
        # connectome data
        if flywire_version in ['783', 783]:
            path_neu = 'data/Completeness_783.csv'
            path_syn = 'data/Connectivity_783.parquet'
        elif flywire_version in ['630', 630]:
            path_neu = 'data/Completeness_630_final.csv'
            path_syn = 'data/Connectivity_630_final.parquet'
        else:
            raise ValueError('flywire_version must be either "783" or "630"')
        self.flywire_version = flywire_version

        self.path_neu = Path(path_neu)
        self.path_syn = Path(path_syn)

        print('Loading neuron information ...')

        # neuron ids
        flywire_ids = pd.read_csv(self.path_neu, index_col=0)
        self.n_neuron = len(flywire_ids)

        super().__init__(self.n_neuron, name=name)

        self.flyid2i = {f: i for i, f in enumerate(flywire_ids.index)}
        self.i2flyid = {i: f for i, f in enumerate(flywire_ids.index)}

        # parameters
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_th = v_th
        self.tau_m = tau_m
        self.tau_syn = tau_syn
        self.tau_ref = tau_ref if tau_ref is None else u.math.full(self.varshape, tau_ref)
        self.spk_fun = spk_fun

        # initializer
        self.V_init = V_init
        self.g_init = g_init

    def init_state(self):
        self.v = brainscale.ETraceState(brainstate.init.param(self.V_init, self.varshape))
        self.g = brainscale.ETraceState(brainstate.init.param(self.g_init, self.varshape))
        self.spike_count = brainscale.ETraceState(jnp.zeros(self.varshape, dtype=brainstate.environ.dftype()))
        if self.tau_ref is not None:
            self.t_ref = brainstate.HiddenState(
                brainstate.init.param(brainstate.init.Constant(-1e7 * u.ms), self.varshape)
            )

    def reset_state(self):
        self.reset_spk_count()
        self.v.value = brainstate.init.param(self.V_init, self.varshape)
        if self.tau_ref is not None:
            self.t_ref.value = brainstate.init.param(brainstate.init.Constant(-1e7 * u.ms), self.varshape)

    def reset_spk_count(self, batch_size=None):
        self.spike_count.value = brainstate.init.param(jnp.zeros, self.varshape, batch_size)

    def get_refractory(self):
        if self.tau_ref is None:
            return jnp.zeros(self.varshape, dtype=bool)
        else:
            t = brainstate.environ.get('t')
            ref = (t - self.t_ref.value) <= self.tau_ref
            return ref

    def get_spike(self, v=None):
        v = self.v.value if v is None else v
        return self.spk_fun((v - self.v_th) / (1. * u.mV))

    def update(self, x: u.Quantity[u.mV]):
        t = brainstate.environ.get('t')

        # numerical integration
        dg = lambda g, t: -g / self.tau_syn
        dv = lambda v, t, g: (self.v_rest - v + g) / self.tau_m
        g = brainstate.nn.exp_euler_step(dg, self.g.value, t)
        g += x  # external input current
        v = brainstate.nn.exp_euler_step(dv, self.v.value, t, g)
        v = self.sum_delta_inputs(v)

        # # numerical integration
        # dg = lambda g, t: -g / self.tau_syn
        # dv = lambda v, t, g: (self.v_rest - v + g) / self.tau_m
        # g = brainstate.nn.exp_euler_step(dg, self.g.value, t)
        # v = brainstate.nn.exp_euler_step(dv, self.v.value, t, self.g.value)
        # v = self.sum_delta_inputs(v)
        # g += x  # external input current

        # refractory period
        if self.tau_ref is not None:
            ref = (t - self.t_ref.value) <= self.tau_ref
            v = u.math.where(ref, self.v.value, v)
            g = u.math.where(ref, self.g.value, g)

        # spikes
        spk = self.get_spike(v)
        self.spike_count.value += spk

        # update states
        spk_current = jax.lax.stop_gradient(spk)
        self.v.value = spk_current * (self.v_reset - v) + v
        self.g.value = g - spk_current * g
        if self.tau_ref is not None:
            self.t_ref.value = u.math.where(spk, t, self.t_ref.value)
        return spk


def load_syn(flywire_version: str | int) -> brainevent.CSR:
    if flywire_version in ['783', 783]:
        path_neu = 'data/Completeness_783.csv'
        path_syn = 'data/Connectivity_783.parquet'
    elif flywire_version in ['630', 630]:
        path_neu = 'data/Completeness_630_final.csv'
        path_syn = 'data/Connectivity_630_final.parquet'
    else:
        raise ValueError('flywire_version must be either "783" or "630"')

    # neuron information
    flywire_ids = pd.read_csv(path_neu, index_col=0)
    n_neuron = len(flywire_ids)

    # synapses: CSR connectivity matrix
    flywire_conns = pd.read_parquet(path_syn)
    i_pre = flywire_conns.loc[:, 'Presynaptic_Index'].values
    i_post = flywire_conns.loc[:, 'Postsynaptic_Index'].values
    weight = flywire_conns.loc[:, 'Excitatory x Connectivity'].values
    sort_indices = np.argsort(i_pre)
    i_pre = i_pre[sort_indices]
    i_post = i_post[sort_indices]
    weight = weight[sort_indices]

    values, counts = np.unique(i_pre, return_counts=True)
    indptr = np.zeros(n_neuron + 1, dtype=int)
    indptr[values + 1] = counts
    indptr = np.cumsum(indptr)
    indices = i_post

    csr = brainevent.CSR(
        (weight, indices, indptr),
        shape=(n_neuron, n_neuron)
    )
    return csr


class Interaction(brainstate.nn.Module):
    """
    A neural network module that manages the interaction between a population of neurons,
    external inputs, and synaptic connections.

    Args:
        pop (Population): The population of neurons in the network.
        pop_input (PopulationInput): The input module for the neuron population.
        csr (brainevent.CSR, optional): The sparse connectivity matrix. Defaults to None.
        conn_mode (str, optional): The mode of connectivity, either 'sparse' or 'sparse+low+rank'. Defaults to 'sparse'.
        conn_param_type (type, optional): The type of connection parameters. Defaults to brainscale.ETraceParam.
    """

    def __init__(
        self,
        pop: Population,
        scale_factor: u.Quantity,
        conn_mode: str = 'sparse+low+rank',
        conn_param_type: type = brainscale.ETraceParam,
        n_rank: int = 20,
    ):
        super().__init__()

        # neuronal and synaptic dynamics
        self.pop = pop

        # delay for changes in post-synaptic neuron
        # Paul et al 2015 doi: https://doi.org/10.3389/fncel.2015.00029
        self.delay = brainstate.nn.Delay(
            jax.ShapeDtypeStruct(self.pop.varshape, brainstate.environ.dftype()),
            entries={'D': 1.8 * u.ms}
        )

        print('Loading synapse information ...')
        csr = load_syn(self.pop.flywire_version)

        # connectivity matrix
        self.conn_mode = conn_mode
        self.scale_factor = scale_factor

        if conn_mode == 'sparse+low+rank':
            # do not train sparse connection
            self.conn = brainstate.nn.SparseLinear(csr, b_init=None, param_type=brainstate.FakeState)

            # train LoRA weights
            self.lora = brainscale.nn.LoRA(
                in_features=self.pop.in_size,
                lora_rank=n_rank,
                out_features=self.pop.out_size,
                A_init=brainstate.init.LecunNormal(unit=u.mV),
                param_type=conn_param_type
            )

        else:
            raise ValueError('conn_mode must be either "sparse" or "sparse+low+rank"')

    def update(self, x=None):
        """
        Update the network state based on the current input.

        Args:
            x (Optional): External input to the network. Defaults to None.

        Returns:
            dict: A dictionary containing the spike, voltage, and conductance states.
        """
        # Update the input module for the neuron population delayed spikes
        pre_spk = self.delay.at('D')
        pre_spk = jax.lax.stop_gradient(pre_spk)

        # compute recurrent connections and update neurons
        if self.conn_mode == 'sparse+low+rank':
            inp = self.conn(brainevent.EventArray(pre_spk)) * self.scale_factor
            inp = inp + self.lora(pre_spk)
        else:
            raise ValueError('mode must be either "sparse" or "sparse+low+rank')

        if x is None:
            x = inp
        else:
            x += inp
        spk = self.pop(x)

        # update delay spikes
        self.delay.update(jax.lax.stop_gradient(spk))

        return spk


class NeuralActivity(brainstate.nn.Module):
    """
    A class to handle neural activity data for drosophila brain simulations.

    This class manages spike rates data from neural activity recordings and provides
    functionality to map between neuropil-level and neuron-level firing rates.

    Attributes:
        pop_size (brainstate.typing.Size): Size of the neuron population.
        neural_activity_id (str): Identifier for the neural activity dataset.
        conversion (str): Method for converting between neuropil and neuron firing rates ('unique' or 'weighted').
        spike_rates (ndarray): Array of spike rates for different neuropils.
        neuropils (ndarray): Names of the neuropils corresponding to spike_rates.
        neuropil_to_connectivity (NestedDict): Mapping from neuropils to their connectivity information.

    Args:
        pop_size (brainstate.typing.Size): Size of the neuron population.
        flywire_version (str): Version of the flywire dataset to use ('783' or '630').
        neural_activity_id (str, optional): Identifier for the neural activity dataset.
            Defaults to '2017-10-26_1'.
        neural_activity_max_fr (u.Quantity, optional): Maximum firing rate for scaling.
            Defaults to 120 Hz.

    Raises:
        ValueError: If ``flywire_version`` is not one of the supported versions or
                   if conversion is not 'unique' or 'weighted'.
    """

    def __init__(
        self,
        pop: Population,
        flywire_version: str,
        neural_activity_id: str = '2017-10-26_1',
        neural_activity_max_fr: u.Quantity = 120 * u.Hz,
        param_type: type = brainscale.ETraceParam,
        seed: int = 2025,
        n_embed: int = 10,
        bin_size: u.Quantity = 0.1 * u.Hz,
    ):
        super().__init__()
        self.pop = pop

        # uniform binning
        self.rng = np.random.RandomState(seed)
        self.n_embed = n_embed
        self.bin_size = bin_size
        print('Loading neural activity information ...')

        # neural activity data
        self.neural_activity_id = neural_activity_id
        data = np.load(f'./data/spike_rates/ito_{neural_activity_id}_spike_rate.npz')
        self.neuropils = data['areas'][1:]
        self.spike_rates = u.math.asarray(data['rates'][1:] * neural_activity_max_fr).T  # [time, neuropil]
        max_firing_rate = self.spike_rates.max()
        print('Maximum firing rate:', max_firing_rate)
        self.max_firing_rate = np.ceil(max_firing_rate.to_decimal(u.Hz))
        self.bins = np.arange(0, self.max_firing_rate, self.bin_size.to_decimal(u.Hz))
        self.bins = np.append(self.bins, neural_activity_max_fr.to_decimal(u.Hz))

        # connectivity data, which show a given neuropil contains which connections
        print('Loading connectivity information ...')
        if flywire_version in ['783', 783]:
            conn_path = 'data/783_connections_processed.csv'
        elif flywire_version in ['630', 630]:
            conn_path = 'data/630_connections_processed.csv'
        else:
            raise ValueError('flywire_version must be either "783" or "630"')
        connectivity = pd.read_csv(conn_path)
        neuropil_to_connectivity = brainstate.util.NestedDict()
        for i, neuropil in enumerate(self.neuropils):
            # find out all connections (spike source) to a given neuropil
            position = connectivity['neuropil'] == neuropil
            pre_indices = connectivity['pre_index'][position].values
            post_indices = connectivity['post_index'][position].values
            syn_count = connectivity['syn_count'][position].values
            # pre/post-synaptic indices and counts
            (
                pre_indices,
                pre_counts,
                post_indices,
                post_counts,
            ) = count_pre_post_connections(pre_indices, post_indices, syn_count)
            pre_weights = pre_counts / pre_counts.sum()
            post_weights = post_counts / post_counts.sum()
            neuropil_to_connectivity[neuropil] = brainstate.util.NestedDict(
                pre_indices=pre_indices,
                post_indices=post_indices,
                pre_weights=pre_weights,
                post_weights=post_weights,
            )
        self.neuropil_to_connectivity = neuropil_to_connectivity

        # neural activity conversion
        self.embedding = jnp.asarray(
            self.rng.normal(0.0, 1.0, (self.n_neuropil, self.n_embed))
        )
        self.neuropil2neuron = brainstate.nn.Sequential(
            brainscale.nn.Linear(
                self.n_neuropil * self.n_embed,
                self.pop.varshape,
                w_init=brainstate.init.KaimingNormal(unit=u.mV),
                b_init=brainstate.init.ZeroInit(unit=u.mV),
                param_type=param_type
            ),
            brainscale.nn.ReLU()
        )

    def update(self, embedding):
        noise_weight = self.neuropil2neuron(embedding)

        # excite neurons
        refractory = self.pop.get_refractory()

        # excitation
        brainstate.nn.poisson_input(
            freq=20 * u.Hz,
            num_input=1,
            weight=noise_weight,
            target=self.pop.v,
            refractory=refractory,
        )

    @property
    def n_neuropil(self) -> int:
        return self.spike_rates.shape[1]

    @property
    def n_time(self) -> int:
        return self.spike_rates.shape[0]

    def neuropil_to_bin_indices(self, neuropil_fr: u.Quantity[u.Hz]):
        """
        Convert neuropil firing rates to bin indices.

        This method maps the given neuropil firing rates to the corresponding bin indices
        based on the pre-defined bins stored in the class instance.

        Args:
            neuropil_fr (u.Quantity[u.Hz]): Firing rates for each neuropil, specified in Hertz units.

        Returns:
            jnp.ndarray: An array of bin indices corresponding to each input firing rate.
        """
        # Convert the neuropil firing rates to decimal values in Hertz and digitize them
        # based on the pre-defined bins stored in the class instance.
        bin_indices = jnp.digitize(neuropil_fr.to_decimal(u.Hz), self.bins, False)
        return bin_indices

    @brainstate.compile.jit(static_argnums=0)
    def neuropil_fr_to_embedding(self, neuropil_fr: u.Quantity[u.Hz]):
        """
        Convert firing rates from neuropil-level to embedding-level.
        This method maps firing rates from individual neuropils to the embedding level
        by applying a one-hot encoding to the firing rates.
        
        Args:
            neuropil_fr (u.Quantity[u.Hz]): Firing rates for each neuropil, specified in Hertz units.
            
        Returns:
            u.Quantity[u.Hz]: Embedding-level firing rates, specified in Hertz units.
        """

        def convert(fr):
            # convert firing rates to bins
            bin_indices = self.neuropil_to_bin_indices(fr)
            embed = self.embedding[bin_indices].flatten()
            return embed

        if neuropil_fr.ndim == 1:
            return convert(neuropil_fr)
        elif neuropil_fr.ndim == 2:
            return jax.vmap(convert)(neuropil_fr)
        else:
            raise ValueError

    def neuron_to_neurpil_fr(self, neuron_fr: u.Quantity[u.Hz]):
        """
        Convert firing rates from neuron-level to neuropil-level.

        This method maps firing rates from individual neurons to the neuropil level by
        aggregating the neuron firing rates using weighted sums based on their
        pre-synaptic connections to each neuropil.

        Args:
            neuron_fr (u.Quantity[u.Hz]): Firing rates for each neuron in the population,
                specified in Hertz units.

        Returns:
            u.Quantity[u.Hz]: Firing rates for each neuropil, specified in Hertz units.

        Note:
            The conversion applies the weights defined by the selected conversion method
            ('unique', 'weighted', or 'average') during class initialization.
        """
        neuropil_fr = []
        for i, neuropil in enumerate(self.neuropils):
            # find out all connections (spike source) to a given neuropil
            pre_indices = self.neuropil_to_connectivity[neuropil]['pre_indices']
            pre_weights = self.neuropil_to_connectivity[neuropil]['pre_weights']
            neuropil_fr.append(u.math.sum(neuron_fr[pre_indices] * pre_weights))
        return u.math.asarray(neuropil_fr)

    def read_neuropil_fr(self, i):
        """
        Read the spike rate of a specific neuropil at a given index.

        Args:
            i (int): The index of the spike rate to read from the stored spike rates.

        Returns:
            u.Quantity: The spike rate at the specified index.
        """
        return self.spike_rates[i]

    def iter_data(
        self,
        batch_size: int,
        drop_last: bool = False
    ):
        """
        Iterate over the neural activity data in batches.

        Args:
            batch_size (int): The size of each batch.
            drop_last (bool, optional): Whether to drop the last batch if it is smaller

        Yields:
            Tuple: A tuple containing the spike rates and neuropils for each batch.
        """
        for i in range(1, self.n_time, batch_size):
            if i + batch_size > self.n_time:
                if drop_last:
                    break
                batch_size = self.n_time - i

            input_neuropil_fr = self.spike_rates[i - 1:i + batch_size - 1]
            output_neuropil_fr = self.spike_rates[i:i + batch_size]
            input_embed = self.neuropil_fr_to_embedding(input_neuropil_fr)
            yield (
                input_embed,
                output_neuropil_fr,
            )


class FiringRateNetwork(brainstate.nn.Module):
    """
    A neural network model for simulating firing rates in the Drosophila brain.

    This class implements a spiking neural network that models the firing rate dynamics
    of neurons in the Drosophila brain. It integrates population dynamics, synaptic
    connections, and neural activity data to provide a comprehensive simulation framework.

    Inherits from brainstate.nn.Module to leverage the neural network functionality.

    Parameters
    ----------
    input_method : str, optional
        Method for processing input to neurons, default is 'relu'.
    flywire_version : str, optional
        Version of the FlyWire connectome to use, default is '630'.
    neural_activity_id : str, optional
        Identifier for the neural activity dataset, default is '2017-10-26_1'.
    neural_activity_max_fr : u.Quantity, optional
        Maximum firing rate for neural activity, default is 100 Hz.
    fr_conversion : str, optional
        Method for converting firing rates between representations, default is 'weighted'.

    Attributes
    ----------
    input_method : str
        Method used for processing neural inputs.
    flywire_version : str
        Version of the FlyWire connectome being used.
    pop_inp : PopulationInput
        Handler for input to the neural population.
    neural_activity : NeuralActivity
        Container for neural activity data and conversion utilities.
    interaction : Interaction
        Neural network for simulating population dynamics.
    """

    def __init__(
        self,
        flywire_version: str = '630',
        neural_activity_id: str = '2017-10-26_1',
        neural_activity_max_fr: u.Quantity = 100. * u.Hz,
        n_rank: int = 20,
        scale_factor=0.3 * 0.275 / 7 * u.mV,
        conn_param_type: type = brainscale.ETraceParam,
        input_param_type: type = brainscale.ETraceParam,
        sampling_rate: u.Quantity = 1.2 * u.Hz,
        seed: int = 2025,
        n_embed: int = 10,
        bin_size: u.Quantity = 0.1 * u.Hz,
    ):
        super().__init__()

        # parameters
        self.flywire_version = flywire_version
        self.n_sample_step = int(1 / sampling_rate / brainstate.environ.get_dt())

        # population and its input
        self.pop = Population(
            flywire_version,
            V_init=v_init,
            g_init=g_init,
            tau_ref=5.0 * u.ms
        )

        # neural activity data
        self.neural_activity = NeuralActivity(
            pop=self.pop,
            flywire_version=flywire_version,
            neural_activity_id=neural_activity_id,
            neural_activity_max_fr=neural_activity_max_fr,
            param_type=input_param_type,
            seed=seed,
            n_embed=n_embed,
            bin_size=bin_size,
        )

        # network
        self.interaction = Interaction(
            self.pop,
            n_rank=n_rank,
            scale_factor=scale_factor,
            conn_param_type=conn_param_type,
        )

    def count_neuropil_fr(self, length: int = None):
        if length is None:
            length = self.n_sample_step
        neuron_fr = self.pop.spike_count.value / (length * brainstate.environ.get_dt())
        neuron_fr = neuron_fr.to(u.Hz)
        fun = self.neural_activity.neuron_to_neurpil_fr
        for i in range(neuron_fr.ndim - 1):
            fun = jax.vmap(fun)
        neuropil_fr = fun(neuron_fr)
        return neuropil_fr

    def update(self, i, last_neuropil_fr: u.Quantity):
        with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
            # give inputs
            self.neural_activity.update(last_neuropil_fr)

            # update network
            spk = self.interaction.update()
            return spk

    @brainstate.compile.jit(static_argnums=0)
    def simulate(self, neuropil_fr, indices):
        def step_run(i):
            self.update(i, neuropil_fr)

        self.pop.reset_spk_count()
        brainstate.compile.for_loop(step_run, indices)
        frs = self.count_neuropil_fr(indices.shape[0])
        return frs


class Trainer:
    """
    A trainer for optimizing neural network models of the Drosophila brain.

    This class handles the training process for firing rate network models, implementing
    optimization strategies to match target neural activity patterns. It sets up the model,
    configures the optimizer, and provides methods for training and evaluation.

    Parameters
    ----------
    lr : float, optional
        Learning rate for the optimizer, default is 1e-3.
    etrace_decay : float or None, optional
        Decay factor for eligibility traces, default is 0.99. If None, uses parameter-dimension VJP.
    grad_clip : float or None, optional
        Maximum norm for gradient clipping, default is 1.0. If None, no clipping is applied.
    neural_activity_id : str, optional
        Identifier for the neural activity dataset, default is '2017-10-26_1'.
    input_method : str, optional
        Method for processing neural input, default is 'relu'.
    fr_conversion : str, optional
        Method for converting firing rates between representations, default is 'weighted'.
    flywire_version : str, optional
        Version of the FlyWire connectome to use, default is '630'.
    max_firing_rate : u.Quantity, optional
        Maximum firing rate for neural activity, default is 100 Hz.
    sampling_rate : u.Quantity, optional
        Rate at which to sample neural activity, default is 1.2 Hz.

    Attributes
    ----------
    etrace_decay : float or None
        Decay factor for eligibility traces.
    grad_clip : float or None
        Maximum norm for gradient clipping.
    indices : ndarray
        Time indices for simulation and training.
    n_sim : int
        Number of simulation steps before training.
    target : FiringRateNetwork
        The neural network model being trained.
    trainable_weights : dict
        Dictionary of trainable parameters in the model.
    opt : brainstate.optim.Adam
        Optimizer for updating model parameters.
    filepath : str
        Path for saving training results.
    """

    def __init__(
        self,
        sim_before_train: float = 0.5,
        lr: float = 1e-3,
        etrace_decay: float | None = 0.99,
        grad_clip: float | None = 1.0,

        # network parameters
        neural_activity_id: str = '2017-10-26_1',
        flywire_version: str = '630',
        max_firing_rate: u.Quantity = 100. * u.Hz,
        loss_fn: str = 'mse',
        vjp_method: str = 'single-step',
        n_rank: int = 20,
        conn_param_type: type = brainscale.ETraceParam,
        input_param_type: type = brainscale.ETraceParam,
        scale_factor=0.01 * u.mV,
        seed: int = 2025,
        n_embed: int = 10,
        bin_size: u.Quantity = 0.1 * u.Hz,
    ):
        # parameters
        self.sim_before_train = sim_before_train
        self.etrace_decay = etrace_decay
        self.grad_clip = grad_clip
        self.loss_fn = loss_fn
        self.vjp_method = vjp_method

        # population and its input
        self.target = FiringRateNetwork(
            flywire_version=flywire_version,
            neural_activity_id=neural_activity_id,
            neural_activity_max_fr=max_firing_rate,
            conn_param_type=conn_param_type,
            input_param_type=input_param_type,
            scale_factor=scale_factor,
            n_rank=n_rank,
            seed=seed,
            n_embed=n_embed,
            bin_size=bin_size,
        )

        # optimizer
        self.trainable_weights = brainstate.graph.states(self.target, brainstate.ParamState)
        self.opt = brainstate.optim.Adam(lr)
        self.opt.register_trainable_weights(self.trainable_weights)

        # train save path
        time_ = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.filepath = (
            f'results/v4/'
            f'{flywire_version}#'
            f'{neural_activity_id}#'
            f'{max_firing_rate / u.Hz}Hz#'
            f'{etrace_decay}#'
            f'{loss_fn}#'
            f'{conn_param_type.__name__}#'
            f'{input_param_type.__name__}#'
            f'{scale_factor.to_decimal(u.mV):5f}#'
            f'{n_rank}#'
            f'{sim_before_train}#'
            f'{seed}#'
            f'{n_embed}#'
            f'{bin_size.to_decimal(u.Hz)}#'
            f'{time_}'
        )

    def get_loss(self, current_neuropil_fr, target_neuropil_fr):
        if self.loss_fn == 'mse':
            loss = braintools.metric.squared_error(current_neuropil_fr, target_neuropil_fr)
        elif self.loss_fn == 'mae':
            loss = braintools.metric.absolute_error(current_neuropil_fr, target_neuropil_fr)
        elif self.loss_fn == 'huber':
            loss = braintools.metric.huber_loss(current_neuropil_fr, target_neuropil_fr)
        elif self.loss_fn == 'cosine_distance':
            loss = braintools.metric.cosine_distance(current_neuropil_fr, target_neuropil_fr)
        elif self.loss_fn == 'log_cosh':
            loss = braintools.metric.log_cosh(current_neuropil_fr, target_neuropil_fr)
        else:
            raise ValueError(f'Unknown loss function: {self.loss_fn}')
        return loss

    @brainstate.compile.jit(static_argnums=0)
    def train(self, input_embed, target_neuropil_fr):
        indices = np.arange(self.target.n_sample_step)

        # last_neuropil_fr: [n_batch, n_neuropil]
        # target_neuropil_fr: [n_batch, n_neuropil]
        n_batch = input_embed.shape[0]

        # model
        if self.etrace_decay is None or self.etrace_decay == 0.:
            model = brainscale.ParamDimVjpAlgorithm(self.target, vjp_method=self.vjp_method)
        else:
            model = brainscale.IODimVjpAlgorithm(self.target, self.etrace_decay, vjp_method=self.vjp_method)
        brainstate.nn.vmap_init_all_states(self.target, axis_size=n_batch, state_tag='hidden')

        @brainstate.augment.vmap_new_states(
            state_tag='etrace',
            axis_size=n_batch,
            in_states=self.target.states('hidden')
        )
        def init():
            model.compile_graph(0, input_embed[0])
            model.show_graph()

        init()

        # simulation without record eligibility trace
        n_sim = int(self.sim_before_train * self.target.n_sample_step)
        if n_sim > 0:
            batch_target = brainstate.nn.Vmap(self.target, vmap_states='hidden', in_axes=(None, 0))
            brainstate.compile.for_loop(
                lambda i: batch_target(i, input_embed),
                indices[:n_sim],
            )

        # simulation with eligibility trace recording
        self.target.pop.reset_spk_count(n_batch)
        model = brainstate.nn.Vmap(model, vmap_states=('hidden', 'etrace'), in_axes=(None, 0))
        brainstate.compile.for_loop(
            lambda i: model(i, input_embed),
            indices[n_sim:-1],
        )

        # training
        def loss_fn(i):
            spk = model(i, input_embed)
            current_neuropil_fr = self.target.count_neuropil_fr(self.target.n_sample_step - n_sim)
            loss_ = self.get_loss(current_neuropil_fr, target_neuropil_fr).mean()
            return u.get_mantissa(loss_), current_neuropil_fr

        grads, loss, neuropil_fr = brainstate.augment.grad(
            loss_fn, self.trainable_weights, return_value=True, has_aux=True
        )(indices[-1])
        max_g = jax.tree.map(lambda x: jnp.abs(x).max(), grads)
        if self.grad_clip is not None:
            grads = brainstate.functional.clip_grad_norm(grads, self.grad_clip)
        self.opt.update(grads)

        target_bin_indices = jax.vmap(self.target.neural_activity.neuropil_to_bin_indices)(target_neuropil_fr)
        predict_bin_indices = jax.vmap(self.target.neural_activity.neuropil_to_bin_indices)(neuropil_fr)
        acc = jnp.mean(jnp.asarray(target_bin_indices == predict_bin_indices, dtype=jnp.float32))

        return loss, neuropil_fr, max_g, acc

    def show_res(self, neuropil_fr, target_neuropil_fr, i_epoch, i_batch, n_neuropil_per_fig=10):
        fig, gs = braintools.visualize.get_figure(n_neuropil_per_fig, 2, 2., 10)
        for i in range(n_neuropil_per_fig):
            xticks = (i + 1 == n_neuropil_per_fig)
            fig.add_subplot(gs[i, 0])
            barplot(
                self.target.neural_activity.neuropils,
                neuropil_fr[i].to_decimal(u.Hz),
                title='Simulated FR',
                xticks=xticks
            )
            fig.add_subplot(gs[i, 1])
            barplot(
                self.target.neural_activity.neuropils,
                target_neuropil_fr[i].to_decimal(u.Hz),
                title='True FR',
                xticks=xticks
            )
        filename = f'{self.filepath}/images/neuropil_fr-at-epoch-{i_epoch}-batch-{i_batch}.png'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
        plt.close()

    def round1_train(
        self,
        train_epoch: int,
        batch_size: int = 128,
        checkpoint_path: str = None,
    ):
        if checkpoint_path is not None:
            braintools.file.msgpack_load(checkpoint_path, self.target.states(brainstate.ParamState))
            filepath = os.path.join(os.path.dirname(checkpoint_path), 'new')
            os.makedirs(filepath, exist_ok=True)
        else:
            filepath = self.filepath

        # training process
        os.makedirs(filepath, exist_ok=True)
        with open(f'{filepath}/losses.txt', 'w') as file:

            # training
            min_loss = np.inf
            for i_epoch in range(train_epoch):
                i_batch = 0
                all_loss = []
                for input_embed, target_neuropil_fr in self.target.neural_activity.iter_data(
                    batch_size=batch_size, drop_last=True
                ):
                    t0 = time.time()
                    res = self.train(input_embed, target_neuropil_fr)
                    loss, neuropil_fr, max_g, acc = jax.block_until_ready(res)
                    t1 = time.time()

                    output(
                        file,
                        f'epoch = {i_epoch}, '
                        f'batch = {i_batch}, '
                        f'loss = {loss:.5f}, '
                        f'bin acc = {acc:.5f}, '
                        f'lr = {self.opt.lr():.6f}, '
                        f'time = {t1 - t0:.5f}s'
                    )
                    output(file, f'max_g = {max_g}')

                    i_batch += 1
                    all_loss.append(loss)
                self.show_res(neuropil_fr, target_neuropil_fr, i_epoch, '')
                self.opt.lr.step_epoch()

                # save checkpoint
                loss = np.mean(all_loss)
                output(file, f'epoch = {i_epoch}, loss = {loss:.5f}, lr = {self.opt.lr():.6f}')
                if loss < min_loss:
                    braintools.file.msgpack_save(
                        f'{filepath}/best-checkpoint.msgpack',
                        self.target.states(brainstate.ParamState),
                    )
                    min_loss = loss


def first_round_train():
    checkpoint_path = (
        'results/v4/630#2017-10-26_1#100.0Hz#0.99#mse#ETraceParam#ETraceParam#'
        '0.000825#20#0.1#2025#10#0.1#2025-03-24-10-54-07/best-checkpoint-round3.msgpack'
    )
    # checkpoint_path = None

    train_epoch = 500
    if checkpoint_path is None:
        neural_activity_id = '2017-10-26_1'
        flywire_version = '630'
        lr, etrace_decay, scale_factor, bin_size = 1e-2, 0.99, 0.0825 / 100 * u.mV, 0.5 * u.Hz
    else:
        lr = brainstate.optim.StepLR(1e-3, step_size=50, gamma=0.9)
        settings = checkpoint_path.split('/')[2].split('#')
        flywire_version = settings[0]
        neural_activity_id = settings[1]
        etrace_decay = float(settings[3])
        scale_factor = float(settings[7]) * u.mV
        bin_size = float(settings[12]) * u.Hz

    brainstate.environ.set(dt=0.2 * u.ms)
    trainer = Trainer(
        lr=lr,
        etrace_decay=etrace_decay,
        sim_before_train=0.1,
        neural_activity_id=neural_activity_id,
        flywire_version=flywire_version,
        max_firing_rate=100.0 * u.Hz,
        loss_fn='mse',
        scale_factor=scale_factor,
        conn_param_type=brainscale.ETraceParam if etrace_decay != 0. else brainscale.NonTempParam,
        input_param_type=brainscale.ETraceParam if etrace_decay != 0. else brainscale.NonTempParam,
        bin_size=bin_size,
    )
    trainer.round1_train(
        train_epoch=train_epoch,
        batch_size=128,
        checkpoint_path=checkpoint_path
    )


def second_round_train():
    lr = 1e-2
    filepath = 'results/v4/630#2017-10-26_1#100.0Hz#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#2025-03-18-21-02-47'

    settings = filepath.split('/')[-1].split('#')
    flywire_version = settings[0]
    neural_activity_id = settings[1]
    max_firing_rate = float(settings[2].split('Hz')[0]) * u.Hz
    etrace_decay = float(settings[3])
    loss_fn = settings[4]
    scale_factor = float(settings[7]) * u.mV
    n_rank = int(settings[8])
    sim_before_train = float(settings[9])

    brainstate.environ.set(dt=0.2 * u.ms)
    trainer = Trainer(
        # lr=lr,
        lr=brainstate.optim.StepLR(lr, step_size=1, gamma=0.9),
        # lr=brainstate.optim.StepLR(1e-1, step_size=5, gamma=0.9),
        etrace_decay=etrace_decay,
        sim_before_train=sim_before_train,
        neural_activity_id=neural_activity_id,
        flywire_version=flywire_version,
        max_firing_rate=max_firing_rate,
        loss_fn=loss_fn,
        scale_factor=scale_factor,
        conn_param_type=brainscale.ETraceParam if etrace_decay != 0. else brainscale.NonTempParam,
        input_param_type=brainscale.ETraceParam if etrace_decay != 0. else brainscale.NonTempParam,
        n_rank=n_rank
    )
    trainer.round2_train(filepath, 100, batch_size=128)


def example_to_simulate():
    import matplotlib.pyplot as plt
    brainstate.environ.set(dt=0.2 * u.ms)
    net = FiringRateNetwork(
        flywire_version='630',
        neural_activity_id='2017-10-26_1',
        neural_activity_max_fr=100 * u.Hz,
        conn_param_type=brainscale.NonTempParam,
        input_param_type=brainscale.NonTempParam,
        n_rank=20,
        # scale_factor=0.0825 / 40 * u.mV,
        scale_factor=0.0825 / 100 * u.mV,
    )
    brainstate.nn.init_all_states(net)

    neuropil_fr = net.neural_activity.read_neuropil_fr(0)
    t0 = 0.0 * u.ms
    t1 = net.n_sample_step * brainstate.environ.get_dt()

    for i in range(10):
        neuropil_fr = net.simulate(neuropil_fr, t0, t0 + t1)
        fig, gs = braintools.visualize.get_figure(2, 1, 3, 10.0)
        fig.add_subplot(gs[0, 0])
        barplot(net.neural_activity.neuropils, neuropil_fr.to_decimal(u.Hz), title='Simulated FR')
        fig.add_subplot(gs[1, 0])
        target_neuropil_fr = net.neural_activity.read_neuropil_fr(i + 1)
        barplot(net.neural_activity.neuropils, target_neuropil_fr.to_decimal(u.Hz), title='True FR')
        plt.show()
        t0 += t1


def example_to_load():
    brainstate.environ.set(dt=0.2 * u.ms)

    filepath = 'results/v4/630#2017-10-26_1#100.0Hz#None#mse#NonTempParam#NonTempParam#0.000825#20#0.1#2025-03-17-12-38-29'
    filepath = 'results/v4/630#2017-10-26_1#100.0Hz#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#2025-03-18-21-02-47'
    setting = filepath.split('/')[2].split('#')
    flywire_version = setting[0]
    neural_activity_id = setting[1]
    max_firing_rate = float(setting[2].split('Hz')[0]) * u.Hz
    etrace_decay = eval(setting[3])
    loss_fn = setting[4]
    conn_param_type = setting[5]
    input_param_type = setting[6]
    scale_factor = float(setting[7])
    n_rank = int(setting[8])
    sim_before_train = float(setting[9])

    net = FiringRateNetwork(
        flywire_version=flywire_version,
        neural_activity_id=neural_activity_id,
        neural_activity_max_fr=max_firing_rate,
        conn_param_type=getattr(brainscale, conn_param_type),
        input_param_type=getattr(brainscale, input_param_type),
        n_rank=n_rank,
        scale_factor=scale_factor * u.mV,
    )
    brainstate.nn.init_all_states(net)
    braintools.file.msgpack_load(
        os.path.join(filepath, 'best-checkpoint.msgpack'),
        net.states(brainstate.ParamState),
    )

    n_time = 100
    # n_time = net.neural_activity.n_time
    n_sim = int(sim_before_train * net.n_sample_step)
    t0 = 0.0 * u.ms
    t1 = net.n_sample_step * brainstate.environ.get_dt()
    simulated_neuropil_fr = []
    indices = np.arange(net.n_sample_step)
    # neuropil_fr = net.neural_activity.read_neuropil_fr(0)
    for i in tqdm(range(n_time)):
        if i < 10:
            neuropil_fr = net.neural_activity.read_neuropil_fr(i)
        indices += i * net.n_sample_step
        net.simulate(neuropil_fr, indices[:n_sim])
        neuropil_fr = net.simulate(neuropil_fr, indices[n_sim:])
        simulated_neuropil_fr.append(neuropil_fr.to_decimal(u.Hz))
        t0 += t1
    simulated_neuropil_fr = np.asarray(simulated_neuropil_fr)  # [n_time, n_neuropil]
    # np.save(os.path.join(filepath, 'simulated_neuropil_fr'), simulated_neuropil_fr)

    import matplotlib.pyplot as plt
    experimental_neuropil_fr = np.asarray(net.neural_activity.spike_rates / u.Hz)  # [n_time, n_neuropil]
    times = np.arange(simulated_neuropil_fr.shape[0]) * t1
    num = 5
    for ii in range(0, net.neural_activity.n_neuropil, num):
        fig, gs = braintools.visualize.get_figure(num, 2, 4, 8.0)
        for i in range(num):
            # plot simulated neuropil firing rate data
            fig.add_subplot(gs[i, 0])
            data = simulated_neuropil_fr[:, i + ii]
            plt.plot(times, data)
            plt.ylim(0., data.max() * 1.05)
            # plot experimental neuropil firing rate data
            fig.add_subplot(gs[i, 1])
            data = experimental_neuropil_fr[:n_time, i + ii]
            plt.plot(times, data)
            plt.ylim(0., data.max() * 1.05)
        plt.show()


if __name__ == '__main__':
    pass
    first_round_train()
    # second_round_train()
    # example_to_simulate()
    # example_to_load()

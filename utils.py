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

import datetime
import os
import subprocess
from pathlib import Path
from typing import NamedTuple, Callable

import brainevent
import brainscale
import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
from scipy import optimize, signal
from tqdm import tqdm


def v_init(shape):
    """
    Initialize membrane voltage values for neurons with random normal distribution.

    This function creates an array of initial membrane voltages using a random normal
    distribution with mean 0.0 and standard deviation 0.8. The values are scaled to
    millivolts using the brainunit library.

    Parameters
    ----------
    shape : tuple
        The shape of the output array, defining the dimensions of the neuron population.

    Returns
    -------
    brainstate.random.RandomState.normal : u.Quantity
        An array of initialized membrane voltages with the specified shape, drawn from
        a normal distribution and scaled to millivolts.
    """
    return brainstate.random.RandomState(42).normal(0., 0.8, shape) * u.mV


def g_init(shape):
    """
    Initialize synaptic conductance values for neurons with random uniform distribution.

    This function creates an array of initial synaptic conductances using a random uniform
    distribution between 0.0 and 0.2. The values are scaled to millivolts using the
    brainunit library.

    Parameters
    ----------
    shape : tuple
        The shape of the output array, defining the dimensions of the neuron population.

    Returns
    -------
    brainstate.random.RandomState.uniform : u.Quantity
        An array of initialized synaptic conductances with the specified shape, drawn from
        a uniform distribution and scaled to millivolts.
    """
    return brainstate.random.RandomState(2025).uniform(0., 0.2, shape) * u.mV


def get_gpu_info() -> str:
    """
    Retrieve the name of the first available NVIDIA GPU.

    This function attempts to get GPU information by executing the nvidia-smi command
    and parsing its output. It extracts the GPU name from the command output and
    formats it by joining the name components with hyphens.

    Returns
    -------
    str
        The name of the first available NVIDIA GPU formatted with hyphens between words.
        Returns 'Unknown' if no GPU is found or if an error occurs during the query.

    Examples
    --------
    >>> get_gpu_info()
    'GeForce-RTX-3080'

    Notes
    -----
    This function requires the nvidia-smi tool to be installed and accessible in the
    system path. It will fail gracefully by returning 'Unknown' if the command
    cannot be executed or returns an error.
    """
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True)
        gpu_names = output.strip().split('\n')
        return '-'.join(str(gpu_names[0]).split(' ')[2:])
    except Exception as e:
        return 'Unknown'


def df_f_to_firing_rate(
    df_f,
    tau_rise: float = 0.07,
    tau_decay: float = 0.5,
    dt: float = 0.033,
    method: str = 'deconvolution'
):
    """
    Convert calcium imaging ΔF/F signals into estimated neuronal firing rates.

    Parameters:
    -----------
    df_f : array-like
        The ΔF/F calcium signal time series.
    tau_rise : float, optional
        Rise time constant of calcium transient in seconds. Default is 0.07s.
    tau_decay : float, optional
        Decay time constant of calcium transient in seconds. Default is 0.5s.
    dt : float, optional
        Time step in seconds. Default is 0.033s (30 Hz).
    method : str, optional
        Method for conversion:
        - 'deconvolution': Wiener deconvolution (default)
        - 'derivative': Simple derivative method
        - 'foopsi': Fast non-negative deconvolution

    Returns:
    --------
    firing_rate : ndarray
        Estimated neuronal firing rate in events/second.
    """
    df_f = np.array(df_f)

    if method == 'derivative':
        # Simple derivative method
        # Taking derivative and rectifying
        d_df_f = np.diff(df_f, prepend=df_f[0])
        firing_rate = np.maximum(0, d_df_f) / dt

    elif method == 'deconvolution':
        # Wiener deconvolution method
        # Create calcium impulse response function
        t = np.arange(0, 5 * tau_decay, dt)
        h = (1 - np.exp(-t / tau_rise)) * np.exp(-t / tau_decay)
        h = h / np.sum(h * dt)  # Normalize

        # Perform deconvolution in frequency domain
        n = len(df_f)
        # Zero padding
        df_f_padded = np.hstack((df_f, np.zeros(len(h))))
        h_padded = np.hstack((h, np.zeros(n)))

        # FFT
        df_f_fft = np.fft.fft(df_f_padded)
        h_fft = np.fft.fft(h_padded)

        # Signal-to-noise ratio estimation (SNR)
        # This is a simplified approach; ideally, estimate from data
        snr = 10  # Example SNR value
        noise_power = np.mean(np.abs(df_f) ** 2) / snr

        # Wiener deconvolution
        c = 1 / (h_fft + noise_power / (np.abs(h_fft) ** 2))
        firing_rate_fft = df_f_fft * c
        firing_rate_full = np.real(np.fft.ifft(firing_rate_fft))

        # Take only the relevant part and rectify
        firing_rate = np.maximum(0, firing_rate_full[:n])

    elif method == 'foopsi':
        # Fast non-negative deconvolution (simplified version)
        g = np.exp(-dt / tau_decay)  # Discrete time constant

        # Objective function for optimization
        @numba.njit
        def objective(s):
            c = np.zeros_like(s)
            for t in range(1, len(s)):
                c[t] = g * c[t - 1] + s[t]
            return np.sum((c - df_f) ** 2) + 0.5 * np.sum(s)  # L1 regularization

        # Constraints: sparsity and non-negativity
        bounds = [(0, None) for _ in range(len(df_f))]

        # Initial guess
        s0 = np.zeros_like(df_f)

        # Optimization
        result = optimize.minimize(objective, s0, bounds=bounds, method='L-BFGS-B')
        firing_rate = result.x / dt

    else:
        raise ValueError(f"Method '{method}' not recognized. Use 'derivative', 'deconvolution', or 'foopsi'.")

    return firing_rate


@numba.njit
def _create_decay_matrix(frame_count, tau_frames):
    i_indices = np.arange(frame_count)[:, np.newaxis]
    j_indices = np.arange(frame_count)[np.newaxis, :]
    # Create matrix where element (i,j) contains (i-j) value
    time_diffs = i_indices - j_indices
    # Apply exponential decay only where i >= j (time_diffs >= 0)
    decay_matrix = np.where(time_diffs >= 0, np.exp(-time_diffs / tau_frames), 0)
    return decay_matrix


def deconvolve_dff_to_spikes(
    dff,
    tau: float = 0.8 * u.second,
    sampling_rate: float = 30. * u.Hz,
    lambda_reg: float = 0.01,
    smooth: bool = True
):
    """
    Convert df/f calcium signals to estimated spike rates using
    a simplified deconvolution approach.

    Parameters:
    -----------
    dff : numpy.ndarray
        The df/f calcium signal time series
    tau : float
        Decay time constant of calcium transient (in seconds)
    sampling_rate : float
        Sampling rate of the signal (Hz)
    lambda_reg : float
        Regularization parameter for sparsity
    smooth : bool
        Whether to apply smoothing to the df/f signal

    Returns:
    --------
    firing_rate : numpy.ndarray
        Estimated firing rate in Hz
    """
    # Convert tau to units of frames
    tau_frames = float(u.maybe_decimal(tau * sampling_rate))

    # Optional smoothing of df/f signal
    if smooth:
        window_size = min(21, len(dff) // 3)
        # Make window_size odd
        if window_size % 2 == 0:
            window_size += 1
        dff = signal.savgol_filter(dff, window_size, 2)

    # Create calcium decay matrix (for deconvolution)
    frame_count = len(dff)
    decay_matrix = _create_decay_matrix(frame_count, tau_frames)

    # Convert to sparse matrix for computational efficiency
    # decay_matrix = sparse.csr_matrix(decay_matrix)

    # Use non-negative least squares with regularization to solve for firing rates
    # min ||decay_matrix * spikes - dff||^2 + lambda * ||spikes||_1
    @numba.njit
    def objective(spikes):
        predicted_dff = decay_matrix.dot(spikes)
        error = np.sum((predicted_dff - dff) ** 2) + lambda_reg * np.sum(np.abs(spikes))
        return error

    # Initial guess of zero spikes
    x0 = np.zeros_like(dff)

    # Non-negative bound constraint
    bounds = [(0, None) for _ in range(frame_count)]

    # Minimize the objective function
    result = optimize.minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
    spikes = result.x

    # Convert to firing rate in Hz (spikes per second)
    firing_rate = spikes * sampling_rate

    return firing_rate


def deconvolve_dff_to_spikes_v2(
    dff,
    tau: float = 0.8 * u.second,
    sampling_rate: float = 30. * u.Hz,
    lambda_reg: float = 0.01,
    smooth: bool = True
):
    """
    Convert df/f calcium signals to estimated spike rates using
    a simplified deconvolution approach.

    Parameters:
    -----------
    dff : numpy.ndarray
        The df/f calcium signal time series
    tau : float
        Decay time constant of calcium transient (in seconds)
    sampling_rate : float
        Sampling rate of the signal (Hz)
    lambda_reg : float
        Regularization parameter for sparsity
    smooth : bool
        Whether to apply smoothing to the df/f signal

    Returns:
    --------
    firing_rate : numpy.ndarray
        Estimated firing rate in Hz
    """
    # Convert tau to units of frames
    tau_frames = float(u.maybe_decimal(tau * sampling_rate))

    # Optional smoothing of df/f signal
    if smooth:
        window_size = min(21, len(dff) // 3)
        # Make window_size odd
        if window_size % 2 == 0:
            window_size += 1
        dff = signal.savgol_filter(dff, window_size, 2)

    # Create calcium decay matrix (for deconvolution)
    frame_count = len(dff)
    decay_matrix = _create_decay_matrix(frame_count, tau_frames)

    # Convert to sparse matrix for computational efficiency
    # decay_matrix = sparse.csr_matrix(decay_matrix)

    # Use non-negative least squares with regularization to solve for firing rates
    # min ||decay_matrix * spikes - dff||^2 + lambda * ||spikes||_1
    @numba.njit
    def objective(spikes):
        predicted_dff = decay_matrix.dot(spikes)
        error = np.sum((predicted_dff - dff) ** 2) + lambda_reg * np.sum(np.abs(spikes))
        return error

    # Initial guess of zero spikes
    x0 = np.zeros_like(dff)

    # Non-negative bound constraint
    bounds = [(0, None) for _ in range(frame_count)]

    # Minimize the objective function
    result = optimize.minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
    spikes = result.x

    # Convert to firing rate in Hz (spikes per second)
    firing_rate = spikes * sampling_rate

    return firing_rate


def filter_region_response(region_response, cutoff=None, fs=None):
    """
    Filter a neural region response trace with a high-pass Butterworth filter.

    This function applies a high-pass Butterworth filter to the input neural region
    response data when both cutoff frequency and sampling rate are provided. If either
    parameter is missing, the original signal is returned unfiltered.

    Parameters
    ----------
    region_response : numpy.ndarray
        Neural region response trace data to be filtered.
    cutoff : float, optional
        High-pass filter cutoff frequency in Hz. If None, no filtering is applied.
    fs : float, optional
        Sampling frequency of the signal in Hz. If None, no filtering is applied.

    Returns
    -------
    numpy.ndarray
        Filtered (or original if fs is None) region response data.

    Notes
    -----
    The function uses a first-order high-pass Butterworth filter from scipy.signal
    with second-order sections (SOS) implementation for numerical stability.

    Despite the function name suggesting low-pass filtering, it actually
    implements a high-pass filter ('hp') as specified in the filter design.
    """
    if fs is not None:
        sos = signal.butter(1, cutoff, 'hp', fs=fs, output='sos')
        region_response_filtered = signal.sosfilt(sos, region_response)
    else:
        region_response_filtered = region_response

    return region_response_filtered


def trim_region_response(
    file_id,
    region_response,
    start_include=100,
    end_include=None
):
    """
    Trim artifacts from neural recording data by excluding problematic time periods.

    This function handles artifact removal in neural recordings by either:
    1. Using predefined trimming indices for known problematic recordings, or
    2. Applying default trimming based on start and end indices

    The function handles both multi-ROI recordings and single-dimension behavioral data.

    Parameters
    ----------
    file_id : str
        Identifier for the recording file. Used to match against known problematic recordings.
    region_response : numpy.ndarray
        Neural activity data with one of these shapes:
        - (n_rois, n_frames): Multi-ROI neural activity recordings
        - (n_frames,): Single-dimension binary behavioral response
    start_include : int, optional
        Index of first frame to include when using default trimming. Defaults to 100.
    end_include : int or None, optional
        Index of last frame to include when using default trimming. If None,
        includes all frames after start_include. Defaults to None.

    Returns
    -------
    numpy.ndarray
        Trimmed version of the input data with problematic periods removed.
        Maintains the same dimensionality as the input.

    Raises
    ------
    ValueError
        If region_response doesn't have either 1 or 2 dimensions.

    Notes
    -----
    The function maintains a dictionary of known problematic recordings with custom
    trimming indices to handle specific artifacts like dropouts or baseline shifts.
    """

    # Key: brain file id
    # Val: time inds to include
    # dropout halfway through
    brains_to_trim = {
        '2018-10-19_1': np.array(list(range(100, 900)) + list(range(1100, 2000))),  # transient dropout spikes
        '2017-11-08_1': np.array(list(range(100, 1900)) + list(range(2000, 4000))),  # baseline shift
        '2018-10-20_1': np.array(list(range(100, 1000)))
    }

    if file_id in brains_to_trim.keys():
        include_inds = brains_to_trim[file_id]
        if len(region_response.shape) == 2:
            region_response_trimmed = region_response[:, include_inds]
        elif len(region_response.shape) == 1:
            region_response_trimmed = region_response[include_inds]
        else:
            raise ValueError

    else:  # use default start / end
        if len(region_response.shape) == 2:
            region_response_trimmed = region_response[:, start_include:end_include]
        elif len(region_response.shape) == 1:
            region_response_trimmed = region_response[start_include:end_include]
        else:
            raise ValueError
    return region_response_trimmed


def barplot(neuropil_names, neuropil_fr, title='', xticks=True):
    """
    Create a bar plot of firing rates for different neuropils in the Drosophila brain.

    This function generates a matplotlib bar plot to visualize firing rates across
    different brain regions (neuropils), with customizable display options for
    axis labels and titles.

    Parameters
    ----------
    neuropil_names : array-like
        Names of the neuropil regions to be displayed on the x-axis.
    neuropil_fr : array-like
        Firing rates (in Hz) corresponding to each neuropil region.
    title : str, optional
        Title for the plot. If empty, no title is displayed. Default is ''.
    xticks : bool, optional
        Whether to display x-axis tick labels (neuropil names). When True,
        names are displayed rotated 90 degrees for readability. Default is True.

    Returns
    -------
    None
        The function creates a plot but does not return any values.

    Notes
    -----
    This function uses matplotlib for visualization and assumes that a figure
    has already been created or will be displayed/saved after calling this function.
    """
    x_pos = np.arange(len(neuropil_names))
    plt.bar(x_pos, neuropil_fr)
    if xticks:
        plt.xticks(x_pos, neuropil_names, rotation=90, fontsize=8)
        plt.xlabel('Neuropil')
    plt.ylabel('Firing Rate (Hz)')
    if title:
        plt.title(title)


def output(file, msg: str):
    """
    Write a message to both the console and a file.

    This function outputs the given message to the standard output (console) and
    writes it to the specified file, appending a newline character. The file is
    flushed immediately after writing to ensure data is committed to disk.

    Parameters
    ----------
    file : file-like object
        An open file-like object with write and flush methods. Typically this is
        a file opened in write or append mode.
    msg : str
        The message string to be output to console and written to the file.

    Returns
    -------
    None
        This function does not return any value.

    Examples
    --------
    >>> with open('log.txt', 'w') as f:
    ...     output(f, 'Processing started')
    Processing started
    """
    print(msg)
    file.write(msg + '\n')
    file.flush()


def get_tokenization_bins(spike_rates, bin_size, neural_activity_max_fr):
    """
    Create bins for discretizing firing rates in the neural activity data.

    This function generates bin edges for categorizing continuous firing rates into discrete bins.
    The bins start from 0 Hz and increase by the specified bin size until reaching the maximum
    observed firing rate, with a final bin extending to the maximum possible firing rate.

    Parameters
    ----------
    spike_rates : u.Quantity
        Array of spike rates with units (typically Hz) to be binned.
        Used to determine the maximum observed firing rate.
    bin_size : u.Quantity
        Size of each bin in Hz units (e.g., 0.1 Hz).
    neural_activity_max_fr : u.Quantity
        Maximum possible firing rate in Hz units. This value is added as the upper edge
        of the final bin to ensure all firing rates can be properly categorized.

    Returns
    -------
    u.Quantity
        An array of bin edges starting from 0 and increasing by bin_size until reaching
        the maximum firing rate, with the final value being neural_activity_max_fr.

    Notes
    -----
    The function ensures all firing rates can be binned by:
    1. Finding the maximum value in the provided spike_rates
    2. Creating regular bins from 0 to this maximum value
    3. Adding an extra bin edge at neural_activity_max_fr to capture any potential outliers
    """
    max_firing_rate = spike_rates.max()
    max_firing_rate = u.math.ceil(max_firing_rate)
    bins = u.math.arange(0 * u.Hz, max_firing_rate, bin_size)
    bins = u.math.append(bins, neural_activity_max_fr)
    return u.math.asarray(bins)


def neuropil_to_bin_indices(neuropil_fr: u.Quantity[u.Hz], bins: u.Quantity[u.Hz]):
    """
    Convert neuropil firing rates to bin indices.

    This method maps the given neuropil firing rates to the corresponding bin indices
    based on the pre-defined bins stored in the class instance.

    Args:
        neuropil_fr (u.Quantity[u.Hz]): Firing rates for each neuropil, specified in Hertz units.

    Returns:
        jnp.ndarray: An array of bin indices corresponding to each input firing rate.
    """
    assert bins.ndim == 1, 'bins must be a 1D array'
    bins = bins.to_decimal(u.Hz) if isinstance(bins, u.Quantity) else bins
    neuropil_fr = neuropil_fr.to_decimal(u.Hz) if isinstance(neuropil_fr, u.Quantity) else neuropil_fr

    # Convert the neuropil firing rates to decimal values in Hertz and digitize them
    # based on the pre-defined bins stored in the class instance.
    fn = lambda x: jnp.digitize(x, bins, False)
    for _ in range(neuropil_fr.ndim - 1):
        fn = jax.vmap(fn)
    bin_indices = fn(neuropil_fr)
    return bin_indices


def load_syn(flywire_version: str | int) -> brainevent.CSR:
    """
    Load synaptic connectivity data from the FlyWire connectome dataset.

    This function loads the neuronal connection data for the Drosophila brain from a
    specified FlyWire connectome version and constructs a Compressed Sparse Row (CSR)
    matrix representation of the synaptic connectivity.

    Parameters
    ----------
    flywire_version : str or int
        Version identifier for the FlyWire connectome dataset.
        Accepted values are '630', 630, '783', or 783.

    Returns
    -------
    brainevent.CSR
        A compressed sparse row matrix representing the synaptic connectivity,
        where each entry (i,j) represents the connection weight from neuron i to neuron j.

    Raises
    ------
    ValueError
        If the flywire_version is not one of the supported versions ('630', 630, '783', 783).

    Notes
    -----
    The function processes the connectivity data by:
    1. Loading neuron information to determine the total number of neurons
    2. Loading synaptic connections from the parquet file
    3. Sorting connections by presynaptic neuron indices
    4. Converting the data to CSR format
    """
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

    weight = jnp.asarray(weight, dtype=brainstate.environ.dftype())
    csr = brainevent.CSR((weight, indices, indptr), shape=(n_neuron, n_neuron))
    return csr


def split_train_test(
    length: int,
    split: float,
    batch_size: int
):
    """
    Split a dataset into training and testing sets based on specified proportions and batch size.

    This function divides the total dataset length into training and testing segments,
    ensuring the training set size is compatible with the specified batch size.

    Parameters
    ----------
    length : int
        Total number of samples in the dataset.
    split : float
        Proportion of data to use for training (0.0 to 1.0).
    batch_size : int
        Size of batches for training. The training set size will be adjusted to be
        divisible by this value (plus 1 sample).

    Returns
    -------
    tuple[int, int]
        A tuple containing (n_train, n_test) where:
        - n_train: Number of samples in the training set
        - n_test: Number of samples in the testing set

    Notes
    -----
    The function ensures that the training set size is compatible with batching by:
    1. Calculating the raw split based on the specified proportion
    2. Adjusting to be divisible by batch_size
    3. Adding 1 to ensure there's always at least one training sample
    """
    n_train = int(length * split) // batch_size * batch_size + 1
    n_test = length - n_train
    return n_train, n_test


@numba.njit
def count_pre_post_connections(
    pre_indices,
    post_indices,
    syn_counts,
):
    """
    Counts the total number of synaptic connections for each pre- and post-synaptic neuron.

    This function iterates through the provided pre-synaptic indices, post-synaptic indices,
    and their corresponding synaptic counts to determine the total number of connections
    associated with each neuron. It aggregates the synaptic counts for each pre- and
    post-synaptic neuron using dictionaries, then converts the results into NumPy arrays
    for efficient computation and use in other functions.

    Args:
        pre_indices (numpy.ndarray): A 1D NumPy array containing the indices of pre-synaptic neurons.
        post_indices (numpy.ndarray): A 1D NumPy array containing the indices of post-synaptic neurons.
        syn_counts (numpy.ndarray): A 1D NumPy array containing the number of synaptic connections
            between corresponding pre- and post-synaptic neurons.

    Returns:
        tuple: A tuple containing four NumPy arrays:
            - pre_indices (numpy.ndarray): A 1D NumPy array of unique pre-synaptic neuron indices.
            - pre_counts (numpy.ndarray): A 1D NumPy array of the total synaptic counts for each
              pre-synaptic neuron, corresponding to the indices in `pre_indices`.
            - post_indices (numpy.ndarray): A 1D NumPy array of unique post-synaptic neuron indices.
            - post_counts (numpy.ndarray): A 1D NumPy array of the total synaptic counts for each
              post-synaptic neuron, corresponding to the indices in `post_indices`.
    """
    pre_to_count = {}
    post_to_count = {}
    for pre, post, syn_count in zip(pre_indices, post_indices, syn_counts):
        if pre not in pre_to_count:
            pre_to_count[pre] = 0
        pre_to_count[pre] += syn_count

        if post not in post_to_count:
            post_to_count[post] = 0
        post_to_count[post] += syn_count

    pre_indices = []
    pre_counts = []
    for pre, count in pre_to_count.items():
        pre_indices.append(pre)
        pre_counts.append(count)

    post_indices = []
    post_counts = []
    for post, count in post_to_count.items():
        post_indices.append(post)
        post_counts.append(count)
    return np.asarray(pre_indices), np.asarray(pre_counts), np.asarray(post_indices), np.asarray(post_counts)


def read_setting(filepath) -> dict:
    """
    Read and parse model configuration settings from the first-round training results file.

    This function opens a text file containing serialized configuration settings from
    the first round of training, converts the stored Namespace object to a dictionary,
    and returns the parsed settings.

    Parameters
    ----------
    filepath : str
        Path to the directory containing the 'first-round-losses.txt' file
        which stores the serialized configuration settings.

    Returns
    -------
    dict
        A dictionary containing the model configuration settings parsed from the file.

    Notes
    -----
    The function expects the first line of the file to contain a serialized Namespace
    object. It replaces 'Namespace' with 'dict' to allow safe evaluation of the string
    as a Python dictionary.
    """
    with open(os.path.join(filepath, 'first-round-losses.txt'), 'r') as f:
        setting = eval(f.readline().replace('Namespace', 'dict'))
    return setting


class FilePath(NamedTuple):
    """
    A named tuple that stores and manages file paths for model checkpoints with their configuration parameters.

    This class provides utilities for creating standardized file paths based on model parameters and
    parsing existing file paths to extract those parameters.

    Attributes
    ----------
    flywire_version : str
        Version of the FlyWire connectome dataset ('630' or '783').
    neural_activity_id : str
        Identifier for the neural activity dataset (e.g., '2017-10-26_1').
    neural_activity_max_fr : u.Quantity[u.Hz]
        Maximum firing rate for normalization, specified in Hz units.
    etrace_decay : float
        Decay factor for eligibility traces (None or 0 for standard backprop).
    loss_fn : str
        Loss function used for training (e.g., 'mse', 'mae', 'huber').
    conn_param_type : type
        Parameter type for connection weights.
    input_param_type : type
        Parameter type for input weights.
    scale_factor : u.Quantity[u.mV]
        Scaling factor for synaptic weights, specified in mV units.
    n_rank : int
        Rank for low-rank approximation in LoRA.
    sim_before_train : float
        Proportion of steps to simulate before recording eligibility traces.
    bin_size : u.Quantity[u.Hz]
        Size of bins for firing rate discretization, specified in Hz units.
    split: float
        The proportion of data used for training.
    """
    flywire_version: str
    neural_activity_id: str
    neural_activity_max_fr: u.Quantity[u.Hz]
    etrace_decay: float
    loss_fn: str
    conn_param_type: type
    input_param_type: type
    scale_factor: u.Quantity[u.mV]
    n_rank: int
    sim_before_train: float
    bin_size: u.Quantity[u.Hz]
    split: float
    fitting_target: str = 'lora'

    def to_filepath(self):
        """
        Convert the configuration parameters to a standardized filepath string.

        Creates a filepath string by concatenating all configuration parameters with '#' delimiters,
        converting quantities to their decimal representations with appropriate units.

        Returns
        -------
        str
            A filepath string containing all parameters in a standardized format:
            'results/flywire_version#neural_activity_id#max_fr#etrace_decay#loss_fn#...'
        """
        return (
            f'results/'
            f'{self.flywire_version}#'
            f'{self.neural_activity_id}#'
            f'{self.neural_activity_max_fr.to_decimal(u.Hz)}#'
            f'{self.etrace_decay}#'
            f'{self.loss_fn}#'
            f'{self.conn_param_type.__name__}#'
            f'{self.input_param_type.__name__}#'
            f'{self.scale_factor.to_decimal(u.mV):5f}#'
            f'{self.n_rank}#'
            f'{self.sim_before_train}#'
            f'{self.bin_size.to_decimal(u.Hz)}#'
            f'{self.split}#'
            f'{self.fitting_target}#'
        )

    @classmethod
    def from_filepath(cls, filepath):
        """
        Parse a filepath string to extract model configuration parameters.

        This method extracts all configuration parameters from a delimited filepath string,
        converts them to appropriate types, and creates a new FilePath instance.

        Parameters
        ----------
        filepath : str
            A filepath string in the format generated by to_filepath().
            Expected format: 'path/to/results/flywire_version#neural_activity_id#max_fr#etrace_decay#...'

        Returns
        -------
        FilePath
            A new FilePath instance with parameters extracted from the filepath.
        """
        setting = filepath.split('/')[1].split('#')
        flywire_version = setting[0]
        neural_activity_id = setting[1]
        max_firing_rate = float(setting[2]) * u.Hz
        etrace_decay = float(setting[3])
        loss_fn = setting[4]
        conn_param_type = setting[5]
        input_param_type = setting[6]
        scale_factor = float(setting[7]) * u.mV
        n_rank = int(setting[8])
        sim_before_train = float(setting[9])
        bin_size = float(setting[10]) * u.Hz
        split = float(setting[11])
        fitting_target = setting[12] if len(setting) > 12 else 'lora'

        return cls(
            flywire_version=flywire_version,
            neural_activity_id=neural_activity_id,
            neural_activity_max_fr=max_firing_rate,
            etrace_decay=etrace_decay,
            loss_fn=loss_fn,
            conn_param_type=getattr(brainscale, conn_param_type),
            input_param_type=getattr(brainscale, input_param_type),
            scale_factor=scale_factor,
            n_rank=n_rank,
            sim_before_train=sim_before_train,
            bin_size=bin_size,
            split=split,
            fitting_target=fitting_target,
        )


class NeuralData:
    """
    A class for handling neural activity data from the Drosophila brain.

    This class manages neural firing rate data, connectivity information,
    and provides methods for data conversion between neuron and neuropil levels.
    It supports data loading, preprocessing, and iteration for training neural networks.

    Attributes:
        bin_size (u.Quantity): Size of bins for firing rate discretization, in Hz.
        neural_activity_id (str): Identifier for the neural activity dataset.
        neuropils (np.ndarray): Array of neuropil names.
        spike_rates (u.Quantity): Array of firing rates with shape [time, neuropil].
        bins (u.Quantity): Bin edges for discretizing firing rates.
        neuropil_to_connectivity (brainstate.util.NestedDict): Dictionary mapping neuropils to their connectivity data.
        split (float): Proportion of data to use for training.
        n_train (int): Number of time points in the training set.
        n_test (int): Number of time points in the test set.
    """

    def __init__(
        self,
        flywire_version: str,
        neural_activity_id: str,
        batch_size: int = 128,
        split: float = 0.7,
        neural_activity_max_fr: u.Quantity = 120 * u.Hz,
        bin_size: u.Quantity = 0.1 * u.Hz,
    ):
        """
        Initialize the NeuralData class.

        Loads neural activity data and connectivity information from the FlyWire
        connectome dataset, and prepares the data for training and testing.

        Args:
            flywire_version (str): Version of the FlyWire connectome dataset ('630' or '783').
            neural_activity_id (str): Identifier for the neural activity dataset.
            split (float, optional): Proportion of data to use for training. Defaults to 0.7.
            neural_activity_max_fr (u.Quantity, optional): Maximum firing rate for normalization. Defaults to 120 Hz.
            bin_size (u.Quantity, optional): Size of bins for firing rate discretization. Defaults to 0.1 Hz.

        Raises:
            ValueError: If flywire_version is not '630' or '783'.
        """
        # uniform binning
        self.bin_size = bin_size
        print('Loading neural activity information ...')

        # neural activity data
        self.neural_activity_id = neural_activity_id
        data = np.load(f'./data/spike_rates/ito_{neural_activity_id}_spike_rate.npz')
        self.neuropils = data['areas'][1:]
        self.spike_rates = u.math.asarray(data['rates'][1:] * neural_activity_max_fr).T  # [time, neuropil]
        self.bins = get_tokenization_bins(self.spike_rates, bin_size, neural_activity_max_fr)
        print(f'Maximum firing rate: {self.spike_rates.max().to_decimal(u.Hz)} Hz')

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

        # training/testing data split
        self.split = split
        self.n_train, self.n_test = split_train_test(self.n_time, split, batch_size=batch_size)

    @property
    def n_neuropil(self) -> int:
        """
        Get the number of neuropils in the dataset.

        Returns:
            int: Number of neuropils.
        """
        return self.spike_rates.shape[1]

    @property
    def n_time(self) -> int:
        """
        Get the number of time points in the dataset.

        Returns:
            int: Number of time points.
        """
        return self.spike_rates.shape[0]

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

        Raises:
            ValueError: If neuropil_fr has more than 2 dimensions.
        """

        def _convert(key, fr):
            # convert firing rates to bins
            right_bin_indices = neuropil_to_bin_indices(fr, self.bins)
            left_bin_indices = right_bin_indices - 1
            left = self.bins[left_bin_indices].to_decimal(u.Hz)
            right = self.bins[right_bin_indices].to_decimal(u.Hz)
            return jax.random.uniform(key, left.shape, minval=left, maxval=right)

        if neuropil_fr.ndim == 1:
            return _convert(brainstate.random.split_key(), neuropil_fr)
        elif neuropil_fr.ndim == 2:
            return jax.vmap(_convert)(brainstate.random.split_key(neuropil_fr.shape[0]), neuropil_fr)
        else:
            raise ValueError

    def neuron_to_neuropil_fr(self, neuron_fr: u.Quantity[u.Hz]):
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
        Read the spike rate of a specific time index.

        Args:
            i (int): The index of the time point to read from the stored spike rates.

        Returns:
            u.Quantity: The spike rates at the specified time index.
        """
        return self.spike_rates[i]

    def count_neuropil_fr(self, spike_count, length: int):
        """
        Convert spike counts to firing rates at the neuropil level.

        This method calculates firing rates from spike counts and maps them to
        neuropil-level firing rates using the connectivity information.

        Args:
            spike_count: Count of spikes for each neuron.
            length (int): Time window length for rate calculation.

        Returns:
            u.Quantity: Firing rates for each neuropil, in Hz units.
        """
        neuron_fr = spike_count / (length * brainstate.environ.get_dt())
        neuron_fr = neuron_fr.to(u.Hz)
        fun = self.neuron_to_neuropil_fr
        for i in range(neuron_fr.ndim - 1):
            fun = jax.vmap(fun)
        neuropil_fr = fun(neuron_fr)
        return neuropil_fr

    @property
    def train_data(self):
        """
        Get the training portion of the spike rate data.

        Returns:
            u.Quantity: Training data with shape [n_train, n_neuropil].
        """
        return self.spike_rates[:self.n_train]

    @property
    def test_data(self):
        """
        Get the testing portion of the spike rate data.

        Returns:
            u.Quantity: Testing data with shape [n_test, n_neuropil].
        """
        return self.spike_rates[self.n_train:]

    def iter_train_data(self, batch_size: int):
        """
        Iterate over the neural activity training data in batches.

        Provides batches of data for training, where each batch contains input firing rates
        and their corresponding target output firing rates for the next time step.

        Args:
            batch_size (int): The size of each batch.

        Yields:
            Tuple[u.Quantity, u.Quantity]: A tuple containing:
                - input_embed: Input firing rates transformed to embeddings
                - output_neuropil_fr: Target output firing rates for the next time step
        """

        # firing rate data
        n = int(self.n_train // batch_size) * batch_size
        spike_rates = np.asarray(self.train_data / u.Hz)[:n]
        spike_rates = spike_rates.reshape(batch_size, -1, spike_rates.shape[1])

        for i in range(1, spike_rates.shape[1]):
            input_neuropil_fr = spike_rates[:, i - 1]
            output_neuropil_fr = spike_rates[:, i]
            input_embed = self.neuropil_fr_to_embedding(input_neuropil_fr)
            yield input_embed, output_neuropil_fr * u.Hz


class Population(brainstate.nn.Neuron):
    """
    A population of neurons with leaky integrate-and-fire dynamics.

    This class implements a population of leaky integrate-and-fire neurons for the Drosophila brain
    simulation, with connectivity based on the FlyWire connectome dataset. Each neuron follows
    standard LIF dynamics with customizable parameters for membrane properties, synaptic
    transmission, and spike generation.

    The dynamics of the neurons are given by the following equations::

       dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
       dg/dt = -g / tau               : volt (unless refractory)

    Parameters
    ----------
    flywire_version : str or int, optional
        Version of the FlyWire connectome dataset to use ('630' or '783'), defaults to '783'
    v_rest : u.Quantity, optional
        The resting potential of the neurons, defaults to 0 mV
    v_reset : u.Quantity, optional
        The reset potential of the neurons after a spike, defaults to 0 mV
    v_th : u.Quantity, optional
        The threshold potential of the neurons for spiking, defaults to 1 mV
    tau_m : u.Quantity, optional
        The membrane time constant of the neurons, defaults to 20 ms
    tau_syn : u.Quantity, optional
        The synaptic time constant of the neurons, defaults to 5 ms
    tau_ref : u.Quantity or None, optional
        The refractory period of the neurons, defaults to 2.2 ms
    spk_fun : Callable, optional
        The spike function of the neurons, defaults to ReluGrad with width=1.5
    V_init : Callable, optional
        Initialization function for membrane voltage, defaults to Constant(0 mV)
    g_init : Callable, optional
        Initialization function for synaptic conductance, defaults to Constant(0 mV)
    name : str, optional
        The name of the population

    Attributes
    ----------
    n_neuron : int
        Number of neurons in the population
    v : brainscale.ETraceState
        Membrane potential state variable
    g : brainscale.ETraceState
        Synaptic conductance state variable
    spike_count : brainscale.ETraceState
        Counter for spikes generated by each neuron
    t_ref : brainstate.HiddenState, optional
        Timestamp of last spike for refractory period calculation
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

        # file paths
        self.path_neu = Path(path_neu)
        self.path_syn = Path(path_syn)

        print('Loading neuron information ...')

        # neuron ids
        flywire_ids = pd.read_csv(self.path_neu, index_col=0)
        self.n_neuron = len(flywire_ids)

        super().__init__(self.n_neuron, name=name)

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


class Interaction(brainstate.nn.Module):
    """
    A module that handles synaptic interactions between neurons in a spiking network.

    This class implements the synaptic connectivity and signal propagation between neurons
    in a population, supporting both sparse connectivity from connectome data and
    trainable low-rank approximation (LoRA) parameters for efficient learning.

    The module manages synaptic delays and implements different connection modes,
    particularly focusing on a combined sparse and low-rank connectivity approach
    for computational efficiency while maintaining biological plausibility.

    Parameters
    ----------
    pop : Population
        The neural population containing the neurons that will interact.
    scale_factor : u.Quantity
        Scaling factor for the synaptic weights, specified with units.
    conn_param_type : type, optional
        Parameter type for connection weights (typically a brainscale parameter class),
        defaults to brainscale.ETraceParam.
    n_rank : int, optional
        Rank for low-rank approximation in LoRA, defaults to 20.

    Attributes
    ----------
    pop : Population
        The neural population being modeled.
    delay : brainstate.nn.Delay
        Module handling synaptic transmission delays.
    conn : brainstate.nn.SparseLinear
        Sparse connectivity matrix representing the connectome.
    lora : brainscale.nn.LoRA
        Low-rank approximation module for trainable connectivity.
    scale_factor : u.Quantity
        Scaling factor applied to synaptic weights.
    """

    def __init__(
        self,
        pop: Population,
        scale_factor: u.Quantity,
        conn_param_type: type = brainscale.ETraceParam,
        n_rank: int = 20,
        fitting_target: str = 'lora'
    ):
        super().__init__()

        self.fitting_target = fitting_target

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
        self.scale_factor = scale_factor

        if fitting_target == 'lora':
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

        elif fitting_target == 'csr':
            # do not train sparse connection
            self.conn = brainscale.nn.SparseLinear(csr, b_init=None)

        else:
            raise ValueError('fitting_target must be either "lora" or "csr"')

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
        inp = self.conn(brainevent.EventArray(pre_spk)) * self.scale_factor
        if self.fitting_target == 'lora':
            inp = inp + self.lora(pre_spk)

        if x is None:
            x = inp
        else:
            x += inp
        spk = self.pop(x)

        # update delay spikes
        self.delay.update(jax.lax.stop_gradient(spk))

        return spk


class BackgroundInputEncoder(brainstate.nn.Module):
    """
    A module for encoding background inputs to a neural population.

    This class transforms embedding inputs into appropriate noise weights
    for background stimulation of neurons in a population. It provides a mechanism
    for introducing controlled background activity in a spiking neural network.

    The encoder uses a neural network to transform input embeddings into
    weights that are then used to modulate Poisson noise inputs to the neurons.
    This allows for realistic background activity simulation in the neural population.

    Parameters
    ----------
    n_in : int
        Number of input features in the embedding vector.
    pop : Population
        The neural population to which background input will be applied.
    param_type : type, optional
        Parameter type class for the encoder weights (typically a brainscale
        parameter class that supports eligibility traces), defaults to brainscale.ETraceParam.

    Attributes
    ----------
    pop : Population
        The neural population being stimulated with background activity.
    encoder : brainstate.nn.Sequential
        Neural network that transforms embeddings into noise weights.
    """

    def __init__(
        self,
        n_in: int,
        pop: Population,
        param_type: type = brainscale.ETraceParam,
    ):
        super().__init__()

        # population
        self.pop = pop

        # neural activity conversion
        self.encoder = brainstate.nn.Sequential(
            brainscale.nn.Linear(
                n_in,
                self.pop.varshape,
                w_init=brainstate.init.KaimingNormal(unit=u.mV),
                b_init=brainstate.init.ZeroInit(unit=u.mV),
                param_type=param_type
            ),
            brainscale.nn.ReLU()
        )

    def update(self, embedding):
        noise_weight = self.encoder(u.get_mantissa(embedding))

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

    def update_test(self, noise_weight):
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


class DrosophilaSpikingNetwork(brainstate.nn.Module):
    """
    A spiking neural network model for simulating firing rate patterns in the Drosophila brain.

    This class implements a biologically plausible spiking neural network that models
    neuronal activity in the Drosophila brain based on FlyWire connectome data.
    It integrates neural population dynamics, synaptic interactions, and mechanisms
    to process neural activity data for realistic brain simulation.

    Parameters
    ----------
    flywire_version : str, optional
        Version of the FlyWire connectome dataset ('630' or '783'), defaults to '630'
    n_rank : int, optional
        Rank for low-rank approximation in LoRA, defaults to 20
    scale_factor : u.Quantity, optional
        Scaling factor for sparse connectivity weights, defaults to 0.3*0.275/7 mV
    conn_param_type : type, optional
        Parameter type for connection weights, defaults to brainscale.ETraceParam
    input_param_type : type, optional
        Parameter type for input weights, defaults to brainscale.ETraceParam
    sampling_rate : u.Quantity, optional
        Rate at which to sample neural activity, defaults to 1.2 Hz

    Attributes
    ----------
    pop : Population
        Neural population containing the simulated neurons
    input : BackgroundInputEncoder
        Module handling neural activity data and conversions
    interaction : Interaction
        Module implementing synaptic interactions between neurons
    n_sample_step : int
        Number of simulation steps per sample period
    flywire_version : str
        Version of the FlyWire connectome being used
    """

    def __init__(
        self,
        n_neuropil: int,
        flywire_version: str = '630',
        n_rank: int = 20,
        scale_factor=0.3 * 0.275 / 7 * u.mV,
        conn_param_type: type = brainscale.ETraceParam,
        input_param_type: type = brainscale.ETraceParam,
        sampling_rate: u.Quantity = 1.2 * u.Hz,
        fitting_target: str = 'lora'
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
        self.input = BackgroundInputEncoder(
            n_in=n_neuropil,
            pop=self.pop,
            param_type=input_param_type,
        )

        # network
        self.interaction = Interaction(
            self.pop,
            n_rank=n_rank,
            scale_factor=scale_factor,
            conn_param_type=conn_param_type,
            fitting_target=fitting_target,
        )

    def update(self, i, embedding: u.Quantity):
        with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
            # give inputs
            self.input.update(embedding)

            # update network
            spk = self.interaction.update()
            return spk

    def update_test(self, i, noise_weight):
        with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
            # give inputs
            self.input.update_test(noise_weight)

            # update network
            spk = self.interaction.update()
            return spk

    @brainstate.compile.jit(static_argnums=0, static_argnames='return_spikes')
    def simulate(self, inp_embedding, indices, return_spikes=False):
        def step_run(i):
            spk = self.update_test(i, noise_weight)
            return spk if return_spikes else None

        noise_weight = self.input.encoder(inp_embedding)
        self.pop.reset_spk_count()
        return brainstate.compile.for_loop(step_run, indices)


class DrosophilaSpikingNetTrainer:
    """
    A class for training spiking neural network models to simulate firing rate patterns
    in Drosophila whole brain data.

    This trainer implements a two-stage training approach:
    1. First round: Trains a spiking neural network to match target neuropil firing rates
    2. Second round (separate method): Trains a recurrent neural network to decode firing patterns

    The class handles training of network parameters using gradient-based optimization with
    eligibility traces for biologically plausible learning.

    Parameters
    ----------
    sim_before_train : float, optional
        Proportion of steps to simulate before recording eligibility traces, defaults to 0.5
    lr : float, optional
        Learning rate for the optimizer, defaults to 1e-3
    etrace_decay : float or None, optional
        Decay factor for eligibility traces (None or 0 for standard backprop), defaults to 0.99
    grad_clip : float or None, optional
        Maximum norm for gradient clipping, defaults to 1.0
    neural_activity_id : str, optional
        Identifier for neural activity dataset, defaults to '2017-10-26_1'
    flywire_version : str, optional
        Version of the FlyWire connectome dataset ('630' or '783'), defaults to '630'
    max_firing_rate : u.Quantity, optional
        Maximum firing rate for normalization, defaults to 100 Hz
    loss_fn : str, optional
        Loss function ('mse', 'mae', 'huber', 'cosine_distance', 'log_cosh'), defaults to 'mse'
    vjp_method : str, optional
        Method for vector-Jacobian product calculation, defaults to 'single-step'
    n_rank : int, optional
        Rank for low-rank approximation in LoRA, defaults to 20
    conn_param_type : type, optional
        Parameter type for connection weights, defaults to brainscale.ETraceParam
    input_param_type : type, optional
        Parameter type for input weights, defaults to brainscale.ETraceParam
    scale_factor : u.Quantity, optional
        Scaling factor for sparse connectivity weights, defaults to 0.01 mV
    bin_size : u.Quantity, optional
        Size of bins for firing rate discretization, defaults to 0.1 Hz

    Methods
    -------
    get_loss(current_neuropil_fr, target_neuropil_fr)
        Computes loss between current and target neuropil firing rates
    train(input_embed, target_neuropil_fr)
        Performs one training step with given input and target
    show_res(neuropil_fr, target_neuropil_fr, i_epoch, i_batch, n_neuropil_per_fig=10)
        Visualizes results comparing simulated vs. target firing rates
    round1_train(train_epoch, batch_size=128, checkpoint_path=None)
        Executes the first round of training
    """

    def __init__(
        self,
        sim_before_train: float = 0.5,
        lr: float = 1e-3,
        etrace_decay: float | None = 0.99,
        grad_clip: float | None = 1.0,
        neural_activity_id: str = '2017-10-26_1',
        flywire_version: str = '630',
        max_firing_rate: u.Quantity = 100. * u.Hz,
        loss_fn: str = 'mse',
        vjp_method: str = 'single-step',
        n_rank: int = 20,
        conn_param_type: type = brainscale.ETraceParam,
        input_param_type: type = brainscale.ETraceParam,
        scale_factor=0.01 * u.mV,
        bin_size: u.Quantity = 0.1 * u.Hz,
        split: float = 0.7,
        fitting_target: str = 'lora',
    ):
        # parameters
        self.sim_before_train = sim_before_train
        self.etrace_decay = etrace_decay
        self.grad_clip = grad_clip
        self.loss_fn = loss_fn
        self.vjp_method = vjp_method

        # input
        self.data = NeuralData(
            flywire_version=flywire_version,
            neural_activity_id=neural_activity_id,
            split=split,
            neural_activity_max_fr=max_firing_rate,
            bin_size=bin_size,
        )

        # population
        self.target = DrosophilaSpikingNetwork(
            n_neuropil=self.data.n_neuropil,
            flywire_version=flywire_version,
            conn_param_type=conn_param_type,
            input_param_type=input_param_type,
            scale_factor=scale_factor,
            n_rank=n_rank,
            fitting_target=fitting_target,
        )

        # optimizer
        self.trainable_weights = brainstate.graph.states(self.target, brainstate.ParamState)
        self.opt = brainstate.optim.Adam(brainstate.optim.StepLR(lr, step_size=10, gamma=0.9))
        self.opt.register_trainable_weights(self.trainable_weights)

        # train save path
        args = FilePath(
            flywire_version=flywire_version,
            neural_activity_id=neural_activity_id,
            neural_activity_max_fr=max_firing_rate,
            conn_param_type=conn_param_type,
            input_param_type=input_param_type,
            etrace_decay=etrace_decay,
            loss_fn=loss_fn,
            scale_factor=scale_factor,
            n_rank=n_rank,
            sim_before_train=sim_before_train,
            bin_size=bin_size,
            split=split,
            fitting_target=fitting_target,
        )
        self.filepath = f"{args.to_filepath()}#{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

    def _get_loss(self, current_neuropil_fr, target_neuropil_fr):
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
    def _train(self, input_embed, target_neuropil_fr):
        indices = np.arange(self.target.n_sample_step)

        # last_neuropil_fr: [n_batch, n_neuropil]
        # target_neuropil_fr: [n_batch, n_neuropil]
        n_batch = input_embed.shape[0]

        # model
        if self.etrace_decay is None or self.etrace_decay == 0.:
            model = brainscale.ParamDimVjpAlgorithm(self.target, vjp_method=self.vjp_method)
        else:
            model = brainscale.IODimVjpAlgorithm(self.target, self.etrace_decay, vjp_method=self.vjp_method)

        # brainstate.nn.vmap_init_all_states(self.target, axis_size=n_batch, state_tag='hidden')

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
            brainstate.compile.for_loop(lambda i: batch_target(i, input_embed), indices[:n_sim])

        # simulation with eligibility trace recording
        self.target.pop.reset_spk_count(n_batch)
        model = brainstate.nn.Vmap(model, vmap_states=('hidden', 'etrace'), in_axes=(None, 0))
        brainstate.compile.for_loop(lambda i: model(i, input_embed), indices[n_sim:-1])

        # training
        def loss_fn(i):
            spk = model(i, input_embed)
            current_neuropil_fr = self.data.count_neuropil_fr(
                self.target.pop.spike_count.value,
                self.target.n_sample_step - n_sim
            )
            loss_ = self._get_loss(current_neuropil_fr, target_neuropil_fr).mean()
            return u.get_mantissa(loss_), current_neuropil_fr

        # gradients and optimizations
        grads, loss, neuropil_fr = brainstate.augment.grad(
            loss_fn, self.trainable_weights, return_value=True, has_aux=True
        )(indices[-1])
        max_g = jax.tree.map(lambda x: jnp.abs(x).max(), grads)
        if self.grad_clip is not None:
            grads = brainstate.functional.clip_grad_norm(grads, self.grad_clip)
        self.opt.update(grads)

        # tokenization bin accuracy
        target_bin_indices = neuropil_to_bin_indices(target_neuropil_fr, self.data.bins)
        predict_bin_indices = neuropil_to_bin_indices(neuropil_fr, self.data.bins)
        acc = jnp.mean(jnp.asarray(target_bin_indices == predict_bin_indices, dtype=jnp.float32))

        return loss, neuropil_fr, max_g, acc

    def _show_res(
        self,
        neuropil_fr: u.Quantity[u.Hz],
        target_neuropil_fr: u.Quantity[u.Hz],
        i_epoch: int,
        i_batch: int,
        n_neuropil_per_fig: int = 10,
        filepath: str = None,
    ):
        filepath = filepath or self.filepath
        fig, gs = braintools.visualize.get_figure(n_neuropil_per_fig, 2, 2., 10)
        for i in range(n_neuropil_per_fig):
            xticks = (i + 1 == n_neuropil_per_fig)
            fig.add_subplot(gs[i, 0])
            barplot(
                self.data.neuropils,
                neuropil_fr[i].to_decimal(u.Hz),
                title='Simulated FR',
                xticks=xticks
            )
            fig.add_subplot(gs[i, 1])
            barplot(
                self.data.neuropils,
                target_neuropil_fr[i].to_decimal(u.Hz),
                title='True FR',
                xticks=xticks
            )
        filename = f'{filepath}/images/neuropil_fr-at-epoch-{i_epoch}-batch-{i_batch}.png'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
        plt.close()


class DrosophilaInputEncoder(brainstate.nn.Module):
    """
    Neural network encoder for processing and transforming Drosophila neural activity data.

    This module implements a recurrent neural network architecture for encoding
    time-series neural activity data in the Drosophila brain. It processes input firing
    rates through normalization, a GRU cell for temporal context, and a linear readout
    layer to produce output firing rates.

    The encoder is designed to be used in the second round of training after the spiking
    neural network has been trained, serving as a decoder/predictor of neural activity
    patterns.

    Parameters
    ----------
    n_in : int
        Number of input features (typically number of neuropils in the dataset)
    n_hidden : int
        Size of the hidden state in the GRU cell
    n_out : int
        Number of output features (typically matches n_in for autoencoding)

    Attributes
    ----------
    norm : brainstate.nn.LayerNorm
        Layer normalization module for input standardization
    rnn : brainstate.nn.GRUCell
        Gated Recurrent Unit cell for processing sequential data
    readout : brainstate.nn.Linear
        Linear transformation from hidden state to output firing rates
    """

    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        self.norm = brainstate.nn.LayerNorm(n_in, use_scale=False, use_bias=False)
        self.rnn = brainstate.nn.GRUCell(n_in, n_hidden)
        self.readout = brainstate.nn.Linear(n_hidden, n_out)

    def update(self, x):
        norm = self.norm(u.get_mantissa(x))
        rnn = self.rnn(norm)
        readout = self.readout(rnn)
        return brainstate.functional.relu(readout) * u.Hz


class DrosophilaInputTrainer:
    """
    A trainer for the Drosophila input encoder/decoder network.

    This class implements the second round of training, focusing on a recurrent neural
    network that can efficiently encode and decode neural activity patterns in the
    Drosophila brain. It uses data generated from the first-round spiking neural
    network to train a GRU-based encoder that can reproduce similar firing patterns.

    The trainer loads experimental and simulated neural activity data, configures
    tokenization for discrete firing rate representation, and handles the training
    process with appropriate optimization strategies and loss functions.

    Parameters
    ----------
    filepath : str
        Path to the directory containing the first-round training results and generated
        neural activity data.
    noise_scale : float, optional
        Scale of noise to add to inputs during training for improved robustness,
        defaults to 0.1.

    Attributes
    ----------
    filepath : str
        Path to the training data directory.
    bins : u.Quantity
        Bin edges for discretizing firing rates.
    true_bin_indices : ndarray
        Indices of bins containing the true firing rates from experimental data.
    simulated_spike_rates : ndarray
        Firing rates generated by the first-round spiking neural network.
    noise_scale : float
        Scale of noise added to inputs during training.
    low_rates : ndarray
        Lower bounds of firing rate bins for each data point.
    high_rates : ndarray
        Upper bounds of firing rate bins for each data point.
    """

    def __init__(
        self,
        filepath: str,
        batch_size: int,
        n_rnn_hidden: int,
        lr_round2: float,
        noise_scale: float = 0.1
    ):
        parser = FilePath.from_filepath(filepath)
        self.filepath = filepath
        self.batch_size = batch_size
        self.lr_round2 = lr_round2

        # load data
        data = np.load(f'data/spike_rates/ito_{parser.neural_activity_id}_spike_rate.npz')
        spike_rates = u.math.asarray(data['rates'][1:] * parser.neural_activity_max_fr).T
        n_train, _ = split_train_test(spike_rates.shape[0], parser.split, batch_size=batch_size)
        spike_rates = spike_rates[:n_train]

        # tokenization
        self.bins = get_tokenization_bins(spike_rates, parser.bin_size, parser.neural_activity_max_fr)
        self.true_bin_indices = neuropil_to_bin_indices(spike_rates[1:], self.bins)

        # load simulated data
        self.simulated_spike_rates = np.load(os.path.join(filepath, 'simulated_neuropil_fr.npy'))[:n_train]
        self.noise_scale = noise_scale

        # experimental data tokenization
        bin_indices = neuropil_to_bin_indices(spike_rates, self.bins)
        self.low_rates = self.bins[bin_indices - 1].to_decimal(u.Hz)
        self.high_rates = self.bins[bin_indices].to_decimal(u.Hz)

        # network and its optimizer
        lr = brainstate.optim.StepLR(self.lr_round2, step_size=10, gamma=0.9)
        self.net = DrosophilaInputEncoder(
            n_in=spike_rates.shape[1],
            n_hidden=n_rnn_hidden,
            n_out=spike_rates.shape[1]
        )
        weights = self.net.states(brainstate.ParamState)
        self.opt = brainstate.optim.Adam(lr=lr)
        self.opt.register_trainable_weights(weights)

    @brainstate.compile.jit(static_argnums=0)
    def generate_inputs(self):
        # sampling from low and high rates, from experimental data
        true_sample_fn = jax.vmap(lambda key: brainstate.random.uniform(self.low_rates, self.high_rates, key=key))
        true_sampling = true_sample_fn(brainstate.random.split_key(self.batch_size // 2))

        # sampling from simulated data, with noise
        simulation_sample_fn = jax.vmap(
            lambda key: jax.random.normal(key, self.simulated_spike_rates.shape) * self.noise_scale +
                        self.simulated_spike_rates
        )
        simulation_sampling = simulation_sample_fn(brainstate.random.split_key(self.batch_size // 2))
        simulation_sampling = jnp.minimum(simulation_sampling, 0.)

        # concatenate experimental and simulated data
        inputs = jnp.concatenate([true_sampling, simulation_sampling], axis=0)
        return jnp.transpose(inputs, (1, 0, 2))

    def f_predict(self, inputs):
        brainstate.nn.vmap_init_all_states(self.net, axis_size=self.batch_size)
        return brainstate.compile.for_loop(self.net, inputs)

    def verify_acc(self, predictions):
        pred_bin_indices = neuropil_to_bin_indices(predictions, self.bins)
        acc = jnp.asarray(pred_bin_indices == self.true_bin_indices, dtype=float).mean()
        return acc

    def loss_fn(self, predictions):
        mse = (
            u.math.square(u.math.relu(self.low_rates[1:] - predictions / u.Hz)) +
            u.math.square(u.math.relu(predictions / u.Hz - self.high_rates[1:]))
        ).mean()
        return mse

    def f_loss(self, inputs):
        predictions = self.f_predict(inputs)
        predictions = u.math.transpose(predictions[:-1], (1, 0, 2))
        mse = self.loss_fn(predictions)
        acc = jax.vmap(self.verify_acc)(predictions).mean()
        return mse, acc

    @brainstate.compile.jit(static_argnums=0)
    def _epoch_train(self, inputs):
        grads, l, acc = brainstate.augment.grad(self.f_loss,
                                                self.net.states(brainstate.ParamState),
                                                return_value=True,
                                                has_aux=True)(inputs)
        self.opt.update(grads)
        return l, acc

    @brainstate.compile.jit(static_argnums=0)
    def _epoch_test(self):
        brainstate.nn.init_all_states(self.net)
        outputs = brainstate.compile.for_loop(self.net, self.simulated_spike_rates)
        mse = self.loss_fn(outputs[:-1])
        acc = jax.vmap(self.verify_acc)(outputs[:-1]).mean()
        return mse, acc


class DrosophilaRestingStateModel:
    """
    A model for simulating resting state neural activity in the Drosophila brain.

    This class integrates the first and second round models to generate realistic resting state
    neural activity patterns in the Drosophila brain. It loads pre-trained weights from both
    the spiking neural network trained in the first round and the GRU-based encoder trained
    in the second round to create a complete system for neural activity simulation.

    The model can be used to generate continuous sequences of neural activity that match the
    statistical properties of observed resting state data, making it useful for studying
    spontaneous neural dynamics in the absence of external stimuli.

    Parameters
    ----------
    filepath : str
        Path to the directory containing the trained model checkpoints from both rounds
        of training. The directory structure should match what is produced by the
        first_round_train and second_round_train functions.

    Attributes
    ----------
    filepath : str
        Path to the directory containing model checkpoints.
    args : FilePath
        Named tuple containing parsed configuration arguments from the filepath.
    data : NeuralData
        Object managing neural activity data and conversions.
    spiking_net : DrosophilaSpikingNetwork
        The first-round spiking neural network model with trained weights.
    """

    def __init__(
        self,
        filepath: str,
        n_rnn_hidden: int = 256,
        load_checkpoint: bool = True,
    ):
        self.load_checkpoint = load_checkpoint
        self.filepath = filepath
        self.args = FilePath.from_filepath(filepath)

        self.data = NeuralData(
            flywire_version=self.args.flywire_version,
            neural_activity_id=self.args.neural_activity_id,
            neural_activity_max_fr=self.args.neural_activity_max_fr,
            bin_size=self.args.bin_size,
            split=self.args.split,
        )

        # spiking neural network for spike generation
        self.spiking_net = DrosophilaSpikingNetwork(
            n_neuropil=self.data.n_neuropil,
            flywire_version=self.args.flywire_version,
            conn_param_type=self.args.conn_param_type,
            input_param_type=self.args.input_param_type,
            n_rank=self.args.n_rank,
            scale_factor=self.args.scale_factor,
            fitting_target=self.args.fitting_target,
        )
        if load_checkpoint:
            braintools.file.msgpack_load(
                os.path.join(filepath, 'first-round-checkpoint.msgpack'),
                self.spiking_net.states(brainstate.ParamState)
            )
        brainstate.nn.init_all_states(self.spiking_net)

        if load_checkpoint:
            # Recurrent neural network for decoding
            self.rnn_net = DrosophilaInputEncoder(
                n_in=self.data.n_neuropil,
                n_hidden=n_rnn_hidden,
                n_out=self.data.n_neuropil
            )
            braintools.file.msgpack_load(
                os.path.join(filepath, f'second-round-checkpoint.msgpack'),
                self.rnn_net.states(brainstate.LongTermState)
            )
            brainstate.nn.init_all_states(self.rnn_net)

    @brainstate.compile.jit(static_argnums=0, static_argnames='return_spikes')
    def _predict(self, neuropil_firing_rate, target_firing_rate, running_indices, return_spikes=False):
        n_sim = int(self.args.sim_before_train * self.spiking_net.n_sample_step)

        # input
        if self.load_checkpoint:
            rnn_out = self.rnn_net(neuropil_firing_rate.to_decimal(u.Hz))
        else:
            rnn_out = neuropil_firing_rate

        # spiking simulation
        spks1 = self.spiking_net.simulate(rnn_out / u.Hz, running_indices[:n_sim], return_spikes=return_spikes)
        spks2 = self.spiking_net.simulate(rnn_out / u.Hz, running_indices[n_sim:], return_spikes=return_spikes)
        if return_spikes:
            spks = jnp.concatenate([spks1, spks2], axis=0)

        # firing rate
        neuropil_fr = self.data.count_neuropil_fr(self.spiking_net.pop.spike_count.value,
                                                  self.spiking_net.n_sample_step - n_sim)

        # bin accuracy and MSE
        target_bin_indices = neuropil_to_bin_indices(target_firing_rate, self.data.bins)
        predict_bin_indices = neuropil_to_bin_indices(neuropil_fr, self.data.bins)
        acc = jnp.mean(jnp.asarray(target_bin_indices == predict_bin_indices, dtype=jnp.float32))
        mse = u.get_mantissa(u.math.square(target_firing_rate - neuropil_fr)).mean()
        if return_spikes:
            return spks, neuropil_fr, mse, acc
        else:
            return neuropil_fr, mse, acc

    def f_test(self, n_time, filename: str = 'neuropil_fr_test'):
        indices = np.arange(self.spiking_net.n_sample_step)
        simulated_spike_indices = []
        simulated_spike_times = []
        simulated_neuropil_fr = []
        for i in range(n_time):
            if i < 100:
                neuropil_fr = self.data.read_neuropil_fr(i)
            target_neuropil_fr = self.data.read_neuropil_fr(i + 1)
            spikes, neuropil_fr, mse, acc = self._predict(neuropil_fr, target_neuropil_fr, indices, return_spikes=True)
            spk_times, spk_indices = np.where(spikes)
            spk_times += i * self.spiking_net.n_sample_step
            simulated_neuropil_fr.append(np.asarray(neuropil_fr.to_decimal(u.Hz)))
            simulated_spike_times.append(spk_times)
            simulated_spike_indices.append(spk_indices)
            indices += self.spiking_net.n_sample_step
        simulated_spike_times = np.concatenate(simulated_spike_times, axis=0)
        simulated_spike_indices = np.concatenate(simulated_spike_indices, axis=0)
        simulated_neuropil_fr = np.asarray(simulated_neuropil_fr)  # [n_time, n_neuropil]

        np.savez(
            os.path.join(self.filepath, filename),
            spike_times=simulated_spike_times,
            spike_indices=simulated_spike_indices,
            simulated_neuropil_fr=simulated_neuropil_fr
        )

    def f_predict(self, filename: str = 'neuropil_fr_predictions'):
        t1 = brainstate.environ.get_dt() * self.spiking_net.n_sample_step
        indices = np.arange(self.spiking_net.n_sample_step)
        bar = tqdm(total=self.data.n_time - 1)
        all_accs = []
        all_losses = []
        simulated_neuropil_fr = []
        for i in range(self.data.n_time - 1):
            if i < 100:
                neuropil_fr = self.data.read_neuropil_fr(i)
            target_neuropil_fr = self.data.read_neuropil_fr(i + 1)
            neuropil_fr, mse, acc = self._predict(neuropil_fr, target_neuropil_fr, indices)
            simulated_neuropil_fr.append(neuropil_fr.to_decimal(u.Hz))
            bar.set_description(f'Bin acc = {acc}, mse = {mse}')
            bar.update(1)
            all_losses.append(float(mse))
            all_accs.append(float(acc))
            indices += self.spiking_net.n_sample_step
        simulated_neuropil_fr = np.asarray(simulated_neuropil_fr)  # [n_time, n_neuropil]
        print(
            f'Mean bin acc  = {np.mean(all_accs):.5f}, '
            f'mean mse loss = {np.mean(all_losses):.5f}'
        )
        np.save(os.path.join(self.filepath, filename), simulated_neuropil_fr)
        experimental_neuropil_fr = np.asarray(self.data.spike_rates[1:] / u.Hz)  # [n_time, n_neuropil]

        num = 5
        filepath = os.path.join(self.filepath, 'figures')
        os.makedirs(filepath, exist_ok=True)
        times = np.arange(simulated_neuropil_fr.shape[0]) * t1
        for ii in range(0, self.data.n_neuropil, num):
            fig, gs = braintools.visualize.get_figure(num, 2, 4, 8.0)
            for i in range(num):
                # plot simulated neuropil firing rate data
                fig.add_subplot(gs[i, 0])
                data = simulated_neuropil_fr[:, i + ii]
                plt.plot(times, data)
                plt.ylim(0., data.max() * 1.05)
                # plot experimental neuropil firing rate data
                fig.add_subplot(gs[i, 1])
                data = experimental_neuropil_fr[:, i + ii]
                plt.plot(times, data)
                plt.ylim(0., data.max() * 1.05)
            plt.savefig(f'{filepath}/{filename}_{ii}.pdf')
        plt.close()

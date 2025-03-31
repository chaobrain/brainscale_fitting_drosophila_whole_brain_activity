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

import subprocess

import brainstate
import brainunit as u
import numba
import numpy as np
from scipy import optimize, signal
import matplotlib.pyplot as plt


def v_init(shape):
    return brainstate.random.RandomState(42).normal(0., 0.8, shape) * u.mV


def g_init(shape):
    return brainstate.random.RandomState(2025).uniform(0., 0.2, shape) * u.mV


def get_gpu_info() -> str:
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
    Low pass filter region response trace.

    :region_response: np array
    :cutoff: Hz
    :fs: Hz
    """
    if fs is not None:
        sos = signal.butter(1, cutoff, 'hp', fs=fs, output='sos')
        region_response_filtered = signal.sosfilt(sos, region_response)
    else:
        region_response_filtered = region_response

    return region_response_filtered


def trim_region_response(file_id, region_response, start_include=100, end_include=None):
    """
    Trim artifacts from selected brain data.

    Dropouts, weird baseline shifts etc.

    :file_id: string
    :region_response: np array
        either:
            nrois x frames (region responses)
            1 x frames (binary behavior response)
    :start_include: beginning timepoint of trace
    :end_include: end timepoint of trace
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




def barplot(neuropil_names, neuropil_fr, title='', xticks=True):
    x_pos = np.arange(len(neuropil_names))
    plt.bar(x_pos, neuropil_fr)
    if xticks:
        plt.xticks(x_pos, neuropil_names, rotation=90, fontsize=8)
        plt.xlabel('Neuropil')
    plt.ylabel('Firing Rate (Hz)')
    if title:
        plt.title(title)


def output(file, msg: str):
    print(msg)
    file.write(msg + '\n')
    file.flush()




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

import os

import braintools
import brainunit as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed

from models import (
    deconvolve_dff_to_spikes,
    filter_region_response,
    trim_region_response,
)


def obtain_atalas_syn_count(flywire_version='783'):
    if flywire_version == '783':
        neurons = pd.read_csv("data/Completeness_783.csv")
        id_to_index = {row: i for i, row in enumerate(neurons['id'].values)}

        for filename in [
            "data/783_connections_no_threshold.csv",
            "data/783_connections.csv"
        ]:
            print('Processing file:', filename)
            connectivity = pd.read_csv(filename)
            indices = np.asarray([id_to_index[id_] for id_ in connectivity['pre_root_id'].values])
            connectivity['pre_index'] = indices
            indices = np.asarray([id_to_index[id_] for id_ in connectivity['post_root_id'].values])
            connectivity['post_index'] = indices
            connectivity.to_csv(os.path.splitext(filename)[0] + '_processed.csv', index=False)

    elif flywire_version == '630':
        neurons = pd.read_csv("data/Completeness_630_final.csv")
        id_to_index = {row: i for i, row in enumerate(neurons['id'].values)}

        for filename in [
            "data/630_connections.csv"
        ]:
            print('Processing file:', filename)
            connectivity = pd.read_csv(filename)
            indices = np.asarray([id_to_index[id_] for id_ in connectivity['pre_root_id'].values])
            connectivity['pre_index'] = indices
            indices = np.asarray([id_to_index[id_] for id_ in connectivity['post_root_id'].values])
            connectivity['post_index'] = indices
            connectivity.to_csv(os.path.splitext(filename)[0] + '_processed.csv', index=False)

    else:
        raise ValueError("Invalid flywire version")

    # for name in atalas['name'].values:
    #     position = connectivity['neuropil'] == name
    #     pre_indices = connectivity['pre_index'][position].values
    #     syn_count = connectivity['syn_count'][position].values
    #
    #     print(pre_indices)
    #     print(syn_count)
    #
    #     # data = data[['pre_root_id', 'syn_count', 'neuropil']]
    #     # print(data)
    #     break
    #
    # # print(atalas)
    # print(connectivity)
    # return atalas


def compute_experimental_fc(filepath, fs=1.2 * u.Hz, cutoff=0.01):
    neural_activity = np.load(filepath)
    areas = neural_activity['areas'][1:]
    traces = neural_activity['traces'][1:]

    # convert to dF/F
    assert traces.ndim == 2
    mean_response = np.mean(traces, axis=1)[:, None]
    dff = (traces - mean_response) / mean_response

    # trim and filter
    resp = filter_region_response(dff, cutoff=cutoff, fs=fs.to_decimal(u.Hz))
    resp = trim_region_response('', resp)

    num = 10
    fig, gs = braintools.visualize.get_figure(num, 2, 2, 8)
    for i in range(num):
        fig.add_subplot(gs[i, 0])
        plt.plot(resp[i])
        plt.xlim(0, resp.shape[1])

        fig.add_subplot(gs[i, 1])
        plt.plot(deconvolve_dff_to_spikes(resp[i], sampling_rate=fs))
        plt.xlim(0, resp.shape[1])
    plt.suptitle(filepath)
    plt.show()


def convert_calcium_to_spike_rate(fs=1.2 * u.Hz, cutoff=0.01):
    def _load_signal(file_id):
        neural_activity = np.load(f'./data/neural_activity/ito_{file_id}.npz')
        areas = neural_activity['areas'][1:]
        traces = neural_activity['traces'][1:]

        # convert to dF/F
        assert traces.ndim == 2
        mean_response = np.mean(traces, axis=1)[:, None]
        dff = (traces - mean_response) / mean_response

        # trim and filter
        resp = filter_region_response(dff, cutoff=cutoff, fs=fs.to_decimal(u.Hz))
        resp = trim_region_response(file_id, resp)

        return resp, traces, areas

    def _convert_to_spike_rate(resp):
        parallel = Parallel(n_jobs=10)
        rates = parallel(
            delayed(lambda x: np.asarray(deconvolve_dff_to_spikes(x, sampling_rate=fs) / u.Hz))(x)
            for x in resp
        )
        return np.asarray(rates)

    def _convert(file_id):
        print('processing file:', file_id)
        resp, traces, areas = _load_signal(file_id)
        rates = _convert_to_spike_rate(resp)
        np.savez(
            f'./data/spike_rates/ito_{file_id}_spike_rate.npz',
            dff=resp,
            traces=traces,
            rates=rates,
            areas=areas
        )

    all_file_ids = [
        # '2017-10-26_1',
        # '2017-10-30_1',
        # '2017-10-30_2',
        '2017-11-08_1',
        '2017-11-08_2',
        # '2017-11-16_1',
        # '2018-10-19_1',
        '2018-10-19_2',
        # '2018-10-20_1',
        '2018-10-31_1',
        '2018-11-03_2',
        '2018-11-03_3',
        '2018-11-03_4',
        '2018-11-03_5',
        '2018-12-12_2',
        '2018-12-12_3',
        '2018-12-12_4',
        '2018-12-14_1',
        '2018-12-14_2',
        '2018-12-14_3',
    ]
    for f in all_file_ids:
        _convert(f)


def visualize_spike_rate():
    data = np.load('./data/spike_rates/ito_2017-10-26_1_spike_rate.npz')

    num = 6
    for i_start in range(0, data['rates'].shape[0], num):
        fig, gs = braintools.visualize.get_figure(num, 2, 2, 8)
        for i in range(num):
            fig.add_subplot(gs[i, 0])
            plt.plot(data['dff'][i + i_start])
            plt.xlim(0, data['dff'].shape[1])
            plt.ylabel(data['areas'][i + i_start])

            fig.add_subplot(gs[i, 1])
            plt.plot(data['rates'][i + i_start])
            plt.xlim(0, data['rates'].shape[1])
        plt.show()


def visualize_dff_fr():
    data = np.load('./data/spike_rates/ito_2017-10-26_1_spike_rate.npz')
    dff = data['dff']
    rates = data['rates']

    num = 40
    fig, gs = braintools.visualize.get_figure(1, 2, 8, 5)
    fig.add_subplot(gs[0, 0])
    xpos = np.arange(num) / 3
    times = np.arange(dff.shape[1]) * 1 / 1.2

    for i in range(num):
        plt.plot(times, dff[i] + xpos[i])
    plt.xlim(0, 1600)
    plt.ylim(-0.5, xpos[-1] + 0.5)
    plt.yticks(xpos, data['areas'][:num])
    plt.xlabel('Time [s]')

    fig.add_subplot(gs[0, 1])
    for i in range(num):
        plt.plot(times, rates[i] * 6 + xpos[i])
    plt.xlim(0, 1600)
    plt.ylim(-0.5, xpos[-1] + 0.5)
    plt.yticks([])
    plt.xlabel('Time [s]')

    sns.despine()

    # plt.savefig('dff-firing-rate.svg', transparent=True, dpi=500)
    plt.show()


def visualize_correlation():
    data = np.load('./data/spike_rates/ito_2017-10-26_1_spike_rate.npz')

    dff_corr = np.corrcoef(data['dff'])
    rates_corr = np.corrcoef(data['rates'])

    fig, gs = braintools.visualize.get_figure(1, 2, 4.5, 6.0)
    fig.add_subplot(gs[0, 0])
    sns.heatmap(dff_corr, cmap='viridis', cbar=True)
    plt.title('dF/F Correlation')

    fig.add_subplot(gs[0, 1])
    sns.heatmap(rates_corr, cmap='viridis', cbar=True)
    plt.title('Spike Rates Correlation')

    matrices_corr = np.corrcoef(dff_corr.flatten(), rates_corr.flatten())[0, 1]
    plt.suptitle(f'MSE: {np.mean((dff_corr - rates_corr) ** 2):.5f}, Corr: {matrices_corr:.5f}')

    plt.show()


if __name__ == '__main__':
    pass
    compute_experimental_fc('./data/neural_activity/ito_2017-10-26_1.npz')

    # convert_calcium_to_spike_rate()
    # visualize_spike_rate()
    # visualize_dff_fr()
    # visualize_correlation()

    # obtain_atalas_syn_count('783')
    # obtain_atalas_syn_count('630')

    # visualize_neural_activity()
    # compute_experimental_fc('2017-10-26_1')

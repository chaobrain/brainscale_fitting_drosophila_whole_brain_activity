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


import braintools
import jax
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import seaborn as sns


def visualize_dff_fr():
    data = np.load('./data/spike_rates/ito_2017-10-26_1_spike_rate.npz')
    dff = data['dff'][1:]
    rates = data['rates'][1:]

    simulated_rates = np.load('results/v3/630#2017-10-26_1#100.0Hz#None#mse#NonTempParam#'
                              'NonTempParam#0.000825#20#0.1#2025-03-17-12-38-29/simulated_neuropil_fr.npy')
    simulated_rates = simulated_rates[:1900].T / 100

    max_x = 800
    inc = 0.3

    num = 40
    fig, gs = braintools.visualize.get_figure(1, 3, 8, 5)
    fig.add_subplot(gs[0, 0])
    xpos = np.arange(num) / 3
    times = np.arange(dff.shape[1]) * 1 / 1.2

    for i in range(num):
        plt.plot(times, dff[i] + xpos[i])
    plt.xlim(0, max_x)
    plt.ylim(-inc, xpos[-1] + inc)
    plt.yticks(xpos, data['areas'][:num])
    plt.xlabel('Time [s]')
    plt.title('dF/F')

    fig.add_subplot(gs[0, 1])
    for i in range(num):
        plt.plot(times, rates[i] * 6 + xpos[i])
    plt.xlim(0, max_x)
    plt.ylim(-inc, xpos[-1] + inc)
    plt.yticks([])
    plt.xlabel('Time [s]')
    plt.title('Deconvolved Firing Rate')

    fig.add_subplot(gs[0, 2])
    for i in range(num):
        plt.plot(times, simulated_rates[i] * 6 + xpos[i])
    plt.xlim(0, max_x)
    plt.ylim(-inc, xpos[-1] + inc)
    plt.yticks([])
    plt.xlabel('Time [s]')
    plt.title('Simulated Firing Rate')

    sns.despine()
    plt.savefig('dff-firing-rate.svg', transparent=True, dpi=500)
    plt.show()


def compare_area_correlation():
    max_x = 800
    simulated_file = ('results/v3/630#2017-10-26_1#100.0Hz#None#mse#NonTempParam#'
                      'NonTempParam#0.000825#20#0.1#2025-03-17-12-38-29/simulated_neuropil_fr.npy')
    max_fr = simulated_file.split('/')[2].split('#')[2].replace('Hz', '')
    simulated_rates = np.load(simulated_file)
    simulated_rates = simulated_rates[:max_x].T / float(max_fr)

    data = np.load('./data/spike_rates/ito_2017-10-26_1_spike_rate.npz')
    dff = data['dff'][1:, :max_x]
    rates = data['rates'][1:, :max_x]
    areas = data['areas'][1:]

    correlations = jax.vmap(lambda x, y: jax.numpy.corrcoef(x, y)[0, 1])(simulated_rates, rates)
    correlations = np.asarray(correlations)

    orient = 'h'
    orient = 'v'

    # Create barplot of correlations with area labels
    seaborn.set_theme(font_scale=0.9, style=None)

    if orient == 'h':
        fig, gs = braintools.visualize.get_figure(1, 1, 12, 3)
        ax = fig.add_subplot(gs[0, 0])
        ax = sns.barplot(x=correlations, y=areas, orient='h', ax=ax)
        plt.xlabel('Correlation Coefficient')
        sns.despine()
        plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    else:
        fig, gs = braintools.visualize.get_figure(1, 1, 4, 10)
        ax = fig.add_subplot(gs[0, 0])
        ax = sns.barplot(x=areas, y=correlations, orient='v', ax=ax)
        plt.ylabel('Correlation Coefficient')
        plt.xticks(rotation=90)
        sns.despine()
        plt.title('Correlation of Simulated and Experimental Firing Rates')

    plt.savefig(f'area_correlations-{orient}.svg', transparent=True, dpi=500)
    plt.show()


def compare_correlation_of_correlation_matrix():
    max_x = 800
    simulated_file = ('results/v3/630#2017-10-26_1#100.0Hz#None#mse#NonTempParam#'
                      'NonTempParam#0.000825#20#0.1#2025-03-17-12-38-29/simulated_neuropil_fr.npy')
    max_fr = simulated_file.split('/')[2].split('#')[2].replace('Hz', '')
    simulated_rates = np.load(simulated_file)
    simulated_rates = simulated_rates[:max_x].T / float(max_fr)

    data = np.load('./data/spike_rates/ito_2017-10-26_1_spike_rate.npz')
    dff = data['dff'][1:, :max_x]
    rates = data['rates'][1:, :max_x]
    areas = data['areas'][1:]

    exp_correlation = np.corrcoef(rates)
    sim_correlation = np.corrcoef(simulated_rates)
    exp_correlation = np.asarray(exp_correlation)
    sim_correlation = np.asarray(sim_correlation)

    corr = np.corrcoef(exp_correlation.flatten(), sim_correlation.flatten())
    print(f"Correlation between correlation matrices: {corr[0, 1]:.4f}")

    np.fill_diagonal(exp_correlation, np.nan)
    np.fill_diagonal(sim_correlation, np.nan)

    cmap = 'coolwarm'
    # cmap = 'RdYlBu'
    # cmap = 'bwr'

    seaborn.set_theme(font_scale=1.0, style=None)
    fig, gs = braintools.visualize.get_figure(1, 2, 9, 6)

    # Plot experimental correlation matrix
    ax = fig.add_subplot(gs[0, 0])
    # cax = sns.heatmap(exp_correlation, cmap="coolwarm", vmin=-1, vmax=1,
    #                   square=True, xticklabels=False, yticklabels=False,
    #                   cbar_kws={"shrink": 0.8}, ax=ax)
    im1 = plt.imshow(exp_correlation, cmap=cmap, vmin=-1, vmax=1)
    # plt.colorbar(im1, shrink=0.8)
    plt.title('Experimental Correlation Matrix')
    plt.axis('off')

    # Plot simulated correlation matrix
    ax = fig.add_subplot(gs[0, 1])
    # cax = sns.heatmap(sim_correlation, cmap="coolwarm", vmin=-1, vmax=1,
    #                   square=True, xticklabels=False, yticklabels=False,
    #                   cbar_kws={"shrink": 0.8}, ax=ax)
    im2 = plt.imshow(sim_correlation, cmap=cmap, vmin=-1, vmax=1)
    plt.colorbar(im2, shrink=0.6)
    plt.title('Simulated Correlation Matrix')
    plt.axis('off')

    plt.savefig('correlation_matrices.svg', transparent=True, dpi=500)
    plt.show()


if __name__ == '__main__':
    # visualize_dff_fr()
    compare_area_correlation()
    # compare_correlation_of_correlation_matrix()

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


import glob
import os.path
import platform
import re

if platform.system() != 'Windows':
    import matplotlib

    matplotlib.use('Agg')

import pandas as pd
import braintools
import brainunit as u
import jax
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import seaborn
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from utils import FilePath, split_train_test, read_setting


def pearson_corr(x: jax.Array, y: jax.Array) -> jax.Array:
    # Pearson 相关系数 #
    # ---
    x_mean = jnp.mean(x)
    y_mean = jnp.mean(y)
    xm = x - x_mean
    ym = y - y_mean
    num = jnp.sum(xm * ym)
    den = jnp.sqrt(jnp.sum(xm ** 2) * jnp.sum(ym ** 2))
    return num / den


def _rankdata(x: jax.Array) -> jax.Array:
    # 对每个元素计算其在排序后的位置（1…N）
    # 简易实现：对每对比较求和
    return jnp.argsort(jnp.argsort(x)) + 1


def spearman_corr(x: jax.Array, y: jax.Array) -> jax.Array:
    # Spearman 等级相关系数 #
    # ---
    # Spearman 秩相关系数衡量的是两个序列的秩之间的线性相关性。
    # 它评估的是两个变量之间的单调关系，即使这种关系不是线性的。
    # 与 Pearson 相关系数类似，其值也介于 -1 和 +1 之间。
    rx = _rankdata(x).astype(jnp.float32)
    ry = _rankdata(y).astype(jnp.float32)
    return pearson_corr(rx, ry)


def kendall_tau(x: jax.Array, y: jax.Array) -> jax.Array:
    n = x.shape[0]
    # 生成所有 i<j 对
    ii, jj = jnp.triu_indices(n, k=1)
    xi, xj = x[ii], x[jj]
    yi, yj = y[ii], y[jj]
    concordant = jnp.sum((xi - xj) * (yi - yj) > 0)
    discordant = jnp.sum((xi - xj) * (yi - yj) < 0)
    return (concordant - discordant) / (0.5 * n * (n - 1))


def cross_correlation(x: jax.Array, y: jax.Array):
    # 互相关（Cross-Correlation） #
    # ----
    # 互相关衡量的是两个序列在不同时间偏移下的相似性。
    # 它常用于信号处理领域，用来寻找一个序列在另一个
    # 序列中的出现位置，或者衡量两个信号之间的延迟。

    # 等价于 signal.correlate(x, y, mode='full')
    nx, ny = x.shape[0], y.shape[0]
    # pad y
    y_pad = jnp.pad(y, (nx - 1, nx - 1))

    # for each lag compute dot
    def body_fun(i, acc):
        segment = y_pad[i:i + nx]
        return acc.at[i].set(jnp.dot(x, segment))

    return jax.lax.fori_loop(0, 2 * nx - 1, body_fun, jnp.zeros(2 * nx - 1))


def jaccard_similarity(a: jax.Array, b: jax.Array) -> jax.Array:
    # 杰卡德相似度（Jaccard Similarity） #
    # ---
    # 用于二值序列/集合，衡量交并比。
    #
    # 假设 a,b 为 0/1 向量
    intersection = jnp.sum(a & b)
    union = jnp.sum(a | b)
    return intersection / union


def dtw_distance(x: jax.Array, y: jax.Array) -> jax.Array:
    # 动态时间规整（DTW, Dynamic Time Warping）距离 #
    # ---
    # 虽然不是“相关系数”，但常用于衡量时序形状相似性。

    """
    Computes the Dynamic Time Warping (DTW) distance using nested jax.lax.fori_loop.
    """
    nx, ny = x.shape[0], y.shape[0]
    # Initialize cost matrix D
    D = jnp.full((nx + 1, ny + 1), jnp.inf)
    D = D.at[0, 0].set(0.0)

    def outer_loop_body(i, D_carry):
        # Inner loop computes values for row 'i'
        def inner_loop_body(j, D_inner_carry):
            cost = jnp.abs(x[i - 1] - y[j - 1])
            min_prev = jnp.minimum(jnp.minimum(D_inner_carry[i - 1, j],  # Value from the previous row
                                               D_inner_carry[i, j - 1]),
                                   # Value from the previous column in the current row
                                   D_inner_carry[i - 1, j - 1])  # Diagonal value from the previous row
            D_updated = D_inner_carry.at[i, j].set(cost + min_prev)
            return D_updated

        # Execute the inner loop for j from 1 to ny
        # The state D_carry is passed to and updated by the inner loop
        D_after_inner = jax.lax.fori_loop(1, ny + 1, inner_loop_body, D_carry)
        return D_after_inner

    # Execute the outer loop for i from 1 to nx
    # The initial state is the initialized D matrix
    D_final = jax.lax.fori_loop(1, nx + 1, outer_loop_body, D)

    return D_final[nx, ny]


def _visualize_low_rank_connectivity(filepath: str):
    params = braintools.file.msgpack_load(os.path.join(filepath, 'first-round-checkpoint.msgpack'))

    lora = params['interaction']['lora']['weight_op']
    B = lora['B']
    A = lora['A']['mantissa']

    # Get original dimensions
    print(f"Matrix B shape: {B.shape}, Matrix A shape: {A.shape}")
    fig, gs = braintools.visualize.get_figure(1, 1, 12, 10)
    low_rank_matrix = B @ A
    ax = fig.add_subplot(gs[0, 0])
    im = ax.matshow(low_rank_matrix, cmap='viridis')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(im)

    path = os.path.join(filepath, 'analysis')
    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, 'low_rank_matrix.pdf'))
    plt.close()


def visualize_low_rank_connectivity_matrix():
    for filepath in glob.glob('results/*'):
        filepath = filepath.replace('\\', '/')
        # for filepath in [
        #     'results/630#2017-10-26_1#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.25#0.6##2025-04-15-17-40-29',
        #     'results/630#2017-10-26_1#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.3#0.8##2025-04-17-10-38-17',
        #     'results/630#2017-10-26_1#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.5#0.6##2025-04-15-17-50-54',
        #     'results/630#2017-10-30_1#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.25#0.6##2025-04-15-17-50-26',
        #     'results/630#2017-10-30_1#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.3#0.6##2025-04-17-10-37-21',
        #     'results/630#2017-10-30_1#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.5#0.6##2025-04-15-17-42-03',
        #     'results/630#2017-10-30_1#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.5#0.6##2025-04-15-17-50-22',
        #     'results/630#2018-10-19_2#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-18-23-43-39',
        #     'results/630#2018-10-20_1#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-20-10-08-30',
        #     'results/630#2018-10-31_1#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-21-01-56-22',
        #     'results/630#2018-11-03_2#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-22-13-03-28',
        #     'results/630#2018-11-03_3#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-18-23-43-48',
        #     'results/630#2018-11-03_4#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-19-16-42-16',
        #     'results/630#2018-11-03_5#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-20-09-41-05',
        #     'results/630#2018-12-12_2#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-20-23-29-39',
        #     'results/630#2018-12-12_3#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-20-22-59-28',
        #     'results/630#2018-12-12_4#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-21-19-14-11',
        #     'results/630#2018-12-14_1#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-22-05-05-42',
        #     'results/630#2018-12-14_2#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-22-15-18-57',
        #     'results/630#2018-12-14_3#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-21-16-35-58'
        # ]:
        try:
            _visualize_low_rank_connectivity(filepath)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")


def _visualize_experimental_and_simulated_firing_rates(filepath: str):
    print(filepath)
    args = FilePath.from_filepath(filepath)
    settings = read_setting(filepath)

    simulated_rates = np.load(os.path.join(filepath, 'neuropil_fr_predictions.npy')).T
    data = np.load(f'./data/spike_rates/ito_{args.neural_activity_id}_spike_rate.npz')
    dff = data['dff'][1:]
    # remove the first time bin,
    # because the prediction is starting from the second time bin
    rates = data['rates'][1:, 1:] * args.neural_activity_max_fr / u.Hz
    areas = data['areas'][1:]
    n_train, n_test = split_train_test(rates.shape[1], args.split, settings['batch_size'])
    n_warpup = 100

    print('n_warpup:', n_warpup)
    print('n_train:', n_train)
    print('n_test:', n_test)

    # [n_neuropil, n_warpup]
    _warmup_rates = rates[:, :n_warpup]
    _simulated_warmup_rates = simulated_rates[:, :n_warpup]

    # [n_neuropil, n_train]
    _train_rates = rates[:, n_warpup:n_train]
    _simulated_train_rates = simulated_rates[:, n_warpup:n_train]

    # [n_neuropil, n_test]
    _test_rates = rates[:, n_train:]
    _simulated_test_rates = simulated_rates[:, n_train:]

    test_cor = np.asarray(jax.vmap(lambda x, y: pearson_corr(x, y))(
        rates[:, n_warpup:],
        simulated_rates[:, n_warpup:],
    ))

    # test_cor = np.asarray(jax.vmap(lambda x, y: pearson_corr(x, y))(_simulated_test_rates, _test_rates))
    # test_cor = np.asarray(jax.vmap(lambda x, y: spearman_corr(x, y))(_simulated_test_rates, _test_rates))
    # test_cor = np.asarray(jax.vmap(lambda x, y: kendall_tau(x, y))(_simulated_test_rates, _test_rates))
    # test_cor = np.asarray(jax.vmap(lambda x, y: dtw_distance(x, y))(_simulated_test_rates, _test_rates))
    test_cor = np.copy(test_cor)

    # Handle NaNs in correlation calculation
    nan_mask = np.isnan(test_cor)
    if np.all(nan_mask):
        print("All test correlations are NaN. Skipping plot.")
        return
    test_cor[nan_mask] = -np.inf  # Replace NaNs for sorting, treat them as worst correlation

    # Get indices for top 5 and bottom 5 correlations
    sorted_indices = np.argsort(test_cor)
    top5_indices = sorted_indices[-5:][::-1]  # Highest correlations first
    bottom5_indices = sorted_indices[:5]  # Lowest correlations first
    plot_indices = np.concatenate((top5_indices, bottom5_indices))

    # Plotting
    fig, gs = braintools.visualize.get_figure(5, 2, 3, 8)  # 5 rows, 2 columns
    time_warmup = np.arange(n_warpup)
    time_train = np.arange(n_warpup, n_train)
    time_test = np.arange(n_train, n_train + n_test)

    for i, idx in enumerate(plot_indices):
        row = i % 5
        col = i // 5  # 0 for top 5, 1 for bottom 5

        ax = fig.add_subplot(gs[row, col])

        # Plot warmup data
        ax.plot(time_warmup, _warmup_rates[idx], label='Exp Train', color='blue', alpha=0.7)
        ax.plot(time_warmup, _simulated_warmup_rates[idx], label='Sim Train', color='lightblue', linestyle='--')

        # Plot training data
        ax.plot(time_train, _train_rates[idx], label='Exp Train', color='blue', alpha=0.7)
        ax.plot(time_train, _simulated_train_rates[idx], label='Sim Train', color='lightblue', linestyle='--')

        # Plot testing data
        ax.plot(time_test, _test_rates[idx], label='Exp Test', color='red', alpha=0.7)
        ax.plot(time_test, _simulated_test_rates[idx], label='Sim Test', color='orange', linestyle='--')

        # Add vertical line separating train/test
        ax.axvline(n_train - 0.5, color='gray', linestyle=':', linewidth=1)

        correlation_val = test_cor[idx] if not nan_mask[idx] else np.nan
        title_prefix = "Top" if col == 0 else "Bottom"
        ax.set_title(f'{title_prefix} {row + 1}: {areas[idx]} (Test Corr: {correlation_val:.3f})', fontsize=8)
        ax.set_xlabel('Time Bin', fontsize=7)
        ax.set_ylabel('Firing Rate (Hz)', fontsize=7)
        ax.tick_params(axis='both', which='major', labelsize=6)
        if i == 0:  # Add legend only once
            ax.legend(fontsize=6, loc='upper right')

    # # plt.suptitle(f'Comparison for {args.neural_activity_id} ({os.path.basename(filepath)})', fontsize=10)
    # path = os.path.join(filepath, 'analysis')
    # os.makedirs(path, exist_ok=True)
    # plot_filename = os.path.join(path, 'top_bottom_5_area_correlation_neuropils.pdf')
    # plt.savefig(plot_filename)
    # print(f"Saved plot to {plot_filename}")
    plt.show()
    plt.close(fig)


def _visualize_firing_rates(
    n_warmup: int,
    n_train: int,
    n_test: int,
    rates: np.ndarray,
    simulated_rates: np.ndarray,
    plot_filename: str = None
):
    # Plotting
    seaborn.set_theme(style=None)
    # plt.style.use('seaborn-v0_8-whitegrid')
    fig, gs = braintools.visualize.get_figure(len(rates), 1, 2, 10)  # 5 rows, 2 columns

    time_warmup = np.arange(n_warmup)
    time_train = np.arange(n_warmup, n_train)
    time_test = np.arange(n_train, n_train + n_test)

    # Plot each neuropil
    axes = []
    for idx in range(rates.shape[0]):
        row = idx % 5
        col = idx // 5
        ax = fig.add_subplot(gs[row, col])
        axes.append(ax)

        # Color coding by phase
        # Warmup phase
        ax.plot(time_warmup, rates[idx, :n_warmup],
                color='#1f77b4', linewidth=1.5, alpha=0.9, label='Exp warmup')
        ax.plot(time_warmup, simulated_rates[idx, :n_warmup],
                color='#1f77b4', linewidth=1, linestyle='--', alpha=0.7, label='Sim warmup')

        # Training phase
        ax.plot(time_train, rates[idx, n_warmup:n_train],
                color='#2ca02c', linewidth=1.5, alpha=0.9, label='Exp train')
        ax.plot(time_train, simulated_rates[idx, n_warmup:n_train],
                color='#2ca02c', linewidth=1, linestyle='--', alpha=0.7, label='Sim train')

        # Test phase
        ax.plot(time_test, rates[idx, n_train:],
                color='#d62728', linewidth=1.5, alpha=0.9, label='Exp test')
        ax.plot(time_test, simulated_rates[idx, n_train:],
                color='#d62728', linewidth=1, linestyle='--', alpha=0.7, label='Sim test')

        # Add phase transition line
        ax.axvline(n_train - 0.5, color='black', linestyle=':', linewidth=0.8)
        ax.axvline(n_warmup - 0.5, color='black', linestyle=':', linewidth=0.8)

        # Phase labels (only add to top row)
        if row == 0:
            ax.text(n_warmup / 2, ax.get_ylim()[1] * 0.9, 'Warmup',
                    ha='center', fontsize=7, bbox=dict(facecolor='white', alpha=0.7, pad=2))
            ax.text((n_warmup + n_train) / 2, ax.get_ylim()[1] * 0.9, 'Train',
                    ha='center', fontsize=7, bbox=dict(facecolor='white', alpha=0.7, pad=2))
            ax.text((n_train + n_train + n_test) / 2, ax.get_ylim()[1] * 0.9, 'Test',
                    ha='center', fontsize=7, bbox=dict(facecolor='white', alpha=0.7, pad=2))

        # # Add titles and correlation values
        # correlation_val = test_cor[idx] if not nan_mask[idx] else np.nan
        # title_prefix = "Top" if col == 0 else "Bottom"
        # ax.set_title(f'{title_prefix} {row + 1}: {areas[idx]}\nCorr: {correlation_val:.3f}',
        #              fontsize=9, fontweight='bold')

        # Axis labels and ticks
        ax.set_ylabel('Firing rate (Hz)', fontsize=8)
        ax.tick_params(axis='y', which='major', labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(-1, n_train + n_test + 1)
        if idx + 1 != len(rates):
            ax.set_xticks([])
        else:
            ax.set_xlabel('Time bin', fontsize=8)

    fig.align_ylabels(axes)

    # Add legend to the first subplot only
    legend_elements = [
        plt.Line2D([0], [0], color='#1f77b4', lw=1.5, label='Exp warmup'),
        plt.Line2D([0], [0], color='#1f77b4', lw=1, linestyle='--', label='Sim warmup'),
        plt.Line2D([0], [0], color='#2ca02c', lw=1.5, label='Exp train'),
        plt.Line2D([0], [0], color='#2ca02c', lw=1, linestyle='--', label='Sim train'),
        plt.Line2D([0], [0], color='#d62728', lw=1.5, label='Exp test'),
        plt.Line2D([0], [0], color='#d62728', lw=1, linestyle='--', label='Sim test')
    ]
    plt.legend(
        handles=legend_elements,
        bbox_to_anchor=(1.01, 1.0),
        ncol=1,
        fontsize=8,
        frameon=True
    )

    # # Add global title
    # fig.suptitle(f'Neural Activity Comparison: {args.neural_activity_id}',
    #              fontsize=12, y=0.995, fontweight='bold')

    # # plt.suptitle(f'Comparison for {args.neural_activity_id} ({os.path.basename(filepath)})', fontsize=10)
    if plot_filename is None:
        plt.show()
    else:
        os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
        plt.savefig(plot_filename, transparent=True, dpi=500)
    plt.close(fig)


def _visualize_experimental_and_simulated_firing_rates_v2(filepath: str):
    print(filepath)
    args = FilePath.from_filepath(filepath)
    settings = read_setting(filepath)

    simulated_rates = np.load(os.path.join(filepath, 'neuropil_fr_predictions.npy')).T
    data = np.load(f'./data/spike_rates/ito_{args.neural_activity_id}_spike_rate.npz')
    dff = data['dff'][1:]
    # remove the first time bin,
    # because the prediction is starting from the second time bin
    rates = data['rates'][1:, 1:] * args.neural_activity_max_fr / u.Hz
    areas = data['areas'][1:]
    n_train, n_test = split_train_test(rates.shape[1], args.split, settings['batch_size'])
    n_warmup = 100

    print('n_warpup:', n_warmup)
    print('n_train:', n_train)
    print('n_test:', n_test)

    # [n_neuropil, n_warpup]
    _warmup_rates = rates[:, :n_warmup]
    _simulated_warmup_rates = simulated_rates[:, :n_warmup]

    # [n_neuropil, n_train]
    _train_rates = rates[:, n_warmup:n_train]
    _simulated_train_rates = simulated_rates[:, n_warmup:n_train]

    # [n_neuropil, n_test]
    _test_rates = rates[:, n_train:]
    _simulated_test_rates = simulated_rates[:, n_train:]

    test_cor = np.asarray(jax.vmap(lambda x, y: pearson_corr(x, y))(rates[:, n_warmup:], simulated_rates[:, n_warmup:]))
    # test_cor = np.asarray(jax.vmap(lambda x, y: pearson_corr(x, y))(_simulated_test_rates, _test_rates))
    # test_cor = np.asarray(jax.vmap(lambda x, y: spearman_corr(x, y))(_simulated_test_rates, _test_rates))
    # test_cor = np.asarray(jax.vmap(lambda x, y: kendall_tau(x, y))(_simulated_test_rates, _test_rates))
    # test_cor = np.asarray(jax.vmap(lambda x, y: dtw_distance(x, y))(_simulated_test_rates, _test_rates))

    # Handle NaNs in correlation calculation
    nan_mask = np.isnan(test_cor)
    if np.all(nan_mask):
        print("All test correlations are NaN. Skipping plot.")
        return
    test_cor = np.copy(test_cor)
    test_cor[nan_mask] = -np.inf  # Replace NaNs for sorting, treat them as worst correlation

    # Select top 5 and bottom 5 correlations
    sorted_indices = np.argsort(test_cor)
    top5_indices = sorted_indices[-5:][::-1]
    bottom5_indices = sorted_indices[:5]
    plot_indices = top5_indices

    for i in range(0, rates.shape[0], 5):
        _visualize_firing_rates(
            n_warmup,
            n_train,
            n_test,
            rates[i:i + 5],
            simulated_rates[i:i + 5],
            os.path.join(filepath, 'analysis', f'firing-rate-comparison-{i}-{i + 1}.png')
        )


def _visualize_experimental_and_simulated_firing_rates_v3(filepath: str):
    print(filepath)
    args = FilePath.from_filepath(filepath)
    settings = read_setting(filepath)

    simulated_rates = np.load(os.path.join(filepath, 'neuropil_fr_predictions.npy')).T
    data = np.load(f'./data/spike_rates/ito_{args.neural_activity_id}_spike_rate.npz')
    dff = data['dff'][1:]
    # remove the first time bin,
    # because the prediction is starting from the second time bin
    rates = data['rates'][1:, 1:] * args.neural_activity_max_fr / u.Hz
    areas = data['areas'][1:]
    n_train, n_test = split_train_test(rates.shape[1], args.split, settings['batch_size'])
    n_warmup = 100

    print('n_warpup:', n_warmup)
    print('n_train:', n_train)
    print('n_test:', n_test)

    # [n_neuropil, n_warpup]
    _warmup_rates = rates[:, :n_warmup]
    _simulated_warmup_rates = simulated_rates[:, :n_warmup]

    # [n_neuropil, n_train]
    _train_rates = rates[:, n_warmup:n_train]
    _simulated_train_rates = simulated_rates[:, n_warmup:n_train]

    # [n_neuropil, n_test]
    _test_rates = rates[:, n_train:]
    _simulated_test_rates = simulated_rates[:, n_train:]

    test_cor = np.asarray(jax.vmap(lambda x, y: pearson_corr(x, y))(rates[:, n_warmup:], simulated_rates[:, n_warmup:]))
    # test_cor = np.asarray(jax.vmap(lambda x, y: pearson_corr(x, y))(_simulated_test_rates, _test_rates))
    # test_cor = np.asarray(jax.vmap(lambda x, y: spearman_corr(x, y))(_simulated_test_rates, _test_rates))
    # test_cor = np.asarray(jax.vmap(lambda x, y: kendall_tau(x, y))(_simulated_test_rates, _test_rates))
    # test_cor = np.asarray(jax.vmap(lambda x, y: dtw_distance(x, y))(_simulated_test_rates, _test_rates))

    # Handle NaNs in correlation calculation
    nan_mask = np.isnan(test_cor)
    if np.all(nan_mask):
        print("All test correlations are NaN. Skipping plot.")
        return
    test_cor = np.copy(test_cor)
    test_cor[nan_mask] = -np.inf  # Replace NaNs for sorting, treat them as worst correlation

    # Select top 5 and bottom 5 correlations
    sorted_indices = np.argsort(test_cor)
    top_indices = sorted_indices[-2:][::-1]

    _visualize_firing_rates(
        n_warmup,
        n_train,
        n_test,
        rates[top_indices],
        simulated_rates[top_indices],
        os.path.join(filepath, 'analysis', f'firing-rate-comparison-top{len(top_indices)}.pdf')
    )


def visualize_experimental_and_simulated_firing_rates():
    # 优：
    # results/630#2018-10-19_1#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-18-10-52-34
    # results/630#2018-10-31_1#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-21-01-56-22
    # results/630#2018-11-03_3#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-18-23-43-48

    # 次优：
    # results/630#2018-12-14_1#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-22-05-05-42
    # results/630#2018-11-03_5#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-20-09-41-05
    # results/630#2018-12-14_3#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-21-16-35-58

    for filepath in glob.glob('results/*'):
        filepath = filepath.replace('\\', '/')
        # _visualize_experimental_and_simulated_firing_rates(filepath)
        # _visualize_experimental_and_simulated_firing_rates_v2(filepath)
        _visualize_experimental_and_simulated_firing_rates_v3(filepath)

    # _visualize_experimental_and_simulated_firing_rates_v2(
    #     'results/630#2018-10-19_1#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-18-10-52-34'
    # )


def _visualize_firing_rates_v4(
    n_warmup: int,
    n_train: int,
    n_test: int,
    rates: np.ndarray,
    trained_rates: np.ndarray,
    untrained_rates: np.ndarray,
    plot_filename: str = None,
    legend: bool = True
):
    # Set nicer theme with clean style
    seaborn.set_theme(style='ticks')
    fig, gs = braintools.visualize.get_figure(3, len(rates), 2, 12)  # 3 rows, len(rates) columns

    # Time arrays for each phase
    time_gap = 1 / 1.2 / u.Hz
    time_warmup = (np.arange(n_warmup) * time_gap).to_decimal(u.second)
    time_train = (np.arange(n_warmup, n_train) * time_gap).to_decimal(u.second)
    time_test = (np.arange(n_train, n_train + n_test) * time_gap).to_decimal(u.second)
    t_warmup = n_warmup * time_gap / u.second
    t_train = n_train * time_gap / u.second
    t_test = n_test * time_gap / u.second

    # Color palette (using more distinguished colors)
    colors = {
        'exp': '#1f77b4',  # blue
        'trained': '#2ca02c',  # green
        'untrained': '#d62728'  # red
    }

    # Phase styling
    phase_styles = {
        'warmup': {'alpha': 0.8, 'lw': 1.2},
        'train': {'alpha': 1.0, 'lw': 1.5},
        'test': {'alpha': 0.9, 'lw': 1.3}
    }

    # Common axis styling function
    def style_axis(ax, title=None, with_xlabel=False):
        ax.set_ylabel('Firing rate (Hz)', fontsize=9)
        ax.tick_params(axis='both', which='major', labelsize=8)
        seaborn.despine(ax=ax)
        ax.set_xlim(-5, t_train + t_test + 5)

        # Add vertical separators for phases
        ax.axvline(t_warmup - 0.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
        ax.axvline(t_train - 0.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)

        # Add background shading for phases
        ax.axvspan(-5, t_warmup - 0.5, alpha=0.1, color='lightblue')
        ax.axvspan(t_warmup - 0.5, t_train - 0.5, alpha=0.1, color='lightgreen')
        ax.axvspan(t_train - 0.5, t_train + t_test + 5, alpha=0.1, color='mistyrose')

        if title:
            ax.set_title(title, fontsize=10, fontweight='bold')
        if with_xlabel:
            ax.set_xlabel('Time [s]', fontsize=9)
        else:
            ax.set_xticks([])

    # Legend elements
    legend_elements = []

    # Plot each neuropil
    for idx in range(rates.shape[0]):
        # Experimental data plot
        ax_exp = fig.add_subplot(gs[0, idx])
        ax_exp.plot(time_warmup, rates[idx, :n_warmup], color=colors['exp'], **phase_styles['warmup'], label='Warmup')
        ax_exp.plot(time_train, rates[idx, n_warmup:n_train], color=colors['exp'], **phase_styles['train'],
                    label='Train')
        ax_exp.plot(time_test, rates[idx, n_train:], color=colors['exp'], **phase_styles['test'], label='Test')
        style_axis(ax_exp, title="Experimental Data")

        # Trained model data plot
        ax_trained = fig.add_subplot(gs[1, idx])
        ax_trained.plot(time_warmup, trained_rates[idx, :n_warmup], color=colors['trained'], **phase_styles['warmup'])
        ax_trained.plot(time_train, trained_rates[idx, n_warmup:n_train], color=colors['trained'],
                        **phase_styles['train'])
        ax_trained.plot(time_test, trained_rates[idx, n_train:], color=colors['trained'], **phase_styles['test'])
        style_axis(ax_trained, title="Trained Model")

        # Untrained model data plot
        ax_untrained = fig.add_subplot(gs[2, idx])
        ax_untrained.plot(time_warmup, untrained_rates[idx, :n_warmup], color=colors['untrained'],
                          **phase_styles['warmup'])
        ax_untrained.plot(time_train, untrained_rates[idx, n_warmup:n_train], color=colors['untrained'],
                          **phase_styles['train'])
        ax_untrained.plot(time_test, untrained_rates[idx, n_train:], color=colors['untrained'], **phase_styles['test'])
        style_axis(ax_untrained, title="Untrained Model", with_xlabel=True)

        # Add column title
        if idx == 0:
            # Create legend elements for the first column only
            for model, color in colors.items():
                legend_elements.append(plt.Line2D([0], [0], color=color, lw=2, label=model.capitalize()))

            # Add phase legend elements
            for phase, style in phase_styles.items():
                legend_elements.append(plt.Line2D([0], [0], color='black', lw=style['lw'],
                                                  alpha=style['alpha'], label=f"{phase.capitalize()} phase"))

    # Add legend outside the plots
    if legend:
        fig.legend(
            handles=legend_elements,
            # loc='center left',
            bbox_to_anchor=(1.0, 0.5),
            fontsize=9,
            frameon=True,
            # title="Legend",
            title_fontsize=10
        )

    # # Adjust layout
    plt.tight_layout(rect=(0, 0, 0.88, 0.95))

    # Save or show plot
    if plot_filename is None:
        plt.show()
    else:
        os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
        plt.savefig(plot_filename, transparent=True, dpi=300, bbox_inches='tight')
    plt.close(fig)


def _visualize_experimental_and_simulated_firing_rates_v4(filepath: str):
    print(filepath)
    args = FilePath.from_filepath(filepath)
    settings = read_setting(filepath)
    if not os.path.exists(os.path.join(filepath, 'neuropil_fr_predictions_untrained.npy')):
        return

    trained_rates = np.load(os.path.join(filepath, 'neuropil_fr_predictions.npy')).T
    untrained_rates = np.load(os.path.join(filepath, 'neuropil_fr_predictions_untrained.npy')).T
    data = np.load(f'./data/spike_rates/ito_{args.neural_activity_id}_spike_rate.npz')
    rates = data['rates'][1:, 1:] * args.neural_activity_max_fr / u.Hz
    n_train, n_test = split_train_test(rates.shape[1], args.split, settings['batch_size'])
    n_warmup = 100

    # correlations
    test_cor = np.asarray(jax.vmap(lambda x, y: pearson_corr(x, y))(rates[:, n_warmup:], trained_rates[:, n_warmup:]))

    # Handle NaNs in correlation calculation
    nan_mask = np.isnan(test_cor)
    if np.all(nan_mask):
        print("All test correlations are NaN. Skipping plot.")
        return
    test_cor = np.copy(test_cor)
    test_cor[nan_mask] = -np.inf  # Replace NaNs for sorting, treat them as worst correlation

    # Select top 5 and bottom 5 correlations
    sorted_indices = np.argsort(test_cor)
    top_indices = sorted_indices[-1:][::-1]

    _visualize_firing_rates_v4(
        n_warmup,
        n_train,
        n_test,
        rates[top_indices],
        trained_rates[top_indices],
        untrained_rates[top_indices],
        os.path.join(filepath, 'analysis', f'fr-comparison-exp-trained-untrained.pdf'),
        legend=False,
    )


def visualize_experimental_trained_untrained_firing_rates():
    for filepath in glob.glob('results/*'):
        filepath = filepath.replace('\\', '/')
        _visualize_experimental_and_simulated_firing_rates_v4(filepath)


def _compare_area_correlation(
    filepath: str,
    savefig: bool = False
):
    args = FilePath.from_filepath(filepath)
    settings = read_setting(filepath)

    simulated_rates = np.load(os.path.join(filepath, 'neuropil_fr_predictions.npy')).T
    data = np.load(f'./data/spike_rates/ito_{args.neural_activity_id}_spike_rate.npz')
    dff = data['dff'][1:]
    rates = data['rates'][1:, 1:] * args.neural_activity_max_fr / u.Hz
    areas = data['areas'][1:]
    n_train, n_test = split_train_test(rates.shape[1], args.split, settings['batch_size'])
    print('n_train:', n_train)
    print('n_test:', n_test)

    _train_rates = rates[:, :n_train]
    _simulated_train_rates = simulated_rates[:, :n_train]

    _test_rates = rates[:, n_train:]
    _simulated_test_rates = simulated_rates[:, n_train:]

    train_cor = np.asarray(jax.vmap(lambda x, y: jax.numpy.corrcoef(x, y)[0, 1])(_simulated_train_rates, _train_rates))
    test_cor = np.asarray(jax.vmap(lambda x, y: jax.numpy.corrcoef(x, y)[0, 1])(_simulated_test_rates, _test_rates))

    orient = 'h'
    orient = 'v'

    # Create barplot of correlations with area labels
    seaborn.set_theme(font_scale=0.9, style=None)

    if orient == 'h':
        fig, gs = braintools.visualize.get_figure(1, 2, 12, 3)
        ax = fig.add_subplot(gs[0, 0])
        seaborn.barplot(x=train_cor, y=areas, orient='h', ax=ax)
        plt.xlabel('Correlation Coefficient (Train)')
        plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
        seaborn.despine()

        ax = fig.add_subplot(gs[0, 1])
        seaborn.barplot(x=test_cor, y=areas, orient='h', ax=ax)
        plt.xlabel('Correlation Coefficient (Test)')
        plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
        seaborn.despine()

    else:
        fig, gs = braintools.visualize.get_figure(2, 1, 4, 10)
        ax = fig.add_subplot(gs[0, 0])
        seaborn.barplot(x=areas, y=train_cor, orient='v', ax=ax)
        plt.title('Training Phase')
        plt.ylabel('Correlation Coefficient')
        plt.xticks(rotation=90)
        seaborn.despine()

        ax = fig.add_subplot(gs[1, 0])
        seaborn.barplot(x=areas, y=test_cor, orient='v', ax=ax)
        plt.title('Testing Phase')
        plt.ylabel('Correlation Coefficient')
        plt.xticks(rotation=90)
        seaborn.despine()

    plt.suptitle('Correlation of Simulated and Experimental Firing Rates')

    if savefig:
        filepath = os.path.join(filepath, 'analysis')
        os.makedirs(filepath, exist_ok=True)
        plt.savefig(os.path.join(filepath, f'area_correlations-{orient}.png'), transparent=True, dpi=500)
    else:
        plt.show()
    plt.close()


def compare_area_correlation():
    for filepath in glob.glob('results/*'):
        filepath = filepath.replace('\\', '/')
        _compare_area_correlation(filepath, savefig=False)

    # for filepath in [
    #     # 优：
    #     'results/630#2018-10-19_1#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-18-10-52-34',
    #     'results/630#2018-10-31_1#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-21-01-56-22',
    #     'results/630#2018-11-03_3#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-18-23-43-48',
    #
    #     # 次优：
    #     'results/630#2018-12-14_1#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-22-05-05-42',
    #     'results/630#2018-11-03_5#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-20-09-41-05',
    #     'results/630#2018-12-14_3#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-21-16-35-58',
    # ]:
    #     _compare_area_correlation(filepath)


def _compare_area_correlation_trained_untrained(filepath: str, savefig: bool = False, orient='v'):
    args = FilePath.from_filepath(filepath)
    settings = read_setting(filepath)
    if not os.path.exists(os.path.join(filepath, 'neuropil_fr_predictions_untrained.npy')):
        return

    assert orient in ['h', 'v'], "orient must be 'h' or 'v'"
    trained_rates = np.load(os.path.join(filepath, 'neuropil_fr_predictions.npy')).T
    untrained_rates = np.load(os.path.join(filepath, 'neuropil_fr_predictions_untrained.npy')).T
    data = np.load(f'./data/spike_rates/ito_{args.neural_activity_id}_spike_rate.npz')
    rates = data['rates'][1:, 1:] * args.neural_activity_max_fr / u.Hz
    areas = data['areas'][1:]
    n_train, n_test = split_train_test(rates.shape[1], args.split, settings['batch_size'])
    print('n_train:', n_train)
    print('n_test:', n_test)

    corr_fn = pearson_corr
    corr_fn = spearman_corr
    corr_fn = dtw_distance
    corr_fn = kendall_tau

    trained_cor = np.asarray(jax.vmap(corr_fn)(rates, trained_rates))
    untrained_cor = np.asarray(jax.vmap(corr_fn)(rates, untrained_rates))

    percent = jnp.sum(trained_cor > untrained_cor) / trained_cor.shape[0]
    print(percent)
    print(f'trained model, mean ± std = {np.mean(trained_cor)} ± {np.std(trained_cor)}')
    print(f'untrained model, mean ± std = {np.mean(untrained_cor)} ± {np.std(untrained_cor)}')

    # Create barplot of correlations with area labels
    seaborn.set_theme(font_scale=0.9, style=None)

    # Create barplot comparing trained and untrained correlations
    fig, gs = braintools.visualize.get_figure(1, 1, 4.5, 12)  # Adjust figure size for better readability
    # fig, gs = braintools.visualize.get_figure(1, 1, 4, 8)  # Adjust figure size for better readability
    ax = fig.add_subplot(gs[0, 0])

    # Create a DataFrame for easier plotting
    df = pd.DataFrame({
        'Area': np.concatenate([areas, areas]),
        'Correlation': np.concatenate([trained_cor, untrained_cor]),
        'Model': ['Trained'] * len(areas) + ['Untrained'] * len(areas)
    })

    # Plot grouped barplot
    seaborn.barplot(x='Area', y='Correlation', hue='Model', data=df, ax=ax)

    # Improve visualization
    plt.xticks(rotation=90, ha='right', fontsize=8)  # Rotate x-axis labels for readability
    plt.xlabel('Brain Area', fontsize=10)  # Set x-axis label and fontsize
    plt.ylabel('Correlation Coefficient', fontsize=10)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    plt.legend(fontsize=8)
    seaborn.despine()

    # Save figure if requested
    if savefig:
        analysis_path = os.path.join(filepath, 'analysis')
        os.makedirs(analysis_path, exist_ok=True)
        plt.savefig(
            os.path.join(analysis_path, f'trained_untrained_area_correlations-{orient}-{percent:.3f}.pdf'),
            dpi=300,
            bbox_inches='tight',
            transparent=True,
        )
    else:
        plt.show()
    plt.close()


def compare_are_correlation_trained_untrained():
    for filepath in glob.glob('results/*'):
        filepath = filepath.replace('\\', '/')
        # _compare_area_correlation_trained_untrained(filepath, savefig=True)
        _compare_area_correlation_trained_untrained(filepath, savefig=False)


def _compare_correlation_matrix(
    filepath: str,
    show_ticks: bool = False,
    savefig: bool = False
):
    print(filepath)
    args = FilePath.from_filepath(filepath)
    settings = read_setting(filepath)

    simulated_rates = np.load(os.path.join(filepath, 'neuropil_fr_predictions.npy')).T
    data = np.load(f'./data/spike_rates/ito_{args.neural_activity_id}_spike_rate.npz')
    dff = data['dff'][1:]
    rates = data['rates'][1:, 1:] * args.neural_activity_max_fr / u.Hz
    areas = data['areas'][1:]
    n_train, n_test = split_train_test(rates.shape[1], args.split, settings['batch_size'])
    print('n_train:', n_train)
    print('n_test:', n_test)

    cmap = 'coolwarm'

    def show(ax_train, ax_test, split):
        if split == 'train':
            _rates = rates[:, :n_train]
            _simulated_rates = simulated_rates[:, :n_train]
        elif split == 'test':
            _rates = rates[:, n_train:]
            _simulated_rates = simulated_rates[:, n_train:]
        else:
            raise ValueError(f"Invalid split: {split}")

        exp_correlation = np.corrcoef(_rates)
        sim_correlation = np.corrcoef(_simulated_rates)
        exp_correlation = np.asarray(exp_correlation)
        sim_correlation = np.asarray(sim_correlation)

        corr = np.corrcoef(exp_correlation.flatten(), sim_correlation.flatten())
        sim = corr[0, 1]
        print(f"Correlation between correlation matrices: {sim:.4f}")

        np.fill_diagonal(exp_correlation, np.nan)
        np.fill_diagonal(sim_correlation, np.nan)

        if show_ticks:
            im1 = ax_train.imshow(exp_correlation, cmap=cmap, vmin=-1, vmax=1)
            ax_train.set_title('Experimental Correlation Matrix')
            # Add area labels
            ax_train.set_xticks(np.arange(len(areas)))
            ax_train.set_yticks(np.arange(len(areas)))
            ax_train.set_xticklabels(areas, rotation=90, fontsize=8)
            ax_train.set_yticklabels(areas, fontsize=8)
            # Show all ticks and label them
            plt.setp(ax_train.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
            # Add grid to separate areas visually
            ax_train.set_xticks(np.arange(len(areas) + 1) - .5, minor=True)
            ax_train.set_yticks(np.arange(len(areas) + 1) - .5, minor=True)
            # ax_train.grid(which="minor", color="w", linestyle='-', linewidth=1)

            im2 = ax_test.imshow(sim_correlation, cmap=cmap, vmin=-1, vmax=1)
            plt.colorbar(im2, shrink=0.6)
            ax_test.set_title(f'Simulated Correlation Matrix (similarity = {sim:.4f})')
            # Add area labels
            ax_test.set_xticks(np.arange(len(areas)))
            ax_test.set_yticks(np.arange(len(areas)))
            ax_test.set_xticklabels(areas, rotation=90, fontsize=8)
            ax_test.set_yticklabels(areas, fontsize=8)
            # Show all ticks and label them
            plt.setp(ax_test.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
            # Add grid to separate areas visually
            ax_test.set_xticks(np.arange(len(areas) + 1) - .5, minor=True)
            ax_test.set_yticks(np.arange(len(areas) + 1) - .5, minor=True)
            # ax_test.grid(which="minor", color="w", linestyle='-', linewidth=1)

            # Add these lines after setting tick labels
            ax_train.tick_params(axis='both', which='major', labelsize=4, width=0.5, length=6)
            ax_test.tick_params(axis='both', which='major', labelsize=4, width=0.5, length=6)

        else:
            im1 = ax_train.imshow(exp_correlation, cmap=cmap, vmin=-1, vmax=1)
            ax_train.set_title('Experimental Correlation Matrix')
            ax_train.axis('off')

            im2 = ax_test.imshow(sim_correlation, cmap=cmap, vmin=-1, vmax=1)
            plt.colorbar(im2, shrink=0.6)
            ax_test.set_title(f'Simulated Correlation Matrix (similarity = {sim:.4f})')
            ax_test.axis('off')

    seaborn.set_theme(font_scale=1.0, style=None)
    fig, gs = braintools.visualize.get_figure(2, 2, 5, 6)
    show(fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), 'train')
    show(fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), 'test')
    plt.suptitle(f'{args.neural_activity_id} {args.flywire_version}')

    if savefig:
        os.makedirs(os.path.join(filepath, 'analysis'), exist_ok=True)
        plt.savefig(os.path.join(filepath, 'analysis', 'correlation-matrix.pdf'), transparent=True, dpi=500)
    else:
        plt.show()
    plt.close()


def _compare_correlation_matrix_v2(
    filepath: str,
    show_ticks: bool = False,
    savefig: bool = False,
    training: bool = True,
):
    print(filepath)
    args = FilePath.from_filepath(filepath)
    settings = read_setting(filepath)

    if training:
        simulated_rates = np.load(os.path.join(filepath, 'neuropil_fr_predictions.npy')).T
    else:
        simulated_rates = np.load(os.path.join(filepath, 'neuropil_fr_predictions_untrained.npy')).T
    data = np.load(f'./data/spike_rates/ito_{args.neural_activity_id}_spike_rate.npz')
    dff = data['dff'][1:]
    rates = data['rates'][1:, 1:] * args.neural_activity_max_fr / u.Hz
    areas = data['areas'][1:]
    n_train, n_test = split_train_test(rates.shape[1], args.split, settings['batch_size'])
    print('n_train:', n_train)
    print('n_test:', n_test)

    cmap = 'coolwarm'
    seaborn.set_theme(font_scale=1.0, style=None)
    fig, gs = braintools.visualize.get_figure(1, 3, 5, 6)

    ax_exp = fig.add_subplot(gs[0, 0])
    ax_train = fig.add_subplot(gs[0, 1])
    ax_test = fig.add_subplot(gs[0, 2])

    _train_rates = rates[:, :n_train]
    _simulated_train_rates = simulated_rates[:, :n_train]
    _test_rates = rates[:, n_train:]
    _simulated_test_rates = simulated_rates[:, n_train:]

    exp_correlation = np.asarray(np.corrcoef(rates))
    sim_train_correlation = np.asarray(np.corrcoef(_simulated_train_rates))
    sim_test_correlation = np.asarray(np.corrcoef(_simulated_test_rates))

    train_corr = np.corrcoef(exp_correlation.flatten(), sim_train_correlation.flatten())[0, 1]
    test_corr = np.corrcoef(exp_correlation.flatten(), sim_test_correlation.flatten())[0, 1]
    print(f"Training correlation between correlation matrices: {train_corr:.4f}")
    print(f"Testing correlation between correlation matrices: {test_corr:.4f}")

    np.fill_diagonal(exp_correlation, np.nan)
    np.fill_diagonal(sim_train_correlation, np.nan)
    np.fill_diagonal(sim_test_correlation, np.nan)

    if show_ticks:
        im1 = ax_exp.imshow(exp_correlation, cmap=cmap, vmin=-1, vmax=1)
        ax_exp.set_title('Experimental Correlation Matrix')
        # Add area labels
        ax_exp.set_xticks(np.arange(len(areas)))
        ax_exp.set_yticks(np.arange(len(areas)))
        ax_exp.set_xticklabels(areas, rotation=90, fontsize=8)
        ax_exp.set_yticklabels(areas, fontsize=8)
        # Show all ticks and label them
        plt.setp(ax_exp.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
        # Add grid to separate areas visually
        ax_exp.set_xticks(np.arange(len(areas) + 1) - .5, minor=True)
        ax_exp.set_yticks(np.arange(len(areas) + 1) - .5, minor=True)
        # ax_train.grid(which="minor", color="w", linestyle='-', linewidth=1)

        im2 = ax_train.imshow(sim_train_correlation, cmap=cmap, vmin=-1, vmax=1)
        # plt.colorbar(im2, shrink=0.6)
        ax_train.set_title(f'Simulated Correlation Matrix (similarity = {train_corr:.4f})')
        # Add area labels
        ax_train.set_xticks(np.arange(len(areas)))
        ax_train.set_yticks(np.arange(len(areas)))
        ax_train.set_xticklabels(areas, rotation=90, fontsize=8)
        ax_train.set_yticklabels(areas, fontsize=8)
        # Show all ticks and label them
        plt.setp(ax_train.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
        # Add grid to separate areas visually
        ax_train.set_xticks(np.arange(len(areas) + 1) - .5, minor=True)
        ax_train.set_yticks(np.arange(len(areas) + 1) - .5, minor=True)

        im3 = ax_test.imshow(sim_test_correlation, cmap=cmap, vmin=-1, vmax=1)
        plt.colorbar(im3, shrink=0.6)
        ax_test.set_title(f'Simulated Correlation Matrix (similarity = {test_corr:.4f})')
        # Add area labels
        ax_test.set_xticks(np.arange(len(areas)))
        ax_test.set_yticks(np.arange(len(areas)))
        ax_test.set_xticklabels(areas, rotation=90, fontsize=8)
        ax_test.set_yticklabels(areas, fontsize=8)
        # Show all ticks and label them
        plt.setp(ax_test.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
        # Add grid to separate areas visually
        ax_test.set_xticks(np.arange(len(areas) + 1) - .5, minor=True)
        ax_test.set_yticks(np.arange(len(areas) + 1) - .5, minor=True)

        # Add these lines after setting tick labels
        ax_exp.tick_params(axis='both', which='major', labelsize=4, width=0.5, length=6)
        ax_train.tick_params(axis='both', which='major', labelsize=4, width=0.5, length=6)
        ax_test.tick_params(axis='both', which='major', labelsize=4, width=0.5, length=6)

    else:
        im1 = ax_exp.imshow(exp_correlation, cmap=cmap, vmin=-1, vmax=1)
        ax_exp.set_title('Experimental Correlation Matrix')
        ax_exp.axis('off')

        im2 = ax_train.imshow(sim_train_correlation, cmap=cmap, vmin=-1, vmax=1)
        # plt.colorbar(im2, shrink=0.6)
        ax_train.set_title(f'Simulated Correlation Matrix (similarity = {train_corr:.4f})')
        ax_train.axis('off')

        im3 = ax_test.imshow(sim_test_correlation, cmap=cmap, vmin=-1, vmax=1)
        ax_test.set_title(f'Simulated Correlation Matrix (similarity = {test_corr:.4f})')
        ax_test.axis('off')
        plt.colorbar(im3, shrink=0.6)
        # plt.colorbar(im3, shrink=0.6, location="bottom")

    plt.suptitle(f'{args.neural_activity_id} {args.flywire_version}')
    if savefig:
        os.makedirs(os.path.join(filepath, 'analysis'), exist_ok=True)
        plt.savefig(
            (
                os.path.join(filepath, 'analysis', 'correlation-matrix-trained.pdf')
                if training else
                os.path.join(filepath, 'analysis', 'correlation-matrix-untrained.pdf')
            ),
            transparent=True,
            dpi=500
        )
    else:
        plt.show()
    plt.close()


def compare_correlation_of_correlation_matrix():
    for filepath in glob.glob('results/*'):
        filepath = filepath.replace('\\', '/')
        try:
            # _compare_correlation_matrix(filepath, show_ticks=True, savefig=False)
            # _compare_correlation_matrix(filepath, show_ticks=True, savefig=True)
            _compare_correlation_matrix_v2(filepath, show_ticks=False, savefig=False, training=True)
            _compare_correlation_matrix_v2(filepath, show_ticks=False, savefig=False, training=False)
        except:
            pass

    # -----
    # confirm the results in
    #       visualize_experimental_and_simulated_firing_rates()
    # -----

    # for filepath in [
    #     'results/630#2018-10-19_1#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-18-10-52-34',
    #     'results/630#2018-10-31_1#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-21-01-56-22',
    #     'results/630#2018-11-03_3#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-18-23-43-48',
    # ]:
    #     _compare_correlation_matrix(filepath, show_ticks=True)


def _visualize_loss(filepath: str, orient: str = 'h'):
    print(filepath)

    # Load the loss data
    with open(os.path.join(filepath, 'first-round-losses.txt'), 'r') as f:
        log_content = f.read()

    regex = r"epoch\s*=\s*(\d+).*, loss\s*=\s*(\d+\.\d+), bin acc\s*=\s*(\d+\.\d+), lr\s*=\s*(\d+\.\d+)"
    matches = re.findall(regex, log_content)

    pattern = r'epoch = (\d+), loss = ([\d.]+), bin acc = ([\d.]+), lr = ([\d.]+)'
    matches = re.findall(pattern, log_content)

    extracted_data = []
    for match in matches:
        epoch, loss, bin_acc, lr = match
        extracted_data.append(
            {
                'epoch': int(epoch),
                'loss': float(loss),
                'bin_acc': float(bin_acc),
                'lr': float(lr)
            }
        )
    epoches = [d['epoch'] for d in extracted_data]
    losses = [d['loss'] for d in extracted_data]
    accs = [d['bin_acc'] for d in extracted_data]

    seaborn.set_theme(font_scale=1.2, style='ticks')

    if orient == 'v':

        fig, gs = braintools.visualize.get_figure(2, 1, 2, 3)
        ax = fig.add_subplot(gs[0])
        ax.plot(epoches[1:], losses[1:])
        ax.set_ylabel('Loss')
        plt.xticks([])
        seaborn.despine()
        ax.set_xlim(-1, 51)

        ax = fig.add_subplot(gs[1])
        ax.plot(epoches, accs)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Bin Accuracy')
        seaborn.despine()
        ax.set_xlim(-1, 51)
        ax.set_xticks([0, 10, 20, 30, 40, 50])

    else:
        fig, gs = braintools.visualize.get_figure(1, 2, 2, 3)
        ax = fig.add_subplot(gs[0])
        ax.plot(epoches[1:], losses[1:])
        ax.set_ylabel('Loss')
        seaborn.despine()
        ax.set_xlim(-1, 51)
        ax.set_xticks([0, 10, 20, 30, 40, 50])
        ax.set_xlabel('Epoch')

        ax = fig.add_subplot(gs[1])
        ax.plot(epoches, accs)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Bin Accuracy')
        seaborn.despine()
        ax.set_xlim(-1, 51)
        ax.set_xticks([0, 10, 20, 30, 40, 50])

    os.makedirs(os.path.join(filepath, 'analysis'), exist_ok=True)
    plt.savefig(os.path.join(filepath, 'analysis', 'loss-acc.pdf'), transparent=True)

    plt.close(fig)
    return extracted_data


def visualize_training_loss_and_accuracy():
    # for filepath in [
    #     # 优：
    #     'results/630#2018-10-19_1#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-18-10-52-34',
    #     'results/630#2018-10-31_1#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-21-01-56-22',
    #     'results/630#2018-11-03_3#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-18-23-43-48',
    #
    #     # 次优：
    #     'results/630#2018-12-14_1#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-22-05-05-42',
    #     'results/630#2018-11-03_5#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-20-09-41-05',
    #     'results/630#2018-12-14_3#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-21-16-35-58',
    # ]:
    #     _visualize_loss(filepath)

    for filepath in glob.glob('results/*'):
        filepath = filepath.replace('\\', '/')
        _visualize_loss(filepath)


class WeightAnalysis:
    def __init__(self, filepath):
        self.filepath = filepath

        params = braintools.file.msgpack_load(os.path.join(filepath, 'first-round-checkpoint.msgpack'))
        lora = params['interaction']['lora']['weight_op']
        B = lora['B']
        A = lora['A']['mantissa']
        self.weights = B @ A

    def analyze_weights(self):
        self._hierarchical_clustering()
        self._pca_visualization()
        self._tsne_visualization()
        self._heatmap()

    def _hierarchical_clustering(self):
        print('hierarchical clustering of weights')
        # 计算层次聚类
        linked = linkage(self.weights, 'ward')  # ward方法最小化聚类内方差

        # 绘制树状图
        fig, gs = braintools.visualize.get_figure(1, 1, 10, 10)
        fig.add_subplot(gs[0, 0])
        dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Weight Index')
        plt.ylabel('Distance')
        plt.savefig(os.path.join(self.filepath, 'low_rank_weight_hierarchical_clustering.pdf'))
        plt.close()

    def _pca_visualization(self, n_components=2):
        print('PCA visualization')
        # 应用PCA降维
        pca = PCA(n_components=n_components)  # 降至2维以便可视化
        weights_pca = pca.fit_transform(self.weights)

        # 可视化PCA结果
        fig, gs = braintools.visualize.get_figure(1, 1, 10, 10)
        fig.add_subplot(gs[0, 0])
        plt.scatter(weights_pca[:, 0], weights_pca[:, 1])
        plt.title('PCA of Weight Matrix')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.grid(True)
        plt.savefig(os.path.join(self.filepath, 'low_rank_weight_pca.pdf'))
        plt.close()

        # 查看每个主成分的解释方差比例
        print("Explained variance ratio:", pca.explained_variance_ratio_)

    def _tsne_visualization(self, n_components=2):
        print('t-SNE visualization')
        # 应用t-SNE
        tsne = TSNE(n_components=n_components, random_state=0)
        weights_tsne = tsne.fit_transform(self.weights)

        # 可视化t-SNE结果
        fig, gs = braintools.visualize.get_figure(1, 1, 10, 10)
        fig.add_subplot(gs[0, 0])
        plt.scatter(weights_tsne[:, 0], weights_tsne[:, 1])
        plt.title('t-SNE visualization of weights')
        plt.xlabel('t-SNE dimension 1')
        plt.ylabel('t-SNE dimension 2')
        plt.savefig(os.path.join(self.filepath, 'low_rank_weight_tSNE.pdf'))
        plt.close()

    def _heatmap(self):
        print('heatmap of weights')
        fig, gs = braintools.visualize.get_figure(1, 1, 10, 10)
        fig.add_subplot(gs[0, 0])
        seaborn.heatmap(self.weights, cmap='viridis')
        plt.title('Weight Matrix Heatmap')
        plt.savefig(os.path.join(self.filepath, 'low_rank_weight_heatmap.pdf'))
        plt.close()


def weight_analysis():
    for filepath in [

        # 'results/630#2017-10-30_1#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.3#0.6##2025-04-17-10-37-21',
        # 'results/630#2018-10-19_1#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-18-10-52-34',
        # 'results/630#2017-11-08_1#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.5#0.6##2025-04-15-17-50-48',
        # 'results/630#2017-10-30_1#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.25#0.6##2025-04-15-17-50-26',
        # 'results/630#2017-10-30_1#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.5#0.6##2025-04-15-17-50-22',
        # 'results/630#2017-10-26_1#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.5#0.6##2025-04-15-17-50-54',

        'results/630#2018-11-03_3#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-18-23-43-48',
        'results/630#2018-11-03_4#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-19-16-42-16',
        'results/630#2018-11-03_5#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-20-09-41-05',
        'results/630#2018-12-12_2#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-20-23-29-39',
        'results/630#2018-12-14_3#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-21-16-35-58',

        'results/630#2018-12-12_3#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-20-22-59-28',
        'results/630#2018-12-12_4#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-21-19-14-11',
        'results/630#2018-12-14_1#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-22-05-05-42',

    ]:
        try:
            WeightAnalysis(filepath).analyze_weights()
        except Exception as e:
            print(f"Error processing {filepath}: {e}")


if __name__ == '__main__':
    pass

    # compare_area_correlation()

    # compare_are_correlation_trained_untrained()

    # visualize_low_rank_connectivity_matrix()

    # visualize_experimental_and_simulated_firing_rates()

    # visualize_training_loss_and_accuracy()
    # visualize_experimental_trained_untrained_firing_rates()

    compare_correlation_of_correlation_matrix()

    # compare_connectivity_matrix()

    # weight_analysis()

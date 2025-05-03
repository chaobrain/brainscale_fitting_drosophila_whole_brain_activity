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


import os.path

import braintools
import brainunit as u
import jax
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from utils import FilePath, split_train_test, read_setting


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
    plt.savefig(os.path.join(filepath, 'low_rank_matrix.pdf'))
    plt.close()


def compare_connectivity_matrix():
    for filepath in [
        'results/630#2017-10-30_1#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.3#0.6##2025-04-17-10-37-21',
        'results/630#2018-10-19_1#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-18-10-52-34',
        'results/630#2017-11-08_1#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.5#0.6##2025-04-15-17-50-48',
        'results/630#2017-10-30_1#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.25#0.6##2025-04-15-17-50-26',
        'results/630#2017-10-30_1#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.5#0.6##2025-04-15-17-50-22',
        'results/630#2017-10-26_1#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.5#0.6##2025-04-15-17-50-54',
    ]:
        try:
            _visualize_low_rank_connectivity(filepath)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")


def compare_area_correlation(split: str = 'train'):
    filepath = 'results/630#2017-10-26_1#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.5#0.6##2025-04-15-17-50-54'
    filepath = 'results/630#2017-10-30_1#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.5#0.6##2025-04-15-17-50-22'
    filepath = 'results/630#2017-10-30_1#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.25#0.6##2025-04-15-17-50-26'
    filepath = 'results/630#2017-10-26_1#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.3#0.8##2025-04-17-10-38-17'
    filepath = 'results/630#2017-10-26_1#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.25#0.6##2025-04-15-17-40-29'
    filepath = 'results/630#2017-10-30_1#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.3#0.6##2025-04-17-10-37-21'
    filepath = 'results/630#2017-10-30_1#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.5#0.6##2025-04-15-17-42-03'
    filepath = 'results/630#2018-11-03_4#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-19-16-42-16'
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

    if split == 'train':
        _rates = rates[:, :n_train]
        _simulated_rates = simulated_rates[:, :n_train]
    elif split == 'test':
        _rates = rates[:, n_train:]
        _simulated_rates = simulated_rates[:, n_train:]
    else:
        raise ValueError(f"Invalid split: {split}")

    correlations = jax.vmap(lambda x, y: jax.numpy.corrcoef(x, y)[0, 1])(_simulated_rates, _rates)
    correlations = np.asarray(correlations)

    orient = 'h'
    orient = 'v'

    # Create barplot of correlations with area labels
    seaborn.set_theme(font_scale=0.9, style=None)

    if orient == 'h':
        fig, gs = braintools.visualize.get_figure(1, 1, 12, 3)
        ax = fig.add_subplot(gs[0, 0])
        ax = seaborn.barplot(x=correlations, y=areas, orient='h', ax=ax)
        plt.xlabel('Correlation Coefficient')
        seaborn.despine()
        plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    else:
        fig, gs = braintools.visualize.get_figure(1, 1, 4, 10)
        ax = fig.add_subplot(gs[0, 0])
        ax = seaborn.barplot(x=areas, y=correlations, orient='v', ax=ax)
        plt.ylabel('Correlation Coefficient')
        plt.xticks(rotation=90)
        seaborn.despine()
        plt.title('Correlation of Simulated and Experimental Firing Rates')

    # plt.savefig(f'area_correlations-{orient}.svg', transparent=True, dpi=500)
    plt.show()


def _compare_correlation_matrix(filepath: str, show_ticks: bool = False):
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
    plt.show()


def compare_correlation_of_correlation_matrix():
    filepath = 'results/630#2017-10-26_1#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.5#0.6##2025-04-15-17-50-54'  # 0.9571/0.6522
    # filepath = 'results/630#2017-10-30_1#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.5#0.6##2025-04-15-17-50-22'  # 0.8893/0.5057
    # filepath = 'results/630#2017-10-30_1#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.25#0.6##2025-04-15-17-50-26'  # 0.9308/0.5225
    filepath = 'results/630#2017-10-26_1#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.3#0.8##2025-04-17-10-38-17'  # 0.9163/0.7520
    # filepath = 'results/630#2017-10-26_1#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.25#0.6##2025-04-15-17-40-29'  # 0.8324/0.6524
    # filepath = 'results/630#2017-10-30_1#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.3#0.6##2025-04-17-10-37-21'  # 0.9103/0.5246
    # filepath = 'results/630#2017-10-30_1#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.5#0.6##2025-04-15-17-42-03'  # 0.6549/0.4835
    filepath = 'results/630#2018-11-03_3#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-18-23-43-48'  # 0.9368/0.5607
    filepath = 'results/630#2018-11-03_4#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-19-16-42-16'
    filepath = 'results/630#2018-10-19_2#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-18-23-43-39'

    for filepath in [
        # 'results/630#2018-11-03_3#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-18-23-43-48',
        # 'results/630#2018-11-03_4#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-19-16-42-16',
        # 'results/630#2018-11-03_5#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-20-09-41-05',
        # 'results/630#2018-12-12_2#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-20-23-29-39',
        # 'results/630#2018-12-14_3#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-21-16-35-58',

        'results/630#2018-12-12_3#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-20-22-59-28',
        'results/630#2018-12-12_4#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-21-19-14-11',
        'results/630#2018-12-14_1#100.0#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#0.4#0.8##2025-04-22-05-05-42',

    ]:
        _compare_correlation_matrix(filepath, show_ticks=True)


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

    # compare_area_correlation('train')
    # compare_area_correlation('test')

    compare_correlation_of_correlation_matrix()

    # compare_connectivity_matrix()
    # weight_analysis()

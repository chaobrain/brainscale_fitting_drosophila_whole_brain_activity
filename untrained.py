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

from args import get_parser
settings = get_parser()

import brainstate
import brainunit as u
import jax
import numpy as np

from drosophila_activity_analysis import _visualize_firing_rates, pearson_corr
from models import DrosophilaRestingStateModel, FilePath, split_train_test, read_setting

brainstate.environ.set(dt=0.2 * u.ms)


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
        os.path.join(filepath, 'untrained_analysis', f'firing-rate-comparison-top{len(top_indices)}.pdf')
    )


# for filepath in glob.glob('results/*'):
#     filepath = filepath.replace('\\', '/')
drosophila = DrosophilaRestingStateModel(settings.filepath, load_checkpoint=False)
drosophila.f_predict(filename='neuropil_fr_predictions_untrained')
_visualize_experimental_and_simulated_firing_rates_v3(settings.filepath)


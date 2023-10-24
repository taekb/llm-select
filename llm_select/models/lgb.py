import numpy as np
import random
import copy

import lightgbm

from ray.air import session
from ray.tune.integration.lightgbm import TuneReportCheckpointCallback

DEFAULT_SEED = 42

def train(
    config, 
    datasets,
    num_boost_round=50, 
    tune=False,
    return_booster=False,
    seed=DEFAULT_SEED
):
    '''Training function for LightGBM, compatible with Ray Tune.'''

    # Seed the search algorithms and schedulers
    random.seed(seed)
    np.random.seed(seed)

    # Unpack datasets
    train_dataset, val_dataset, test_dataset = datasets
    D_train = lightgbm.Dataset(train_dataset.X, label=train_dataset.Y)
    D_val = lightgbm.Dataset(val_dataset.X, label=val_dataset.Y)
    D_test = lightgbm.Dataset(test_dataset.X, label=test_dataset.Y)
    
    # Saved as: <dir specified in run.config>/<ckpt_dir>/model.xgb
    callbacks = [lightgbm.early_stopping(stopping_rounds=10)]
    if tune:
        callbacks.append(TuneReportCheckpointCallback(filename='model.mdl'))

    params = copy.deepcopy(config)

    booster = lightgbm.train(
        params, 
        D_train, 
        valid_sets=[D_train, D_val, D_test], 
        valid_names=['train', 'val', 'test'],
        num_boost_round=num_boost_round,
        callbacks=callbacks
    )

    if return_booster:
        return booster
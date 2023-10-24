import numpy as np
import random

from sklearn.utils.class_weight import compute_class_weight

import xgboost

from ray.air import session
from ray.tune.integration.xgboost import TuneReportCheckpointCallback

DEFAULT_SEED = 42

def train(
    config, 
    datasets,
    num_boost_round=10, 
    tune=False,
    return_booster=False,
    seed=DEFAULT_SEED,
    verbose=False
):
    '''Training function for XGBoost, compatible with Ray Tune.'''

    # Seed the search algorithms and schedulers
    random.seed(seed)
    np.random.seed(seed)

    # Unpack datasets
    train_dataset, val_dataset, test_dataset = datasets
    D_train = xgboost.DMatrix(train_dataset.X, label=train_dataset.Y)
    D_val = xgboost.DMatrix(val_dataset.X, label=val_dataset.Y)
    D_test = xgboost.DMatrix(test_dataset.X, label=test_dataset.Y)

    # Compute class weight
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_dataset.Y),
        y=train_dataset.Y
    )
    pos_weight = class_weights[1] / class_weights[0]
    config['scale_pos_weight'] = pos_weight
    
    # Saved as: <dir specified in run.config>/<ckpt_dir>/model.xgb
    if tune:
        callbacks = [TuneReportCheckpointCallback(filename='model.xgb')]
    else:
        callbacks = None

    booster = xgboost.train(
        config, 
        D_train, 
        evals=[(D_train, 'train'), (D_val, 'val'), (D_test, 'test')], 
        num_boost_round=num_boost_round,
        verbose_eval=verbose,
        early_stopping_rounds=10,
        callbacks=callbacks
    )

    if return_booster:
        return booster
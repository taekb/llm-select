import os
import os.path as osp
import argparse
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
import shutil
import random
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

import torch
torch.set_default_dtype(torch.float32)

from ray import tune
from ray import air
from ray.air.integrations.wandb import WandbLoggerCallback

import warnings 
warnings.filterwarnings('ignore')

import datasets as datasets
import models.linear as linear
import models.lgb as lgb
import models.xgb as xgb
import models.mlp as mlp

# Default seed
DEFAULT_SEED = 42

DATASET_TO_CLASS = {
    'acs': datasets.ACSDataset,
    'mimic-icd': datasets.MIMICICDDataset
}

# W&B API key
with open('../config/wandb_api_key.txt', 'r') as fh:
    wandb_api_key = fh.read()

def load_dataset(
    datapred='mimic-icd/ckd', 
    split='train', 
    select='llm-ratio',
    ratio=1.,
    seed=DEFAULT_SEED, 
    llm_model='', 
    llm_decoding='', 
    context_config='',
    example_config='',
    normalize=True,
    split_train_val=True,
    flatten=False,
    L=24, # For MIMIC
    bin_size=4, # For MIMIC
    **kwargs
):
    '''Data loading function.'''

    dataset, pred = datapred.split('/')

    with_context = True if context_config == 'context' else False
    with_examples = True if example_config == 'examples' else False
    with_expls = True if example_config == 'expls' else False

    base_data_config = dict(
        select=select,
        split=split,
        llm_model=llm_model,
        llm_decoding=llm_decoding,
        with_context=with_context,
        with_examples=with_examples,
        with_expls=with_expls,
        ratio=ratio,
        threshold=None,
        seed=seed,
        stratify_split=True,
        split_train_val=split_train_val,
        normalize=normalize,
        return_tensor=True, # NOTE: Can probably remove the manual casting in mlp.py and lstm.py
        flatten=flatten,
        **kwargs
    )
    
    # ACS (Classification)
    if dataset == 'acs':
        return datasets.ACSDataset(pred=pred, **base_data_config)

    # MIMIC-IV ICD (Classification)
    elif dataset == 'mimic-icd':
        return datasets.MIMICICDDataset(
            pred=pred, 
            L=L, 
            bin_size=bin_size,
            **base_data_config
        )

'''
    Logistic/linear regression CV functions.

'''

def hpo_linear(
    args, 
    fs_method='llm-ratio:gpt-4-0613:greedy::', 
    ratio=1,
    schedule='ASHA', 
    max_num_epochs=20, 
    grace_period=1, 
    num_samples=30,
    gpus_per_trial=0.5,
    seed=DEFAULT_SEED,
    **kwargs
):
    '''Hyperparameter optimization for logistic/linear regression.'''

    # Parse dataset and prediction names
    dataset, pred = args.datapred.split('/')

    if 'llm' in fs_method:
        # Format: llm-ratio:{llm_model}:{llm_decoding}:{context}:{examples/expls}
        _, llm_model, llm_decoding, context_config, example_config = fs_method.split(':')
        select = 'llm-ratio'
    else:
        llm_model, llm_decoding, context_config, example_config = '', '', '', ''

        if 'mrmr' in fs_method:
            select = 'all'
        else:
            select = fs_method

    # Load datasets
    base_data_config = dict(
        datapred=args.datapred,
        select=select,
        ratio=ratio,
        llm_model=llm_model,
        llm_decoding=llm_decoding,
        context_config=context_config,
        example_config=example_config,
        flatten=True,
        seed=seed
    )
    train_dataset = load_dataset(split='train', **base_data_config)
    val_dataset = load_dataset(split='val', **base_data_config)
    test_dataset = load_dataset(split='test', **base_data_config)

    # Load the feature mask from LassoNet/gLASSO/MRMR
    if fs_method in ['glasso', 'lassonet', 'mrmr']:
        mask_path = f'./selection/{fs_method}/{dataset}/{pred}_mask_{ratio}_{seed}.pkl'
        with open(mask_path, 'rb') as fh:
            mask = pickle.load(fh)

        train_dataset.X = train_dataset.X[:,mask]
        val_dataset.X = val_dataset.X[:,mask]
        test_dataset.X = test_dataset.X[:,mask]

    dataset_list = [train_dataset, val_dataset, test_dataset]

    # Hyperparameter search space
    config = {
        'batch_size': tune.choice([256,512,1024]),
        'lr': tune.loguniform(1e-4,1e-2),
        'l2': tune.loguniform(1e-3,1)
    }

    # Evaluation and model selection metrics
    pred_type = DATASET_TO_CLASS[dataset].get_pred_type()
    
    if pred_type == 'classification':
        metric_columns = ['train_auroc', 'val_auroc']
        selection_metric = 'val_auroc'
        selection_mode = 'max'
    
    elif pred_type == 'regression':
        metric_columns = ['train_mae', 'val_mae']
        selection_metric = 'val_mae'
        selection_mode = 'min'

    # Set up scheduler
    if schedule == 'ASHA':
        scheduler = tune.schedulers.ASHAScheduler(
            time_attr='training_iteration',
            max_t=max_num_epochs,
            grace_period=grace_period,
            reduction_factor=2
        )
    else:
        scheduler = None

    # Command-line reporter
    reporter = tune.CLIReporter(metric_columns=metric_columns)

    # Seed the search algorithms and schedulers
    random.seed(seed)
    np.random.seed(seed)

    # Set up trial output directory
    if 'llm' in fs_method:
        exp_outdir = osp.join(args.outdir, f'{dataset}/{pred}/{llm_model}/{llm_decoding}')
        exp_outdir += '_context' if context_config == 'context' else ''
        exp_outdir += '_examples' if example_config == 'examples' else ''
        exp_outdir += '_expls' if example_config == 'expls' else ''
        exp_outdir += f'_{ratio}_{seed}'
        
        # W&B config
        group = f'linear_{llm_model}_{llm_decoding}'
        group += '_context' if context_config == 'context' else ''
        group += '_examples' if example_config == 'examples' else ''
        group += '_expls' if example_config == 'expls' else ''
        group += f'_{ratio}_{seed}'
    else:
        exp_outdir = osp.join(args.outdir, f'{dataset}/{pred}/{fs_method}_{ratio}_{seed}')
        group = f'linear_{fs_method}_{ratio}_{seed}'

    callbacks = [] if args.run_locally else [
        WandbLoggerCallback(
            project=f'{dataset}_{pred}',
            group=group,
            api_key=wandb_api_key
        )
    ]

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                linear.train,
                datasets=dataset_list,
                epochs=max_num_epochs,
                tune=True,
                seed=seed
            ),
            resources={'cpu': 4, 'gpu': gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            metric=selection_metric,
            mode=selection_mode,
            scheduler=scheduler,
            num_samples=num_samples
        ),
        run_config=air.RunConfig(
            local_dir=exp_outdir,
            name='linear',
            progress_reporter=reporter,
            verbose=2,
            checkpoint_config=air.CheckpointConfig(
                num_to_keep=1,
                checkpoint_score_attribute=selection_metric,
                checkpoint_score_order=selection_mode,
                checkpoint_at_end=False
            ),
            stop=({'training_iteration': max_num_epochs} if schedule is None else None),
            callbacks=callbacks
        ),
        param_space=config
    )
    hpo_results = tuner.fit()

    # Save HPO summary
    # Retrieve the best metrics recorded in each trial
    df = hpo_results.get_dataframe(filter_metric=selection_metric, filter_mode=selection_mode)
    df.to_csv(osp.join(exp_outdir, 'linear_hpo_summary.csv'), index=False, header=True)

    # Identify trial with best result based on validation metric from each trial
    best_result = hpo_results.get_best_result(metric=selection_metric, mode=selection_mode, scope='all')

    # Load and save best hyperparameter config
    best_config = best_result.config # Dictionary
    best_config_df = pd.DataFrame(best_config, index=[0])
    print(f'\nBest trial config:\n{best_config_df}\n')

    best_config_path = osp.join(exp_outdir, 'linear_best_config.csv')
    best_config_df.to_csv(best_config_path, index=False, header=True)

    # Best metrics achieved within best trial
    metrics_df = best_result.metrics_dataframe
    metrics_df.to_csv(osp.join(exp_outdir, 'linear_best_trial_metrics.csv'), index=False, header=True)
    best_metrics = metrics_df[metrics_df[selection_metric] == metrics_df[selection_metric].max()]

    metric_name = selection_metric.split('_')[-1]
    print(f'Best validation loss: {best_metrics["val_loss"].values[0]:.4f}')
    print(f'Best validation {metric_name}: {best_metrics[selection_metric].values[0]:.4f}')

    # Gather the test performance for best model
    test_loss = best_metrics['test_loss'].values[0]
    test_metric = best_metrics[f'test_{metric_name}'].values[0]
    print(f'Test loss: {test_loss:.4f}')
    print(f'Test {metric_name}: {test_metric:.4f}')

    # Remove all trained models except the best
    best_logdir = df[df['trial_id'] == best_metrics['trial_id'].values[0]]['logdir'].values[0] # Full absolute path
    trial_dirs = [f for f in os.listdir(osp.join(exp_outdir, 'linear')) if ('train' in f) and (f not in best_logdir)]
    for trial_dir in trial_dirs:
        try:
            shutil.rmtree(osp.join(exp_outdir, f'linear/{trial_dir}'))
        except:
            print(f'Removing trained model directory failed: {osp.join(exp_outdir, f"linear/{trial_dir}")}')

    return test_metric
    
'''
    LightGBM HPO functions.

'''

def hpo_lgb(
    args,
    fs_method='llm-ratio:gpt-4-0613:greedy::',
    ratio=1,
    schedule='ASHA', 
    num_boost_round=50, 
    grace_period=1, 
    num_samples=30,
    seed=DEFAULT_SEED,
    **kwargs
):
    '''Hyperparameter optimization for LightGBM.'''    

    # Parse dataset and prediction names
    dataset, pred = args.datapred.split('/')

    if 'llm' in fs_method:
        # Format: llm-ratio:{llm_model}:{llm_decoding}:{context}:{examples/expls}
        _, llm_model, llm_decoding, context_config, example_config = fs_method.split(':')
        select = 'llm-ratio'
    else:
        llm_model, llm_decoding, context_config, example_config = '', '', '', ''

        if 'mrmr' in fs_method:
            select = 'all'
        else:
            select = fs_method

    # Load datasets
    base_data_config = dict(
        datapred=args.datapred,
        select=select,
        ratio=ratio,
        llm_model=llm_model,
        llm_decoding=llm_decoding,
        context_config=context_config,
        example_config=example_config,
        flatten=True,
        seed=seed
    )
    train_dataset = load_dataset(split='train', **base_data_config)
    val_dataset = load_dataset(split='val', **base_data_config)
    test_dataset = load_dataset(split='test', **base_data_config)

    # Load the feature mask from LassoNet/gLASSO/MRMR
    if fs_method in ['glasso', 'lassonet', 'mrmr']:
        mask_path = f'./selection/{fs_method}/{dataset}/{pred}_mask_{ratio}_{seed}.pkl'
        with open(mask_path, 'rb') as fh:
            mask = pickle.load(fh)

        train_dataset.X = train_dataset.X[:,mask]
        val_dataset.X = val_dataset.X[:,mask]
        test_dataset.X = test_dataset.X[:,mask]

    dataset_list = [train_dataset, val_dataset, test_dataset]

    # Hyperparameter search space
    config = {
        'boosting_type': 'gbdt',
        'seed': seed,
        'n_jobs': 4,
        'num_leaves': tune.randint(20,60), # Default is 31
        'learning_rate': tune.uniform(1e-2,0.5), # Default is 0.1
        'subsample': tune.uniform(0.5,1),
        'lambda': tune.uniform(0,2),
        'min_child_weight': tune.loguniform(1e-3,1) # Default is 1e-3
    }

    # Evaluation and model selection metrics
    pred_type = DATASET_TO_CLASS[dataset].get_pred_type()

    if pred_type == 'classification':
        metric_columns = ['train-auc', 'val-auc']
        selection_metric = 'val-auc'
        selection_mode = 'max'
        config['objective'] = 'binary'
        config['metric'] = ['binary_logloss', 'auc']
        config['is_unbalance'] = True
    
    elif pred_type == 'regression':
        metric_columns = ['train-mae', 'val-mae']
        selection_metric = 'val-l1'
        selection_mode = 'min'
        config['objective'] = 'mean_squared_error'
        config['metric'] = ['l2', 'l1']

    # Set up scheduler
    if schedule == 'ASHA':
        scheduler = tune.schedulers.ASHAScheduler(
            time_attr='training_iteration',
            max_t=num_boost_round,
            grace_period=grace_period,
            reduction_factor=2
        )
    else:
        scheduler = None

    # Command-line reporter
    reporter = tune.CLIReporter(metric_columns=metric_columns)

    # Seed the search algorithms and schedulers
    random.seed(seed)
    np.random.seed(seed)

    # Set up trial output directory
    if 'llm' in fs_method:
        exp_outdir = osp.join(args.outdir, f'{dataset}/{pred}/{llm_model}/{llm_decoding}')
        exp_outdir += '_context' if context_config == 'context' else ''
        exp_outdir += '_examples' if example_config == 'examples' else ''
        exp_outdir += '_expls' if example_config == 'expls' else ''
        exp_outdir += f'_{ratio}_{seed}'

        # W&B config
        group = f'lgb_{llm_model}_{llm_decoding}'
        group += '_context' if context_config == 'context' else ''
        group += '_examples' if example_config == 'examples' else ''
        group += '_expls' if example_config == 'expls' else ''
        group += f'_{ratio}_{seed}'
    else:
        exp_outdir = osp.join(args.outdir, f'{dataset}/{pred}/{fs_method}_{ratio}_{seed}')
        group = f'lgb_{fs_method}_{ratio}_{seed}'

    callbacks = [] if args.run_locally else [
        WandbLoggerCallback(
            project=f'{dataset}_{pred}',
            group=group,
            api_key=wandb_api_key
        )
    ]

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                lgb.train,
                datasets=dataset_list,
                num_boost_round=num_boost_round,
                tune=True,
                seed=seed
            ),
            resources={'cpu': 4}
        ),
        tune_config=tune.TuneConfig(
            metric=selection_metric,
            mode=selection_mode,
            scheduler=scheduler,
            num_samples=num_samples
        ),
        run_config=air.RunConfig(
            local_dir=exp_outdir,
            name='lgb',
            progress_reporter=reporter,
            verbose=2,
            checkpoint_config=air.CheckpointConfig(
                num_to_keep=1,
                checkpoint_score_attribute=selection_metric,
                checkpoint_score_order=selection_mode,
                checkpoint_at_end=False
            ),
            stop=({'training_iteration': num_boost_round} if schedule is None else None),
            callbacks=callbacks
        ),
        param_space=config
    )
    hpo_results = tuner.fit()

    # Save HPO summary
    df = hpo_results.get_dataframe(filter_metric=selection_metric, filter_mode=selection_mode)
    df.to_csv(osp.join(exp_outdir, f'lgb_hpo_summary.csv'), index=False, header=True)

    # Identify trial with best result based on validation metric from each trial
    best_result = hpo_results.get_best_result(metric=selection_metric, mode=selection_mode, scope='all')

    # Load and save best hyperparameter config
    best_config = best_result.config # Dictionary
    best_config.pop('metric')
    best_config_df = pd.DataFrame(best_config, index=[0])
    print(f'\nBest trial config:\n{best_config_df}\n')

    best_config_path = osp.join(exp_outdir, 'lgb_best_config.csv')
    best_config_df.to_csv(best_config_path, index=False, header=True)

    # Best metrics achieved within best trial
    metrics_df = best_result.metrics_dataframe
    metrics_df.to_csv(osp.join(exp_outdir, 'lgb_best_trial_metrics.csv'), index=False, header=True)
    best_metrics = metrics_df[metrics_df[selection_metric] == metrics_df[selection_metric].max()]

    metric_name = selection_metric.split('-')[-1]
    print(f'Best validation {metric_name}: {best_metrics[selection_metric].values[0]:.4f}')

    # Gather the test performance for best model
    test_metric = best_metrics[f'test-{metric_name}'].values[0]
    print(f'Test {metric_name}: {test_metric:.4f}')

    # Remove all trained models except the best
    best_logdir = df[df['trial_id'] == best_metrics['trial_id'].values[0]]['logdir'].values[0] # Full absolute path
    trial_dirs = [f for f in os.listdir(osp.join(exp_outdir, 'lgb')) if ('train' in f) and (f not in best_logdir)]
    for trial_dir in trial_dirs:
        try:
            shutil.rmtree(osp.join(exp_outdir, f'lgb/{trial_dir}'))
        except:
            print(f'Removing trained model directory failed: {osp.join(exp_outdir, f"lgb/{trial_dir}")}')

    return test_metric

'''
    XGBoost HPO functions.

'''

def hpo_xgb(
    args,
    fs_method='llm-ratio:gpt-4-0613:greedy::',
    ratio=1,
    schedule='ASHA', 
    num_boost_round=50, 
    grace_period=1, 
    num_samples=30,
    seed=DEFAULT_SEED,
    **kwargs
):
    '''Hyperparameter optimization for XGBoost.'''    

    # Parse dataset and prediction names
    dataset, pred = args.datapred.split('/')

    if 'llm' in fs_method:
        # Format: llm-ratio:{llm_model}:{llm_decoding}:{context}:{examples/expls}
        _, llm_model, llm_decoding, context_config, example_config = fs_method.split(':')
        select = 'llm-ratio'
    else:
        llm_model, llm_decoding, context_config, example_config = '', '', '', ''

        if 'mrmr' in fs_method:
            select = 'all'
        else:
            select = fs_method

    # Load datasets
    base_data_config = dict(
        datapred=args.datapred,
        select=select,
        ratio=ratio,
        llm_model=llm_model,
        llm_decoding=llm_decoding,
        context_config=context_config,
        example_config=example_config,
        flatten=True,
        seed=seed
    )
    train_dataset = load_dataset(split='train', **base_data_config)
    val_dataset = load_dataset(split='val', **base_data_config)
    test_dataset = load_dataset(split='test', **base_data_config)

    # Load the feature mask from LassoNet/gLASSO/MRMR
    if fs_method in ['glasso', 'lassonet', 'mrmr']:
        mask_path = f'./selection/{fs_method}/{dataset}/{pred}_mask_{ratio}_{seed}.pkl'
        with open(mask_path, 'rb') as fh:
            mask = pickle.load(fh)

        train_dataset.X = train_dataset.X[:,mask]
        val_dataset.X = val_dataset.X[:,mask]
        test_dataset.X = test_dataset.X[:,mask]
        
    dataset_list = [train_dataset, val_dataset, test_dataset]

    # Hyperparameter search space
    config = {
        'booster': 'gbtree',
        'seed': seed,
        'n_jobs': 4,
        'max_depth': tune.randint(6,12),
        'learning_rate': tune.uniform(1e-2,0.5),
        'subsample': tune.uniform(0.5,1),
        'lambda': tune.uniform(0,2),
        'min_child_weight': tune.uniform(1e-3,2) # Default is 1
    }
    
    # Evaluation and model selection metrics
    pred_type = DATASET_TO_CLASS[dataset].get_pred_type()

    if pred_type == 'classification':
        metric_columns = ['train-auc', 'val-auc']
        selection_metric = 'val-auc'
        selection_mode = 'max'
        config['objective'] = 'binary:logistic'
        config['eval_metric'] = ['logloss', 'auc']
    
    elif pred_type == 'regression':
        metric_columns = ['train-mae', 'val-mae']
        selection_metric = 'val-mae'
        selection_mode = 'min'
        config['objective'] = 'reg:squarederror'
        config['eval_metric'] = ['rmse', 'mae']

    # Set up scheduler
    if schedule == 'ASHA':
        scheduler = tune.schedulers.ASHAScheduler(
            time_attr='training_iteration',
            max_t=num_boost_round,
            grace_period=grace_period,
            reduction_factor=2
        )
    else:
        scheduler = None

    # Command-line reporter
    reporter = tune.CLIReporter(metric_columns=metric_columns)

    # Seed the search algorithms and schedulers
    random.seed(seed)
    np.random.seed(seed)

    # Set up trial output directory
    if 'llm' in fs_method:
        exp_outdir = osp.join(args.outdir, f'{dataset}/{pred}/{llm_model}/{llm_decoding}')
        exp_outdir += '_context' if context_config == 'context' else ''
        exp_outdir += '_examples' if example_config == 'examples' else ''
        exp_outdir += '_expls' if example_config == 'expls' else ''
        exp_outdir += f'_{ratio}_{seed}'

        # W&B config
        group = f'xgb_{llm_model}_{llm_decoding}'
        group += '_context' if context_config == 'context' else ''
        group += '_examples' if example_config == 'examples' else ''
        group += '_expls' if example_config == 'expls' else ''
        group += f'_{ratio}_{seed}'
    else:
        exp_outdir = osp.join(args.outdir, f'{dataset}/{pred}/{fs_method}_{ratio}_{seed}')
        group = f'xgb_{fs_method}_{ratio}_{seed}'

    callbacks = [] if args.run_locally else [
        WandbLoggerCallback(
            project=f'{dataset}_{pred}',
            group=group,
            api_key=wandb_api_key
        )
    ]

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                xgb.train,
                datasets=dataset_list,
                num_boost_round=num_boost_round,
                tune=True,
                seed=seed
            ),
            resources={'cpu': 4}
        ),
        tune_config=tune.TuneConfig(
            metric=selection_metric,
            mode=selection_mode,
            scheduler=scheduler,
            num_samples=num_samples
        ),
        run_config=air.RunConfig(
            local_dir=exp_outdir,
            name='xgb',
            progress_reporter=reporter,
            verbose=2,
            checkpoint_config=air.CheckpointConfig(
                num_to_keep=1,
                checkpoint_score_attribute=selection_metric,
                checkpoint_score_order=selection_mode,
                checkpoint_at_end=False
            ),
            stop=({'training_iteration': num_boost_round} if schedule is None else None),
            callbacks=callbacks
        ),
        param_space=config
    )
    hpo_results = tuner.fit()

    # Save HPO summary
    df = hpo_results.get_dataframe(filter_metric=selection_metric, filter_mode=selection_mode)
    df.to_csv(osp.join(exp_outdir, f'xgb_hpo_summary.csv'), index=False, header=True)

    # Identify trial with best result based on validation metric from each trial
    best_result = hpo_results.get_best_result(metric=selection_metric, mode=selection_mode, scope='all')

    # Load and save best hyperparameter config
    best_config = best_result.config # Dictionary
    best_config.pop('eval_metric')
    best_config_df = pd.DataFrame(best_config, index=[0])
    print(f'\nBest trial config:\n{best_config_df}\n')

    best_config_path = osp.join(exp_outdir, 'xgb_best_config.csv')
    best_config_df.to_csv(best_config_path, index=False, header=True)

    # Best metrics achieved within best trial
    metrics_df = best_result.metrics_dataframe
    metrics_df.to_csv(osp.join(exp_outdir, 'xgb_best_trial_metrics.csv'), index=False, header=True)
    best_metrics = metrics_df[metrics_df[selection_metric] == metrics_df[selection_metric].max()]

    metric_name = selection_metric.split('-')[-1]
    print(f'Best validation {metric_name}: {best_metrics[selection_metric].values[0]:.4f}')

    # Gather the test performance for best model
    test_metric = best_metrics[f'test-{metric_name}'].values[0]
    print(f'Test {metric_name}: {test_metric:.4f}')

    # Remove all trained models except the best
    best_logdir = df[df['trial_id'] == best_metrics['trial_id'].values[0]]['logdir'].values[0] # Full absolute path
    trial_dirs = [f for f in os.listdir(osp.join(exp_outdir, 'xgb')) if ('train' in f) and (f not in best_logdir)]
    for trial_dir in trial_dirs:
        try:
            shutil.rmtree(osp.join(exp_outdir, f'xgb/{trial_dir}'))
        except:
            print(f'Removing trained model directory failed: {osp.join(exp_outdir, f"xgb/{trial_dir}")}')

    return test_metric

'''
    MLP HPO functions.

'''

def hpo_mlp(
    args, 
    fs_method='llm-ratio:gpt-4-0613:greedy::', 
    ratio=1,
    schedule='ASHA', 
    max_num_epochs=20, 
    grace_period=1, 
    num_samples=30,
    gpus_per_trial=0.5,
    seed=DEFAULT_SEED,
    **kwargs
):
    '''Hyperparameter optimization for MLP.'''

    # Parse dataset and prediction names
    dataset, pred = args.datapred.split('/')

    if 'llm' in fs_method:
        # Format: llm-ratio:{llm_model}:{llm_decoding}:{context}:{examples/expls}
        _, llm_model, llm_decoding, context_config, example_config = fs_method.split(':')
        select = 'llm-ratio'
    else:
        llm_model, llm_decoding, context_config, example_config = '', '', '', ''

        if 'mrmr' in fs_method:
            select = 'all'
        else:
            select = fs_method

    # Load datasets
    base_data_config = dict(
        datapred=args.datapred,
        select=select,
        ratio=ratio,
        llm_model=llm_model,
        llm_decoding=llm_decoding,
        context_config=context_config,
        example_config=example_config,
        flatten=True,
        seed=seed
    )
    train_dataset = load_dataset(split='train', **base_data_config)
    val_dataset = load_dataset(split='val', **base_data_config)
    test_dataset = load_dataset(split='test', **base_data_config)

    # Load the feature mask from LassoNet/gLASSO/MRMR
    if fs_method in ['glasso', 'lassonet', 'mrmr']:
        mask_path = f'./selection/{fs_method}/{dataset}/{pred}_mask_{ratio}_{seed}.pkl'
        with open(mask_path, 'rb') as fh:
            mask = pickle.load(fh)

        train_dataset.X = train_dataset.X[:,mask]
        val_dataset.X = val_dataset.X[:,mask]
        test_dataset.X = test_dataset.X[:,mask]

    dataset_list = [train_dataset, val_dataset, test_dataset]

    # Hyperparameter search space
    config = {
        'd_hidden': tune.randint(200,500),
        'num_layers': tune.choice([2,3,4]),
        'dropout': tune.uniform(0,0.5),
        'batch_size': tune.choice([256,512,1024]),
        'lr': tune.loguniform(1e-4,1e-2)
    }

    # Evaluation and model selection metrics
    pred_type = DATASET_TO_CLASS[dataset].get_pred_type()
    
    if pred_type == 'classification':
        metric_columns = ['train_auroc', 'val_auroc']
        selection_metric = 'val_auroc'
        selection_mode = 'max'
    
    elif pred_type == 'regression':
        metric_columns = ['train_mae', 'val_mae']
        selection_metric = 'val_mae'
        selection_mode = 'min'

    # Set up scheduler
    if schedule == 'ASHA':
        scheduler = tune.schedulers.ASHAScheduler(
            time_attr='training_iteration',
            max_t=max_num_epochs,
            grace_period=grace_period,
            reduction_factor=2
        )
    else:
        scheduler = None

    # Command-line reporter
    reporter = tune.CLIReporter(metric_columns=metric_columns)

    # Seed the search algorithms and schedulers
    random.seed(seed)
    np.random.seed(seed)

    # Set up trial output directory
    if 'llm' in fs_method:
        exp_outdir = osp.join(args.outdir, f'{dataset}/{pred}/{llm_model}/{llm_decoding}')
        exp_outdir += '_context' if context_config == 'context' else ''
        exp_outdir += '_examples' if example_config == 'examples' else ''
        exp_outdir += '_expls' if example_config == 'expls' else ''
        exp_outdir += f'_{ratio}_{seed}'
        
        # W&B config
        group = f'mlp_{llm_model}_{llm_decoding}'
        group += '_context' if context_config == 'context' else ''
        group += '_examples' if example_config == 'examples' else ''
        group += '_expls' if example_config == 'expls' else ''
        group += f'_{ratio}_{seed}'
    else:
        exp_outdir = osp.join(args.outdir, f'{dataset}/{pred}/{fs_method}_{ratio}_{seed}')
        group = f'mlp_{fs_method}_{ratio}_{seed}'

    callbacks = [] if args.run_locally else [
        WandbLoggerCallback(
            project=f'{dataset}_{pred}',
            group=group,
            api_key=wandb_api_key
        )
    ]

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                mlp.train,
                datasets=dataset_list,
                epochs=max_num_epochs,
                tune=True,
                seed=seed
            ),
            resources={'cpu': 4, 'gpu': gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            metric=selection_metric,
            mode=selection_mode,
            scheduler=scheduler,
            num_samples=num_samples
        ),
        run_config=air.RunConfig(
            local_dir=exp_outdir,
            name='mlp',
            progress_reporter=reporter,
            verbose=2,
            checkpoint_config=air.CheckpointConfig(
                num_to_keep=1,
                checkpoint_score_attribute=selection_metric,
                checkpoint_score_order=selection_mode,
                checkpoint_at_end=False
            ),
            stop=({'training_iteration': max_num_epochs} if schedule is None else None),
            callbacks=callbacks
        ),
        param_space=config
    )
    hpo_results = tuner.fit()

    # Save HPO summary
    # Retrieve the best metrics recorded in each trial
    df = hpo_results.get_dataframe(filter_metric=selection_metric, filter_mode=selection_mode)
    df.to_csv(osp.join(exp_outdir, 'mlp_hpo_summary.csv'), index=False, header=True)

    # Identify trial with best result based on validation metric from each trial
    best_result = hpo_results.get_best_result(metric=selection_metric, mode=selection_mode, scope='all')

    # Load and save best hyperparameter config
    best_config = best_result.config # Dictionary
    best_config_df = pd.DataFrame(best_config, index=[0])
    print(f'\nBest trial config:\n{best_config_df}\n')

    best_config_path = osp.join(exp_outdir, 'mlp_best_config.csv')
    best_config_df.to_csv(best_config_path, index=False, header=True)

    # Best metrics achieved within best trial
    metrics_df = best_result.metrics_dataframe
    metrics_df.to_csv(osp.join(exp_outdir, 'mlp_best_trial_metrics.csv'), index=False, header=True)
    best_metrics = metrics_df[metrics_df[selection_metric] == metrics_df[selection_metric].max()]

    metric_name = selection_metric.split('_')[-1]
    print(f'Best validation loss: {best_metrics["val_loss"].values[0]:.4f}')
    print(f'Best validation {metric_name}: {best_metrics[selection_metric].values[0]:.4f}')

    # Gather the test performance for best model
    test_loss = best_metrics['test_loss'].values[0]
    test_metric = best_metrics[f'test_{metric_name}'].values[0]
    print(f'Test loss: {test_loss:.4f}')
    print(f'Test {metric_name}: {test_metric:.4f}')

    # Remove all trained models except the best
    best_logdir = df[df['trial_id'] == best_metrics['trial_id'].values[0]]['logdir'].values[0] # Full absolute path
    trial_dirs = [f for f in os.listdir(osp.join(exp_outdir, 'mlp')) if ('train' in f) and (f not in best_logdir)]
    for trial_dir in trial_dirs:
        try:
            shutil.rmtree(osp.join(exp_outdir, f'mlp/{trial_dir}'))
        except:
            print(f'Removing trained model directory failed: {osp.join(exp_outdir, f"mlp/{trial_dir}")}')

    return test_metric

def main(args):
    start_time = datetime.now()

    print(f'Running experiments for: {args.datapred}...')

    os.makedirs(osp.join(args.outdir, 'final_results'), exist_ok=True)
    results_path = osp.join(args.outdir, 'final_results/{}_{}_results'.format(*args.datapred.split('/')))
    results_path += f'_{args.exp_suffix}' if args.exp_suffix != '' else ''
    results_path += '.csv'
    results_dict = {'model': [], 'fs_method': [], 'ratio': [], 'seed': [], 'test_metric': []}

    for model in args.models:
        for fs_method in args.fs_methods:
            if fs_method == 'all':
                for seed in args.seeds:
                    print('\nRunning HPO:')
                    print(f'- Model: {model}')
                    print(f'- LLM Selection: {fs_method}')
                    print(f'- Seed: {seed}')

                    base_hpo_config = dict(
                        fs_method=fs_method, 
                        ratio=1,
                        seed=seed, 
                        gpus_per_trial=0.25
                    )

                    # Logistic/linear regression
                    if model =='linear':
                        test_metric = hpo_linear(args, max_num_epochs=15, num_samples=40, **base_hpo_config)
                    # LightGBM
                    elif model == 'lgb':
                        test_metric = hpo_lgb(args, num_boost_round=50, num_samples=40, **base_hpo_config)
                    # XGBoost
                    elif model == 'xgb':
                        test_metric = hpo_xgb(args, num_boost_round=50, num_samples=40, **base_hpo_config)
                    # MLP
                    elif model == 'mlp':
                        test_metric = hpo_mlp(args, max_num_epochs=15, num_samples=40, **base_hpo_config)

                    results_dict['model'].append(model)
                    results_dict['fs_method'].append(fs_method)
                    results_dict['ratio'].append(1)
                    results_dict['seed'].append(seed)
                    results_dict['test_metric'].append(test_metric)
            else:
                for ratio in args.ratios:
                    for seed in args.seeds:
                        print('\nRunning HPO:')
                        print(f'- Model: {model}')
                        print(f'- LLM Selection: {fs_method}')
                        print(f'- Ratio: {ratio}')
                        print(f'- Seed: {seed}')

                        base_hpo_config = dict(
                            fs_method=fs_method, 
                            ratio=ratio,
                            seed=seed, 
                            gpus_per_trial=0.25
                        )

                        # Logistic/linear regression
                        if model =='linear':
                            test_metric = hpo_linear(args, max_num_epochs=15, num_samples=40, **base_hpo_config)
                        # LightGBM
                        elif model == 'lgb':
                            test_metric = hpo_lgb(args, num_boost_round=50, num_samples=40, **base_hpo_config)
                        # XGBoost
                        elif model == 'xgb':
                            test_metric = hpo_xgb(args, num_boost_round=50, num_samples=40, **base_hpo_config)
                        # MLP
                        elif model == 'mlp':
                            test_metric = hpo_mlp(args, max_num_epochs=15, num_samples=40, **base_hpo_config)

                        results_dict['model'].append(model)
                        results_dict['fs_method'].append(fs_method)
                        results_dict['ratio'].append(ratio)
                        results_dict['seed'].append(seed)
                        results_dict['test_metric'].append(test_metric)

                    # Save intermediate results
                    if osp.exists(results_path):
                        prev_results_df = pd.read_csv(results_path)
                        results_df = pd.DataFrame(results_dict)
                        updated_results_df = pd.concat([prev_results_df, results_df], axis=0)
                        updated_results_df.drop_duplicates(inplace=True)
                        updated_results_df.to_csv(results_path, index=False, header=True)
                        print(f'\nResults added for {fs_method} (seed={seed}): {results_path}')
                    else:
                        results_df = pd.DataFrame(results_dict)
                        results_df.to_csv(results_path, index=False, header=True)
                        print(f'\nResults saved for {fs_method} (seed={seed}): {results_path}')
    
    # Save results
    if osp.exists(results_path):
        prev_results_df = pd.read_csv(results_path)
        results_df = pd.DataFrame(results_dict)
        updated_results_df = pd.concat([prev_results_df, results_df], axis=0)
        updated_results_df.drop_duplicates(inplace=True)
        updated_results_df.sort_values(by=['model', 'fs_method', 'ratio', 'seed'], inplace=True)
        updated_results_df.to_csv(results_path, index=False, header=True)
    else:
        results_df = pd.DataFrame(results_dict)
        results_df.to_csv(results_path, index=False, header=True)
    
    print(f'\nFinal results saved for {args.datapred}: {results_path}')
    print(f'[Elapsed] {str(datetime.now() - start_time)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapred', help='Dataset + prediction task pair', default='mimic-icd/ckd', type=str)
    parser.add_argument('--models', help='Models to evaluate', nargs='*', default=['mlp'], type=str)
    parser.add_argument('--fs_methods', help='Feature selection methods', nargs='*', default=['all'], type=str)
    parser.add_argument(
        '--ratios', 
        help='Feature selection ratios', 
        nargs='*', 
        default=[0.3],
        type=float
    )
    parser.add_argument('--seeds', help='Random seeds for reproducibility', nargs='*', default=[1,2,3,4,5], type=int)
    parser.add_argument('--outdir', help='Output directory', default='./results/benchmark', type=str)
    parser.add_argument('--exp_suffix', help='Optional suffix to add to results file', default='', type=str)
    parser.add_argument('--run_locally', help='Option to run Ray Tune locally without W&B', default=False, action='store_true')
    args = parser.parse_args()

    main(args)

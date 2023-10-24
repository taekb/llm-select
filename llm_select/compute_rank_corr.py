import os
import os.path as osp
import random
import copy
from collections import defaultdict
import yaml
import pandas as pd
from datetime import datetime
from scipy.stats import kendalltau, pearsonr, spearmanr
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

import argparse
from thefuzz import fuzz
from tqdm import tqdm
import numpy as np
import kneed
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance
from skfeature.function.similarity_based.fisher_score import fisher_score
from skfeature.function.similarity_based.lap_score import lap_score
from skfeature.utility.construct_W import construct_W
import xgboost as xgb
import lightgbm as lgb

import matplotlib.pyplot as plt
import shap

from ray import tune
from ray import air
from ray.air.integrations.wandb import WandbLoggerCallback

import datasets
import models

# Default seed
DEFAULT_SEED = 42

# Default directories
ABS_DIR = osp.dirname(osp.abspath(__file__))
DATA_DIR = osp.join(ABS_DIR, '../data')
CONFIG_DIR = osp.join(ABS_DIR, '../config')
PROMPT_DIR = osp.join(ABS_DIR, '../prompts')
PROMPT_OUTDIR = osp.join(ABS_DIR, '../prompt_outputs')

DATASET_TO_CLASS = {
    'cars': datasets.CarsDataset,
    'pima': datasets.PimaDataset,
    'give-me-credit': datasets.GiveMeCreditDataset,
    'compas': datasets.COMPASDataset,
    'miami-housing': datasets.MiamiHousingDataset,
    'wine': datasets.WineDataset,
    'credit-g': datasets.CreditGDataset,
    'diabetes': datasets.DiabetesDataset,
    'calhousing': datasets.CalHousingDataset,
    'bank': datasets.BankDataset
}

LLM_PLOTNAME = {
    'gpt-4-0613': 'GPT-4',
    'gpt-3.5-turbo': 'GPT-3.5',
    'llama-2-70b-chat': 'Llama-2-70B',
    'llama-2-13b-chat': 'Llama-2-13B',
    'llama-2-7b-chat': 'Llama-2-7B'
}

# W&B API key
with open(osp.join(CONFIG_DIR, 'wandb_api_key.txt'), 'r') as fh:
    wandb_api_key = fh.read()

def load_dataset(
    datapred,
    split, 
    select, 
    split_train_val=False,
    seed=DEFAULT_SEED, 
    normalize=False,
    **kwargs
):
    '''Data loading function for feature importance score comparison.'''

    dataset, pred = datapred.split('/')

    base_data_config = dict(
        select=select,
        split=split,
        split_train_val=split_train_val,
        seed=seed,
        stratify_split=True,
        normalize=normalize,
        flatten=True,
        return_tensor=False,
        **kwargs
    )

    # Used Car Price (Regression)
    if dataset == 'cars':
        return datasets.CarsDataset(**base_data_config)

    # Pima Indians Diabetes (Classification)
    elif dataset == 'pima':
        return datasets.PimaDataset(**base_data_config)
    
    # Give Me Credit (Classification)
    elif dataset == 'give-me-credit':
        return datasets.GiveMeCreditDataset(
            remove_outliers=True, 
            balance_classes=True,
            **base_data_config
        )

    # COMPAS Recidivism Risk (Classification)
    elif dataset == 'compas':
        return datasets.COMPASDataset(**base_data_config)

    # Miami Housing Price (Regression)
    elif dataset == 'miami-housing':
        return datasets.MiamiHousingDataset(**base_data_config)

    # Wine (Regression)
    elif dataset == 'wine':
        return datasets.WineDataset(**base_data_config)

    # German Credit Risk (Regression)
    elif dataset == 'credit-g':
        return datasets.CreditGDataset(**base_data_config)

    # Diabetes (Regression)
    elif dataset == 'diabetes':
        return datasets.DiabetesDataset(**base_data_config)
    
    # California Housing Price (Regression)
    elif dataset == 'calhousing':
        return datasets.CalHousingDataset(**base_data_config)
    
    # Bank Subscription (Classification)
    elif dataset == 'bank':
        return datasets.BankDataset(**base_data_config)

def run_kmeans(X, max_k=25, seed=DEFAULT_SEED):
    '''Runs K-means maximum number of times and returns approximately optimal number of clusters.'''

    centroids = {}
    inertias = []
    with tqdm(range(5,max_k+1), leave=False) as tkmeans:
        tkmeans.set_description('Running K-means')
        for k in tkmeans:
            kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init='auto', random_state=seed)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
            centroids[k] = kmeans.cluster_centers_

    # Find approximately optimal number of clusters
    optimal_k = kneed.KneeLocator(range(5,max_k+1), inertias, curve='convex', direction='decreasing').knee

    return centroids[optimal_k]

def compute_shap(
    datapred, 
    shap_model_type, 
    max_k=25, 
    schedule='ASHA', 
    outdir='./results/rank_corr/', 
    seed=DEFAULT_SEED,
    run_locally=False
):
    '''Computes the mean absolute Shapley values for a given dataset with the specified model.'''

    # Classification/regression
    dataset, pred = datapred.split('/')
    pred_type = DATASET_TO_CLASS[dataset].get_pred_type()

    # Linear model
    if shap_model_type == 'linear':
        # Load dataset
        train_dataset = load_dataset(datapred, split='train', select='all', seed=seed)
        test_dataset = load_dataset(datapred, split='test', select='all', seed=seed)
        normalizer = train_dataset.get_normalizer()

        # Logistic Regression
        if pred_type == 'classification':
            model = LogisticRegression(class_weight='balanced', penalty='l2', solver='lbfgs')

            # Hyperparameter search space
            param_space = {'clf__C': [0.1, 0.5, 1, 5, 10, 50, 100]}

            # Classification scoring
            scoring = 'roc_auc'
            refit = True

            # Stratified K-fold CV
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

        # Ridge regression
        elif pred_type == 'regression':
            model = Ridge()

            # Hyperparameter search space
            param_space = {'clf__alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]}

            # Regression scoring
            scoring = 'neg_mean_absolute_error'
            refit = True

            # Standard K-fold CV
            cv = 5

        # Pipeline
        pipe = Pipeline([('normalize', normalizer), ('clf', model)])

        # Grid Search CV
        grid = GridSearchCV(
            pipe, param_space, verbose=args.verbose, cv=cv, scoring=scoring, refit=refit, n_jobs=64
        )
        grid.fit(train_dataset.X, train_dataset.Y)
        
        # Compute Shapley values based on best model
        best_estimator = grid.best_estimator_
        best_normalizer = best_estimator.named_steps['normalize']
        X_train_norm = best_normalizer.transform(train_dataset.X)
        X_test_norm = best_normalizer.transform(test_dataset.X)

        # Numeric
        if isinstance(best_normalizer, StandardScaler):
            feature_names = test_dataset.columns
        
        # Categorical
        elif isinstance(best_normalizer, (OneHotEncoder, OrdinalEncoder)):
            feature_names = []
            cat_concepts = test_dataset.columns
            for i, concept in enumerate(cat_concepts):
                categories = best_normalizer.categories_
                feature_names += [f'{concept}_{cat}' for cat in categories[i]]
        
        # Numeric + Categorical
        elif isinstance(best_normalizer, ColumnTransformer):
            feature_names = list(np.array(test_dataset.columns)[best_normalizer.transformers[0][2]])
            cat_concepts = list(np.array(test_dataset.columns)[best_normalizer.transformers[1][2]])
            for i, concept in enumerate(cat_concepts):
                categories = best_normalizer.named_transformers_['onehot'].categories_
                feature_names += [f'{concept}_{cat}' for cat in categories[i]]
        
        else:
            raise RuntimeError('[Error] Invalid normalizer type encountered.')

        f = lambda x: (
            best_estimator.named_steps['clf'].predict_proba(x)[:,-1] 
            if pred_type == 'classification' 
            else best_estimator.named_steps['clf'].predict(x)
        )

        # Run K-means and use centroids as background marginalization data
        bg_data = run_kmeans(X_train_norm, max_k=max_k)
        #kmeans = shap.kmeans(X_train_norm, k=kmeans_k)
        explainer = shap.KernelExplainer(f, bg_data)
        shap_values = explainer.shap_values(X_test_norm)

        plt.figure()
        shap.summary_plot(shap_values, feature_names=feature_names)
        plt.savefig(osp.join(outdir, f'{dataset}/{pred}_shap_linear_{seed}.png'), bbox_inches='tight')
        
        # Group the Shapley values together for the categorical features
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0) # [D,]
        shap_dict = defaultdict(float)
        for i, col in enumerate(test_dataset.columns):
            feature_name = col.split('_')[0]
            shap_dict[feature_name] += mean_abs_shap[i]

    elif shap_model_type == 'xgb':
        # Load dataset
        train_dataset = load_dataset(
            datapred, split='train', select='all', split_train_val=True, normalize=True, seed=seed
        )
        val_dataset = load_dataset(
            datapred, split='val', select='all', split_train_val=True, normalize=True, seed=seed
        )
        test_dataset = load_dataset(
            datapred, split='test', select='all', split_train_val=True, normalize=True, seed=seed
        )
        dataset_list = [train_dataset, val_dataset, test_dataset]

        config = {
            'booster': 'gbtree',
            'seed': seed,
            'n_jobs': 4,
            'max_depth': tune.randint(6,10),
            'learning_rate': tune.uniform(1e-2,0.5),
            'subsample': tune.uniform(0.5,1),
            'lambda': tune.uniform(0,2),
            'min_child_weight': tune.choice([1,2,3])
        }

        if pred_type == 'classification':
            config['objective'] = 'binary:logistic'
            config['eval_metric'] = ['logloss', 'auc']

            class_weights = compute_class_weight(
                class_weight='balanced', classes=np.unique(train_dataset.Y), y=train_dataset.Y
            )
            pos_weight = class_weights[1] / class_weights[0]
            config['scale_pos_weight'] = pos_weight

            selection_metric = 'val-auc'
            selection_mode = 'max'

        elif pred_type == 'regression':
            config['objective'] = 'reg:squarederror'
            config['eval_metric'] = ['rmse', 'mae']
            selection_metric = 'val-mae'
            selection_mode = 'min'

        if schedule == 'ASHA':
            scheduler = tune.schedulers.ASHAScheduler(
                max_t=30,
                grace_period=3,
                reduction_factor=2
            )
        else:
            scheduler = None
        
        reporter = tune.CLIReporter(metric_columns=['train-logloss', 'train-auc', 'val-logloss', 'val-auc'])
        
        # Set up W&B callback
        callbacks = [] if run_locally else [
            WandbLoggerCallback(
                project='rank-corr',
                group=f'{dataset}_{pred}_xgb_{seed}',
                api_key=wandb_api_key
            )
        ]

        # Seed the search algorithms and schedulers
        random.seed(seed)
        np.random.seed(seed)

        # Set up trial output directory
        # e.g.: ./results/rank_corr/bank/subscription
        exp_outdir = osp.join(args.outdir, f'{dataset}/{pred}')

        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(
                    models.xgb.train,
                    datasets=dataset_list,
                    num_boost_round=100,
                    tune=True,
                    seed=seed
                ),
                resources={'cpu': 4}
            ),
            tune_config=tune.TuneConfig(
                metric=selection_metric,
                mode=selection_mode,
                scheduler=scheduler,
                num_samples=20
            ),
            run_config=air.RunConfig(
                local_dir=exp_outdir,
                name=f'xgb_{seed}', # ./results/rank_corr/bank/subscription
                progress_reporter=reporter,
                verbose=2,
                checkpoint_config=air.CheckpointConfig(
                    num_to_keep=1,
                    checkpoint_score_attribute=selection_metric,
                    checkpoint_score_order=selection_mode,
                    checkpoint_at_end=False
                ),
                stop=({'training_iteration': 30} if schedule is None else None),
                callbacks=callbacks
            ),
            param_space=config
        )
        hpo_results = tuner.fit()

        # Save hpo summary
        df = hpo_results.get_dataframe(filter_metric=selection_metric, filter_mode=selection_mode)
        df.to_csv(osp.join(exp_outdir, 'xgb_hpo_summary.csv'), index=False, header=True)

        # Identify trial with best result based on best validation AUPRC/AUROC within each trial
        best_result = hpo_results.get_best_result(metric=selection_metric, mode=selection_mode, scope='all')
        
        # Load and save best hyperparameter config
        best_config = best_result.config # Dictionary
        best_config.pop('eval_metric')
        best_config_df = pd.DataFrame(best_config, index=[0])
        print('\nBest trial config:')
        print(f'{best_config_df}\n')

        best_config_path = osp.join(exp_outdir, 'xgb_best_config.csv')
        best_config_df.to_csv(best_config_path, index=False, header=True)

        # Best metrics achieved within best trial
        metrics_df = best_result.metrics_dataframe
        metrics_df.to_csv(osp.join(exp_outdir, 'xgb_best_trial_metrics.csv'), index=False, header=True)
        best_metrics = metrics_df[metrics_df[selection_metric] == metrics_df[selection_metric].max()]

        metric_name = selection_metric.split('-')[-1]
        print(f'Best validation {metric_name}: {best_metrics[selection_metric].values[0]:.4f}')
        
        # Restore best training checkpoint
        best_model = xgb.Booster()
        best_ckpt = best_result.best_checkpoints[0][0] # Checkpoint object
        with best_ckpt.as_directory() as best_ckpt_dir:
            best_model.load_model(osp.join(best_ckpt_dir, 'model.xgb'))

        # Compute Shapley scores based on the best model
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(test_dataset.X) # [N,D]

        plt.figure()
        shap.summary_plot(shap_values, test_dataset.X, feature_names=test_dataset.columns)
        plt.savefig(osp.join(outdir, f'{dataset}/{pred}_shap_xgb_{seed}.png'), bbox_inches='tight')
        
        # Group the Shapley values together for the categorical features
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0) # [D,]
        shap_dict = defaultdict(float)
        for i, col in enumerate(test_dataset.columns):
            feature_name = col.split('_')[0]
            shap_dict[feature_name] += mean_abs_shap[i]

    elif shap_model_type == 'lgb':
        # Load dataset
        train_dataset = load_dataset(
            datapred, split='train', select='all', split_train_val=True, normalize=True, seed=seed
        )
        val_dataset = load_dataset(
            datapred, split='val', select='all', split_train_val=True, normalize=True, seed=seed
        )
        test_dataset = load_dataset(
            datapred, split='test', select='all', split_train_val=True, normalize=True, seed=seed
        )
        dataset_list = [train_dataset, val_dataset, test_dataset]

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

        if pred_type == 'classification':
            config['objective'] = 'binary'
            config['metric'] = ['binary_logloss', 'auc']
            config['is_unbalance'] = True

            metric_columns = ['train-auc', 'val-auc']
            selection_metric = 'val-auc'
            selection_mode = 'max'

        elif pred_type == 'regression':
            config['objective'] = 'mean_squared_error'
            config['metric'] = ['l2', 'l1']

            metric_columns = ['train-l1', 'val-l1']
            selection_metric = 'val-l1'
            selection_mode = 'min'

        if schedule == 'ASHA':
            scheduler = tune.schedulers.ASHAScheduler(
                max_t=30,
                grace_period=3,
                reduction_factor=2
            )
        else:
            scheduler = None
        
        reporter = tune.CLIReporter(metric_columns=metric_columns)
        
        # Set up W&B callback
        callbacks = [] if run_locally else [
            WandbLoggerCallback(
                project='rank-corr',
                group=f'{dataset}_{pred}_lgb_{seed}',
                api_key=wandb_api_key
            )
        ]

        # Seed the search algorithms and schedulers
        random.seed(seed)
        np.random.seed(seed)

        # Set up trial output directory
        # e.g.: ./results/rank_corr/bank/subscription
        exp_outdir = osp.join(args.outdir, f'{dataset}/{pred}')

        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(
                    models.lgb.train,
                    datasets=dataset_list,
                    num_boost_round=100,
                    tune=True,
                    seed=seed
                ),
                resources={'cpu': 4}
            ),
            tune_config=tune.TuneConfig(
                metric=selection_metric,
                mode=selection_mode,
                scheduler=scheduler,
                num_samples=20
            ),
            run_config=air.RunConfig(
                local_dir=exp_outdir,
                name=f'lgb_{seed}', # ./results/rank_corr/bank/subscription
                progress_reporter=reporter,
                verbose=2,
                checkpoint_config=air.CheckpointConfig(
                    num_to_keep=1,
                    checkpoint_score_attribute=selection_metric,
                    checkpoint_score_order=selection_mode,
                    checkpoint_at_end=False
                ),
                stop=({'training_iteration': 30} if schedule is None else None),
                callbacks=callbacks
            ),
            param_space=config
        )
        hpo_results = tuner.fit()

        # Save hpo summary
        df = hpo_results.get_dataframe(filter_metric=selection_metric, filter_mode=selection_mode)
        df.to_csv(osp.join(exp_outdir, 'lgb_hpo_summary.csv'), index=False, header=True)

        # Identify trial with best result based on best validation AUPRC/AUROC within each trial
        best_result = hpo_results.get_best_result(metric=selection_metric, mode=selection_mode, scope='all')
        
        # Load and save best hyperparameter config
        best_config = best_result.config # Dictionary
        best_config.pop('metric')
        best_config_df = pd.DataFrame(best_config, index=[0])
        print('\nBest trial config:')
        print(f'{best_config_df}\n')

        best_config_path = osp.join(exp_outdir, 'lgb_best_config.csv')
        best_config_df.to_csv(best_config_path, index=False, header=True)

        # Best metrics achieved within best trial
        metrics_df = best_result.metrics_dataframe
        metrics_df.to_csv(osp.join(exp_outdir, 'lgb_best_trial_metrics.csv'), index=False, header=True)
        best_metrics = metrics_df[metrics_df[selection_metric] == metrics_df[selection_metric].max()]

        metric_name = selection_metric.split('-')[-1]
        print(f'Best validation {metric_name}: {best_metrics[selection_metric].values[0]:.4f}')
        
        # Restore best training checkpoint
        best_ckpt = best_result.best_checkpoints[0][0] # Checkpoint object
        with best_ckpt.as_directory() as best_ckpt_dir:
            best_model = lgb.Booster(model_file=osp.join(best_ckpt_dir, 'model.mdl'))

        # Compute Shapley scores based on the best model
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(test_dataset.X) # [N,D]

        plt.figure()
        shap.summary_plot(shap_values, test_dataset.X, feature_names=test_dataset.columns)
        plt.savefig(osp.join(outdir, f'{dataset}/{pred}_shap_lgb_{seed}.png'), bbox_inches='tight')
        
        # Group the Shapley values together for the categorical features
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0) # [D,]
        shap_dict = defaultdict(float)
        for i, col in enumerate(test_dataset.columns):
            feature_name = col.split('_')[0]
            shap_dict[feature_name] += mean_abs_shap[i]

    return shap_dict

def compute_mean_shap(args, datapred, shap_model_type='xgb'):
    '''Computes the global Shapley values averaged over multiple seeds.'''

    shap_scores_list = defaultdict(list)
    for seed in args.seeds:
        shap_scores_seed = compute_shap(
            datapred, shap_model_type, max_k=args.max_k, seed=seed, run_locally=args.run_locally
        )
        for k in shap_scores_seed.keys():
            shap_scores_list[k].append(shap_scores_seed[k])
    
    shap_scores = {k: np.mean(v) for k,v in shap_scores_list.items()}

    return shap_scores

def compute_mi(datapred, seed=DEFAULT_SEED):
    '''Computes the mutual information between each feature and the target.'''

    dataset, _ = datapred.split('/')
    pred_type = DATASET_TO_CLASS[dataset].get_pred_type()

    mi_f = mutual_info_classif if pred_type == 'classification' else mutual_info_regression
    train_dataset = load_dataset(datapred, split='train', select='all', normalize=True, seed=seed)

    # Compute mutual information
    mi_dict = {}
    for i, feature_name in enumerate(train_dataset.columns):
        is_discrete = True if len(feature_name.split('_')) >= 2 else False
        mi_dict[feature_name] = mi_f(
            train_dataset.X[:,i:i+1], train_dataset.Y, discrete_features=is_discrete, random_state=seed
        )

    return mi_dict

def compute_pearson(datapred, seed=DEFAULT_SEED):
    '''
        Computes the absolute Pearson correlation coefficient between each feature and the target.

        Only applicable to regression datasets, as the target variable must be continuous.
    '''

    train_dataset = load_dataset(datapred, split='train', select='all', normalize=True, seed=seed)

    # Compute the absolute Pearson correlation
    pearson_dict = {}
    for i, feature_name in enumerate(train_dataset.columns):
        pearson_dict[feature_name] = np.abs(pearsonr(train_dataset.X[:,i], train_dataset.Y).statistic)

    return pearson_dict

def compute_spearman(datapred, seed=DEFAULT_SEED):
    '''
        Computes the absolute Spearman correlation coefficient between each feature and the target.
    
        Only applicable to regression datasets, as the target variable must be continuous.
    '''

    train_dataset = load_dataset(datapred, split='train', select='all', normalize=True, seed=seed)

    # Compute the absolute Pearson correlation
    spearman_dict = {}
    for i, feature_name in enumerate(train_dataset.columns):
        spearman_dict[feature_name] = np.abs(spearmanr(train_dataset.X[:,i], train_dataset.Y).statistic)

    return spearman_dict

def compute_fisher_scores(datapred, seed=DEFAULT_SEED, subsample_thresh=50000):
    '''
        Computes the Fisher score for each feature.

        Only applicable to classification datasets, as class membership information is required.
    '''

    # If dataset size is above threshold, subsample with stratified splitting
    # NOTE: This is only done since Fisher score computation can crash if dataset is too large
    # TODO: Double-check if this is really necessary
    train_dataset = load_dataset(datapred, split='train', select='all', normalize=True, seed=seed)
    X, Y = train_dataset.X, train_dataset.Y

    if train_dataset.X.shape[0] > subsample_thresh:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=subsample_thresh, random_state=seed)
        selected_idx, _ = next(sss.split(X, Y))
        X, Y = X[selected_idx], Y[selected_idx]

    # Compute Fisher score
    fisher_scores = fisher_score(X, Y)
    fisher_dict = {}
    for i, feature_name in enumerate(train_dataset.columns):
        fisher_dict[feature_name] = fisher_scores[i]

    return fisher_dict

def compute_inv_lap_scores(datapred, seed=DEFAULT_SEED):
    '''Computes the inverse Laplacian score for each feature.'''

    dataset = load_dataset(datapred, split='all', select='all', normalize=True, seed=seed)

    # Compute affinity matrix
    W = construct_W(dataset.X, metric='cosine', neighbor_mode='knn', weight_mode='cosine', k=5)
    lap_scores = lap_score(dataset.X, W=W)
    inv_lap_scores = 1 / lap_scores
    
    inv_lap_dict = {}
    for i, feature_name in enumerate(dataset.columns):
        inv_lap_dict[feature_name] = inv_lap_scores[i]

    return inv_lap_dict

def compute_perm_importance(datapred, seed=DEFAULT_SEED, n_repeats=30):
    '''Computes the permutation importance of each feature using a linear model.'''

    train_dataset = load_dataset(
        datapred, split='train', select='all', normalize=False, split_train_val=False, seed=seed
    )
    test_dataset = load_dataset(
        datapred, split='test', select='all', normalize=False, split_train_val=False, seed=seed
    )
    normalizer = train_dataset.get_normalizer()
    pred_type = train_dataset.get_pred_type()

    if pred_type == 'classification':
        model = LogisticRegression(class_weight='balanced')
        param_space = {'clf__C': [0.1, 0.5, 1, 5, 10, 50, 100]}
        scoring = 'roc_auc'
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    elif pred_type == 'regression':
        model = Ridge()
        param_space = {'clf__alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]}
        scoring = 'neg_mean_absolute_error'
        cv = 5

    # Run grid search
    pipe = Pipeline([('normalize', normalizer), ('clf', model)])
    grid = GridSearchCV(
        pipe, param_space, verbose=args.verbose, cv=cv, scoring=scoring, refit=True, n_jobs=64
    ).fit(train_dataset.X, train_dataset.Y)
    
    # Compute permutation feature importance with best model
    perm = permutation_importance(
        grid.best_estimator_.named_steps['clf'],
        grid.best_estimator_.named_steps['normalize'].transform(test_dataset.X),
        test_dataset.Y,
        n_repeats=n_repeats,
        random_state=seed,
        scoring=scoring,
        n_jobs=64
    )
    perm_scores = perm.importances_mean

    # Fetch column names from dummy dataset
    columns = load_dataset(datapred, split='test', select='all', normalize=True).columns

    perm_dict = {}
    for i, feature_name in enumerate(columns):
        perm_dict[feature_name] = perm_scores[i]

    return perm_dict

def compute_kendall_tau(s1, s2):
    '''Computes Kendall's tau between LLM scores and another set of feature importances.'''

    scores_1 = []
    scores_2 = []

    # NOTE: Assuming that s1 is always a set of concept-level scores
    # For handling case when feature importance in s2 is at the feature-level instead of the concept-level
    expand = len(s1) != len(s2)

    for k in s2.keys():
        if expand and len(k.split('_')) >= 2: # Check if categorical
            scores_1.append(s1[k.split('_')[0]])
        else:
            scores_1.append(s1[k])
        
        scores_2.append(s2[k])

    tau = kendalltau(scores_1, scores_2).statistic

    return tau

def main(args):
    start_time = datetime.now()
    os.makedirs(args.outdir, exist_ok=True)

    # Check if rank correlations have already been calculated
    rank_corr_path = osp.join(args.outdir, 'rank_corr')
    rank_corr_path += f'_{args.result_suffix}' if args.result_suffix != '' else ''
    rank_corr_path += '.csv'

    prev_exists = False
    if osp.exists(rank_corr_path):
        print('Loading existing rank correlation results.\n')
        rank_corr_df = pd.read_csv(rank_corr_path)
        prev_exists = True

    # Check if anything should be added
    # Columns: datapred, llm_model, fi_method, corr_metric
    rank_corr_dict = {
        'datapred': [],
        'llm_model': [],
        'fi_method': [],
        'kendall_tau': []
    }

    for datapred in args.datapreds:
        dataset, pred = datapred.split('/')

        for llm_model in args.llm_models:
            # Load LLM feature importance scores from greedy decoding
            greedy_outpath = osp.join(PROMPT_OUTDIR, f'{dataset}/{llm_model}/{pred}_greedy.yaml')
            with open(greedy_outpath, 'r') as fh:
                greedy_output = yaml.load(fh, Loader=yaml.Loader)
                greedy_scores = {k: v['score'] for k,v in greedy_output.items()}
                concepts = list(greedy_scores.keys())

            # Correlation with LLM scores from self-consistency decoding
            if 'consistent' in args.recompute or (not prev_exists or not (
                (rank_corr_df['datapred'] == datapred) & \
                (rank_corr_df['llm_model'] == llm_model) & \
                (rank_corr_df['fi_method'] == 'consistent')
            ).any()):
                consistent_outpath = osp.join(PROMPT_OUTDIR, f'{dataset}/{llm_model}/{pred}_consistent.yaml')
                with open(consistent_outpath, 'r') as fh:
                    consistent_output = yaml.load(fh, Loader=yaml.Loader)
                    consistent_scores = {k: v['score'] for k,v in consistent_output.items()}

                rank_corr_dict['datapred'].append(datapred)
                rank_corr_dict['llm_model'].append(llm_model)
                rank_corr_dict['fi_method'].append('consistent')
                tau_consistent = compute_kendall_tau(greedy_scores, consistent_scores)
                rank_corr_dict['kendall_tau'].append(tau_consistent)

                if args.verbose:
                    print(r'datapred={}, llm_model={}, fi_method=consistent, tau={}'.format(
                        datapred, llm_model, tau_consistent
                    ))

            # Correlation with LLM ranking
            if 'rank' in args.recompute or (not prev_exists or not (
                (rank_corr_df['datapred'] == datapred) & \
                (rank_corr_df['llm_model'] == llm_model) & \
                (rank_corr_df['fi_method'] == 'rank')
            ).any()):
                rank_outpath = osp.join(PROMPT_OUTDIR, f'{dataset}/{llm_model}/{pred}_rank.yaml')
                with open(rank_outpath, 'r') as fh:
                    rank_output = yaml.load(fh, Loader=yaml.Loader) # NOTE: Ranked list of concepts
                    
                    # Use fuzzy string matching to ensure correct concept string
                    fuzzy_matched = []
                    for c in rank_output:
                        fuzz_scores = list(map(lambda x: fuzz.ratio(c,x), concepts))
                        fuzzy_matched.append(concepts[np.argmax(fuzz_scores)])

                    # Remove duplicates if present
                    if len(set(fuzzy_matched)) < len(fuzzy_matched):
                        no_duplicates = []
                        observed = set()
                        for c in fuzzy_matched:
                            if c not in observed:
                                observed.add(c)
                                no_duplicates.append(c)
                        
                        rank_output = no_duplicates
                    else:
                        rank_output = fuzzy_matched

                    # Average if there are missing concepts (NOTE: Sometimes occurs for Llama-2-7B)
                    if len(rank_output) != len(concepts):
                        missing = list(set(concepts) - set(rank_output))
                        rank_kendall_taus = []

                        for seed in args.seeds:
                            missing_cp = copy.deepcopy(missing)
                            np.random.seed(seed)
                            np.random.shuffle(missing_cp)
                            sampled_rank_output = rank_output + missing_cp
                            sampled_rank_scores = {k: len(concepts)-i for i,k in enumerate(sampled_rank_output)}
                            rank_kendall_taus.append(compute_kendall_tau(greedy_scores, sampled_rank_scores))
                        
                        tau_rank = np.nanmean(rank_kendall_taus)
                    else:
                        rank_scores = {k: len(concepts)-i for i,k in enumerate(rank_output)}
                        tau_rank = compute_kendall_tau(greedy_scores, rank_scores)

                rank_corr_dict['datapred'].append(datapred)
                rank_corr_dict['llm_model'].append(llm_model)
                rank_corr_dict['fi_method'].append('rank')
                rank_corr_dict['kendall_tau'].append(tau_rank)

                if args.verbose:
                    print(r'datapred={}, llm_model={}, fi_method=rank, tau={}'.format(
                        datapred, llm_model, tau_rank
                    ))

            # Correlation with LLM sequential selection (Nonempty)
            if 'forward-seq' in args.recompute or (not prev_exists or not (
                (rank_corr_df['datapred'] == datapred) & \
                (rank_corr_df['llm_model'] == llm_model) & \
                (rank_corr_df['fi_method'] == 'forward-seq')
            ).any()):
                seq_rank_outpath = osp.join(ABS_DIR, f'results/linear_compare/{dataset}/{llm_model}_forward-seq.yaml')
                with open(seq_rank_outpath, 'r') as fh:
                    seq_rank_outputs = yaml.load(fh, Loader=yaml.Loader) # List of lists of ranks
                    
                    seq_rank_kendall_taus = []
                    for seq_rank_output in seq_rank_outputs:
                        seq_rank_scores = {k: len(concepts)-i for i,k in enumerate(seq_rank_output)}
                        seq_rank_kendall_taus.append(compute_kendall_tau(greedy_scores, seq_rank_scores))

                    tau_seq_rank = np.nanmean(seq_rank_kendall_taus)

                rank_corr_dict['datapred'].append(datapred)
                rank_corr_dict['llm_model'].append(llm_model)
                rank_corr_dict['fi_method'].append('forward-seq')
                rank_corr_dict['kendall_tau'].append(tau_seq_rank)

                if args.verbose:
                    print(r'datapred={}, llm_model={}, fi_method=forward-seq, tau={}'.format(
                        datapred, llm_model, tau_seq_rank
                    ))

            # Correlation with LLM sequential selection (Empty)
            if 'forward-seq-empty' in args.recompute or (not prev_exists or not (
                (rank_corr_df['datapred'] == datapred) & \
                (rank_corr_df['llm_model'] == llm_model) & \
                (rank_corr_df['fi_method'] == 'forward-seq-empty')
            ).any()):
                seq_rank_outpath = osp.join(ABS_DIR, f'results/linear_compare/{dataset}/{llm_model}_forward-seq-empty.yaml')
                with open(seq_rank_outpath, 'r') as fh:
                    seq_rank_outputs = yaml.load(fh, Loader=yaml.Loader) # List of lists of ranks
                    
                    seq_rank_kendall_taus = []
                    for seq_rank_output in seq_rank_outputs:
                        seq_rank_scores = {k: len(concepts)-i for i,k in enumerate(seq_rank_output)}
                        seq_rank_kendall_taus.append(compute_kendall_tau(greedy_scores, seq_rank_scores))

                    tau_seq_rank = np.nanmean(seq_rank_kendall_taus)

                rank_corr_dict['datapred'].append(datapred)
                rank_corr_dict['llm_model'].append(llm_model)
                rank_corr_dict['fi_method'].append('forward-seq-empty')
                rank_corr_dict['kendall_tau'].append(tau_seq_rank)

                if args.verbose:
                    print(r'datapred={}, llm_model={}, fi_method=forward-seq-empty, tau={}'.format(
                        datapred, llm_model, tau_seq_rank
                    ))
                    
            # Correlation with LLM scores from greedy decoding with range 0-10
            if 'range_1' in args.recompute or (not prev_exists or not (
                (rank_corr_df['datapred'] == datapred) & \
                (rank_corr_df['llm_model'] == llm_model) & \
                (rank_corr_df['fi_method'] == 'range_1')
            ).any()):
                range_outpath_1 = osp.join(PROMPT_OUTDIR, f'{dataset}/{llm_model}/{pred}_greedy_0.0_10.0.yaml')
                with open(range_outpath_1, 'r') as fh:
                    range_output_1 = yaml.load(fh, Loader=yaml.Loader) # Dictionary
                    range_scores_1 = {k: v['score'] for k,v in range_output_1.items()}

                rank_corr_dict['datapred'].append(datapred)
                rank_corr_dict['llm_model'].append(llm_model)
                rank_corr_dict['fi_method'].append('range_1')
                tau_range_1 = compute_kendall_tau(greedy_scores, range_scores_1)
                rank_corr_dict['kendall_tau'].append(tau_range_1)
                
                if args.verbose:
                    print(r'datapred={}, llm_model={}, fi_method=range_1, tau={}'.format(
                        datapred, llm_model, tau_range_1
                    ))

            # Correlation with LLM scores from greedy decoding with range 8-24
            if 'range_2' in args.recompute or (not prev_exists or not (
                (rank_corr_df['datapred'] == datapred) & \
                (rank_corr_df['llm_model'] == llm_model) & \
                (rank_corr_df['fi_method'] == 'range_2')
            ).any()):
                range_outpath_2 = osp.join(PROMPT_OUTDIR, f'{dataset}/{llm_model}/{pred}_greedy_8.0_24.0.yaml')
                with open(range_outpath_2, 'r') as fh:
                    range_output_2 = yaml.load(fh, Loader=yaml.Loader) # Dictionary
                    range_scores_2 = {k: v['score'] for k,v in range_output_2.items()}

                rank_corr_dict['datapred'].append(datapred)
                rank_corr_dict['llm_model'].append(llm_model)
                rank_corr_dict['fi_method'].append('range_2')
                tau_range_2 = compute_kendall_tau(greedy_scores, range_scores_2)
                rank_corr_dict['kendall_tau'].append(tau_range_2)

                if args.verbose:
                    print(r'datapred={}, llm_model={}, fi_method=range_2, tau={}'.format(
                        datapred, llm_model, tau_range_2
                    ))
            
            # Correlation with Shapley values (XGBoost)
            if 'shap-xgb' in args.recompute or (not prev_exists or not (
                (rank_corr_df['datapred'] == datapred) & \
                (rank_corr_df['llm_model'] == llm_model) & \
                (rank_corr_df['fi_method'] == 'shap-xgb')
            ).any()):
                shap_scores_xgb = compute_mean_shap(args, datapred, shap_model_type='xgb')
                rank_corr_dict['datapred'].append(datapred)
                rank_corr_dict['llm_model'].append(llm_model)
                rank_corr_dict['fi_method'].append('shap-xgb')
                tau_shap_xgb = compute_kendall_tau(greedy_scores, shap_scores_xgb)
                rank_corr_dict['kendall_tau'].append(tau_shap_xgb)

                if args.verbose:
                    print(r'datapred={}, llm_model={}, fi_method=shap-xgb, tau={}'.format(
                        datapred, llm_model, tau_shap_xgb
                    ))

            # Correlation with Shapley values (LightGBM)
            if 'shap-lgb' in args.recompute or (not prev_exists or not (
                (rank_corr_df['datapred'] == datapred) & \
                (rank_corr_df['llm_model'] == llm_model) & \
                (rank_corr_df['fi_method'] == 'shap-lgb')
            ).any()):
                shap_scores_lgb = compute_mean_shap(args, datapred, shap_model_type='lgb')
                rank_corr_dict['datapred'].append(datapred)
                rank_corr_dict['llm_model'].append(llm_model)
                rank_corr_dict['fi_method'].append('shap-lgb')
                tau_shap_lgb = compute_kendall_tau(greedy_scores, shap_scores_lgb)
                rank_corr_dict['kendall_tau'].append(tau_shap_lgb)

                if args.verbose:
                    print(r'datapred={}, llm_model={}, fi_method=shap-lgb, tau={}'.format(
                        datapred, llm_model, tau_shap_lgb
                    ))

            # Correlation with mutual information values
            if 'mi' in args.recompute or (not prev_exists or not (
                (rank_corr_df['datapred'] == datapred) & \
                (rank_corr_df['llm_model'] == llm_model) & \
                (rank_corr_df['fi_method'] == 'mi')
            ).any()):
                # Average over multiple seeds
                tau_mi_list = []
                for seed in args.seeds:
                    mi_scores = compute_mi(datapred, seed=seed)
                    tau_mi_list.append(compute_kendall_tau(greedy_scores, mi_scores))

                rank_corr_dict['datapred'].append(datapred)
                rank_corr_dict['llm_model'].append(llm_model)
                rank_corr_dict['fi_method'].append('mi')
                tau_mi = np.nanmean(tau_mi_list)
                rank_corr_dict['kendall_tau'].append(tau_mi)

                if args.verbose:
                    print(r'datapred={}, llm_model={}, fi_method=mi, tau={}'.format(
                        datapred, llm_model, tau_mi
                    ))

            # Correlation with Pearson correlation
            if 'pearson' in args.recompute or (not prev_exists or not (
                (rank_corr_df['datapred'] == datapred) & \
                (rank_corr_df['llm_model'] == llm_model) & \
                (rank_corr_df['fi_method'] == 'pearson')
            ).any()):
                # Average over multiple seeds
                tau_pearson_list = []
                for seed in args.seeds:
                    pearson_scores = compute_pearson(datapred, seed=seed)
                    tau_pearson_list.append(compute_kendall_tau(greedy_scores, pearson_scores))

                rank_corr_dict['datapred'].append(datapred)
                rank_corr_dict['llm_model'].append(llm_model)
                rank_corr_dict['fi_method'].append('pearson')
                tau_pearson = np.nanmean(tau_pearson_list)
                rank_corr_dict['kendall_tau'].append(tau_pearson)

                if args.verbose:
                    print(r'datapred={}, llm_model={}, fi_method=pearson, tau={}'.format(
                        datapred, llm_model, tau_pearson
                    ))

            # Correlation with Spearman correlation
            if 'spearman' in args.recompute or (not prev_exists or not (
                (rank_corr_df['datapred'] == datapred) & \
                (rank_corr_df['llm_model'] == llm_model) & \
                (rank_corr_df['fi_method'] == 'spearman')
            ).any()):
                # Only compute Spearman correlation if the dataset is used for regression
                if 'wine' in dataset:
                    pred_type = 'regression'
                else:
                    pred_type = DATASET_TO_CLASS[dataset].get_pred_type()
                
                if pred_type == 'regression':
                    # Average over multiple seeds
                    tau_spearman_list = []
                    for seed in args.seeds:
                        spearman_scores = compute_spearman(datapred, seed=seed)
                        tau_spearman_list.append(compute_kendall_tau(greedy_scores, spearman_scores))

                    rank_corr_dict['datapred'].append(datapred)
                    rank_corr_dict['llm_model'].append(llm_model)
                    rank_corr_dict['fi_method'].append('spearman')
                    tau_spearman = np.nanmean(tau_spearman_list)
                    rank_corr_dict['kendall_tau'].append(tau_spearman)

                    if args.verbose:
                        print(r'datapred={}, llm_model={}, fi_method=spearman, tau={}'.format(
                            datapred, llm_model, tau_spearman
                        ))

            # Correlation with Fisher score
            if 'fisher' in args.recompute or (not prev_exists or not (
                (rank_corr_df['datapred'] == datapred) & \
                (rank_corr_df['llm_model'] == llm_model) & \
                (rank_corr_df['fi_method'] == 'fisher')
            ).any()):
                # Only compute Fisher score if the dataset is used for classification
                if 'wine' in dataset:
                    pred_type = 'regression'
                else:
                    pred_type = DATASET_TO_CLASS[dataset].get_pred_type()
                
                if pred_type == 'classification':
                    # Average over multiple seeds
                    tau_fisher_list = []
                    for seed in args.seeds:
                        fisher_scores = compute_fisher_scores(datapred, seed=seed)
                        tau_fisher_list.append(compute_kendall_tau(greedy_scores, fisher_scores))

                    rank_corr_dict['datapred'].append(datapred)
                    rank_corr_dict['llm_model'].append(llm_model)
                    rank_corr_dict['fi_method'].append('fisher')
                    tau_fisher = np.nanmean(tau_fisher_list)
                    rank_corr_dict['kendall_tau'].append(tau_fisher)

                    if args.verbose:
                        print(r'datapred={}, llm_model={}, fi_method=fisher, tau={}'.format(
                            datapred, llm_model, tau_fisher
                        ))

            # Correlation with permutation importance
            if 'perm' in args.recompute or (not prev_exists or not (
                (rank_corr_df['datapred'] == datapred) & \
                (rank_corr_df['llm_model'] == llm_model) & \
                (rank_corr_df['fi_method'] == 'perm')
            ).any()):
                perm_scores = compute_perm_importance(datapred)
                rank_corr_dict['datapred'].append(datapred)
                rank_corr_dict['llm_model'].append(llm_model)
                rank_corr_dict['fi_method'].append('perm')
                tau_perm = compute_kendall_tau(greedy_scores, perm_scores)
                rank_corr_dict['kendall_tau'].append(tau_perm)

                if args.verbose:
                    print(r'datapred={}, llm_model={}, fi_method=perm, tau={}'.format(
                        datapred, llm_model, tau_perm
                    ))
 
            # Correlation with random feature selection
            if 'random' in args.recompute or (not prev_exists or not (
                (rank_corr_df['datapred'] == datapred) & \
                (rank_corr_df['llm_model'] == llm_model) & \
                (rank_corr_df['fi_method'] == 'random')
            ).any()):
                tau_random_list = []
                for seed in args.seeds:
                    np.random.seed(seed)
                    random_scores = np.random.uniform(low=0, high=1, size=(len(concepts),))
                    random_scores = {k: random_scores[i] for i,k in enumerate(concepts)}
                    tau_random_list.append(compute_kendall_tau(greedy_scores, random_scores))

                rank_corr_dict['datapred'].append(datapred)
                rank_corr_dict['llm_model'].append(llm_model)
                rank_corr_dict['fi_method'].append('random')
                tau_random = np.nanmean(tau_random_list)
                rank_corr_dict['kendall_tau'].append(tau_random)

                if args.verbose:
                    print(r'datapred={}, llm_model={}, fi_method=random, tau={}'.format(
                        datapred, llm_model, tau_random
                    ))

    # Save the computed rank correlations
    if prev_exists:
        rank_corr_df = pd.concat([rank_corr_df, pd.DataFrame(rank_corr_dict)], axis=0)
        rank_corr_df.drop_duplicates(subset=['datapred','llm_model','fi_method'], keep='last', inplace=True)
    else:
        rank_corr_df = pd.DataFrame(rank_corr_dict)

    rank_corr_df.sort_values(by=['datapred', 'llm_model', 'fi_method'], inplace=True)
    rank_corr_df.to_csv(rank_corr_path, index=False, header=True)

    print(f'\nElapsed: {str(datetime.now() - start_time)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datapreds', 
        help='Dataset + prediction task pairs', 
        nargs='*', 
        default=[
            'bank/subscription',
            'calhousing/price',
            'compas/recid',
            'credit-g/risk',
            'pima/diabetes',
            'diabetes/progression',
            'give-me-credit/delinquency',
            'miami-housing/price',
            'wine/quality',
            'cars/price'
        ], 
        type=str
    )
    parser.add_argument(
        '--llm_models', 
        help='LLMs used for evaluation', 
        nargs='*', 
        default=['gpt-4-0613', 'gpt-3.5-turbo', 'llama-2-70b-chat', 'llama-2-13b-chat', 'llama-2-7b-chat'], 
        type=str
    )
    parser.add_argument('--max_k', help='Maximum number of centroids in K-means (used for KernelSHAP)', default=25, type=int)
    parser.add_argument('--seeds', help='Random seeds for reproducibility', nargs='*', default=[1,2,3,4,5], type=int)
    parser.add_argument('--outdir', help='Output directory', default='./results/rank_corr', type=str)
    parser.add_argument('--recompute', help='Importance metrics to recompute', nargs='*', default=[], type=str)
    parser.add_argument('--run_locally', help='Option to run Ray Tune locally without W&B', default=False, action='store_true')
    parser.add_argument('--verbose', help='Verbosity', default=False, action='store_true')
    parser.add_argument('--result_suffix', help='Optional suffix for result', default='', type=str)
    args = parser.parse_args()

    main(args)
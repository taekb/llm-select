import os
import os.path as osp
import pickle
from pprint import pprint
from functools import partial
from datetime import datetime
import pandas as pd
import sys
sys.path.append('../')

import yaml
from textwrap import dedent
import argparse
from thefuzz import fuzz
from tqdm import tqdm
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.feature_selection import (
    mutual_info_classif,
    mutual_info_regression,
    SelectPercentile, 
    SelectFromModel,
    RFE
)
from sklearn.metrics import (
    make_scorer, 
    roc_auc_score, 
    average_precision_score,
    mean_absolute_error,
    mean_squared_error
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
import seaborn as sns

import langchain
import langchain.output_parsers as output_parsers
import langchain.chat_models as chat_models
import langchain.llms as llms
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage
from langchain.prompts.chat import (
    SystemMessage, 
    HumanMessagePromptTemplate, 
    MessagesPlaceholder
)
from langchain.memory import (
    ConversationBufferMemory, # Stores all previous messages in buffer
    ConversationBufferWindowMemory # Stores only last k interactions in buffer
)

import ray

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

from lassonet import LassoNetClassifier, LassoNetRegressor
from mrmr import mrmr_classif, mrmr_regression

import datasets
import selection

# Directories
HF_LLAMA2_DIR = 'meta-llama/Llama-2-{}b-chat-hf'

# Default settings
DATASET_TO_FSCONFIG = {
    'cars': {
        'lasso': [0.01, 0.05, 0.075, 0.1, 0.2, 0.3, 0.5, 1],
        'lassonet': [0.1 * i for i in range(1,11)],
        'random-concept': [0.15 * i for i in range(1,8)],
        'llm-rank': [0.15 * i for i in range(1,8)],
        'llm-ratio': [0.15 * i for i in range(1,8)],
        'llm-forward-seq': [0.15 * i for i in range(1,8)],
        'llm-forward-seq-empty': [0.15 * i for i in range(1,8)],
        'filter-mi': [0.1 * i for i in range(1,11)],
        'forward-seq': [0.1 * i for i in range(1,11)],
        'backward-seq': [0.1 * i for i in range(1,11)],
        'rfe': [0.1 * i for i in range(1,11)],
        'mrmr': [0.1 * i for i in range(1,11)]
    },
    'pima': {
        'lasso': [0.01, 0.025, 0.05, 0.1, 0.5, 1], # C values
        'lassonet': [0.125 * i for i in range(1,9)],
        'random-concept': [0.125 * i for i in range(1,9)],
        'llm-rank': [0.125 * i for i in range(1,9)],
        'llm-ratio': [0.125 * i for i in range(1,9)],
        'llm-forward-seq': [0.125 * i for i in range(1,9)],
        'llm-forward-seq-empty': [0.125 * i for i in range(1,9)],
        'filter-mi': [0.125 * i for i in range(1,9)],
        'forward-seq': [0.125 * i for i in range(1,9)],
        'backward-seq': [0.125 * i for i in range(1,9)],
        'rfe': [0.125 * i for i in range(1,9)],
        'mrmr': [0.125 * i for i in range(1,9)]
    },
    'give-me-credit': {
        'lasso': [0.00075, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.03, 0.05], # C values
        'lassonet': [0.1 * i for i in range(1,11)],
        'random-concept': [0.1 * i for i in range(1,11)],
        'llm-rank': [0.1 * i for i in range(1,11)],
        'llm-ratio': [0.1 * i for i in range(1,11)],
        'llm-forward-seq': [0.1 * i for i in range(1,11)],
        'llm-forward-seq-empty': [0.1 * i for i in range(1,11)],
        'filter-mi': [0.1 * i for i in range(1,11)],
        'forward-seq': [0.1 * i for i in range(1,11)],
        'backward-seq': [0.1 * i for i in range(1,11)],
        'rfe': [0.1 * i for i in range(1,11)],
        'mrmr': [0.1 * i for i in range(1,11)]
    },
    'compas': {
        'lasso': [0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 1, 10], # C values
        'lassonet': [0.1 * i for i in range(1,11)],
        'random-concept': [0.1 * i for i in range(1,11)],
        'llm-rank': [0.1 * i for i in range(1,11)],
        'llm-ratio': [0.1 * i for i in range(1,11)],
        'llm-forward-seq': [0.1 * i for i in range(1,11)],
        'llm-forward-seq-empty': [0.1 * i for i in range(1,11)],
        'filter-mi': [0.1 * i for i in range(1,11)],
        'forward-seq': [0.1 * i for i in range(1,11)],
        'backward-seq': [0.1 * i for i in range(1,11)],
        'rfe': [0.1 * i for i in range(1,11)],
        'mrmr': [0.1 * i for i in range(1,11)]
    },
    'miami-housing': {
        'lasso': [0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25],
        'lassonet': [0.1 * i for i in range(1,11)],
        'random-concept': [0.1 * i for i in range(1,11)],
        'llm-rank': [0.1 * i for i in range(1,11)],
        'llm-ratio': [0.1 * i for i in range(1,11)],
        'llm-forward-seq': [0.1 * i for i in range(1,11)],
        'llm-forward-seq-empty': [0.1 * i for i in range(1,11)],
        'filter-mi': [0.1 * i for i in range(1,11)],
        'forward-seq': [0.1 * i for i in range(1,11)],
        'backward-seq': [0.1 * i for i in range(1,11)],
        'rfe': [0.1 * i for i in range(1,11)],
        'mrmr': [0.1 * i for i in range(1,11)]
    },
    'wine': {
        'lasso': [0.001, 0.005, 0.01, 0.025, 0.05, 0.1],
        'lassonet': [0.1 * i for i in range(1,11)],
        'random-concept': [0.1 * i for i in range(1,11)],
        'llm-rank': [0.1 * i for i in range(1,11)],
        'llm-ratio': [0.1 * i for i in range(1,11)],
        'llm-forward-seq': [0.1 * i for i in range(1,11)],
        'llm-forward-seq-empty': [0.1 * i for i in range(1,11)],
        'filter-mi': [0.1 * i for i in range(1,11)],
        'forward-seq': [0.1 * i for i in range(1,11)],
        'backward-seq': [0.1 * i for i in range(1,11)],
        'rfe': [0.1 * i for i in range(1,11)],
        'mrmr': [0.1 * i for i in range(1,11)]
    },
    'credit-g': {
        'lasso': [0.05, 0.1, 0.5, 1, 5, 10, 50, 100], # C values
        'lassonet': [0.1 * i for i in range(1,11)],
        'random-concept': [0.1 * i for i in range(1,11)],
        'llm-rank': [0.1 * i for i in range(1,11)],
        'llm-ratio': [0.1 * i for i in range(1,11)],
        'llm-forward-seq': [0.1 * i for i in range(1,11)],
        'llm-forward-seq-empty': [0.1 * i for i in range(1,11)],
        'filter-mi': [0.1 * i for i in range(1,11)],
        'forward-seq': [0.1 * i for i in range(1,11)],
        'backward-seq': [0.1 * i for i in range(1,11)],
        'rfe': [0.1 * i for i in range(1,11)],
        'mrmr': [0.1 * i for i in range(1,11)]
    },
    'diabetes': {
        'lasso': [0.1, 0.5, 1, 5, 10, 20, 40],
        'lassonet': [0.1 * i for i in range(1,11)],
        'random-concept': [0.1 * i for i in range(1,11)],
        'llm-rank': [0.1 * i for i in range(1,11)],
        'llm-ratio': [0.1 * i for i in range(1,11)],
        'llm-forward-seq': [0.1 * i for i in range(1,11)],
        'llm-forward-seq-empty': [0.1 * i for i in range(1,11)],
        'filter-mi': [0.1 * i for i in range(1,11)],
        'forward-seq': [0.1 * i for i in range(1,11)],
        'backward-seq': [0.1 * i for i in range(1,11)],
        'rfe': [0.1 * i for i in range(1,11)],
        'mrmr': [0.1 * i for i in range(1,11)]
    },
    'calhousing': {
        'lasso': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
        'lassonet': [0.125 * i for i in range(1,9)],
        'random-concept': [0.125 * i for i in range(1,9)],
        'llm-rank': [0.125 * i for i in range(1,9)],
        'llm-ratio': [0.125 * i for i in range(1,9)],
        'llm-forward-seq': [0.125 * i for i in range(1,9)],
        'llm-forward-seq-empty': [0.125 * i for i in range(1,9)],
        'filter-mi': [0.125 * i for i in range(1,9)],
        'forward-seq': [0.125 * i for i in range(1,9)],
        'backward-seq': [0.125 * i for i in range(1,9)],
        'rfe': [0.125 * i for i in range(1,9)],
        'mrmr': [0.125 * i for i in range(1,9)]
    },
    'bank': {
        'lasso': [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1], # C values
        'lassonet': [0.1 * i for i in range(1,11)],
        'random-concept': [0.1 * i for i in range(1,11)],
        'llm-rank': [0.1 * i for i in range(1,11)],
        'llm-ratio': [0.1 * i for i in range(1,11)],
        'llm-forward-seq': [0.1 * i for i in range(1,11)],
        'llm-forward-seq-empty': [0.1 * i for i in range(1,11)],
        'filter-mi': [0.1 * i for i in range(1,11)],
        'forward-seq': [0.1 * i for i in range(1,11)],
        'backward-seq': [0.1 * i for i in range(1,11)],
        'rfe': [0.1 * i for i in range(1,11)],
        'mrmr': [0.1 * i for i in range(1,11)]
    }
}

# Total input dimensionality when using all features
DATAPRED_TO_DIMS = {
    'bank/subscription': 51,
    'calhousing/price': 8,
    'diabetes/progression': 10,
    'credit-g/risk': 61,
    'wine/quality': 11,
    'miami-housing/price': 15,
    'compas/recid': 23,
    'give-me-credit/delinquency': 10,
    'pima/diabetes': 8,
    'cars/price': 11
}

# Classification/regression
DATASET_TO_PREDTYPE = {
    'bank': 'classification',
    'calhousing': 'regression',
    'diabetes': 'regression',
    'credit-g': 'classification',
    'wine': 'regression',
    'miami-housing': 'regression',
    'compas': 'classification',
    'give-me-credit': 'classification',
    'pima': 'classification',
    'cars': 'regression'
}

LLM_PLOTNAME = {
    'gpt-4-0613': 'GPT-4',
    'gpt-3.5-turbo': 'GPT-3.5',
    'llama-2-70b-chat': 'Llama-2-70B',
    'llama-2-13b-chat': 'Llama-2-13B',
    'llama-2-7b-chat': 'Llama-2-7B'
}

# Load OpenAI API key
with open('../config/openai_api_key.txt', 'r') as fh:
    openai_api_key = fh.read().strip()

def load_dataset(
    args, 
    split, 
    select, 
    fs_config, 
    seed, 
    split_train_val=False,
    normalize=False, 
    llm_model='', 
    llm_decoding='', 
    context_config='',
    example_config='',
    **kwargs
):
    '''Data loading function for feature selection comparison.'''

    dataset, _ = args.datapred.split('/')

    with_context = True if context_config == 'context' else False
    with_examples = True if example_config == 'examples' else False
    with_expls = True if example_config == 'expls' else False

    base_config = dict(
        select=select,
        split=split,
        split_train_val=split_train_val,
        llm_model=llm_model,
        llm_decoding=llm_decoding,
        with_context=with_context,
        with_examples=with_examples,
        with_expls=with_expls,
        ratio=(fs_config if 'threshold' not in select else None),
        threshold=(fs_config if 'threshold' in select else None),
        seed=seed,
        stratify_split=True,
        normalize=normalize,
        flatten=True,
        return_tensor=False,
        **kwargs
    )

    # Used Car Price (Regression)
    if dataset == 'cars':
        return datasets.CarsDataset(**base_config)

    # Pima Indians Diabetes (Classification)
    elif dataset == 'pima':
        return datasets.PimaDataset(**base_config)

    # Give Me Credit (Classification)
    elif dataset == 'give-me-credit':
        return datasets.GiveMeCreditDataset(
            remove_outliers=True, 
            balance_classes=True,
            **base_config
        )

    # COMPAS Recidivism Risk (Classification)
    elif dataset == 'compas':
        return datasets.COMPASDataset(**base_config)

    # Miami Housing Price (Regression)
    elif dataset == 'miami-housing':
        return datasets.MiamiHousingDataset(**base_config)

    # Wine (Regression)
    elif dataset == 'wine':
        return datasets.WineDataset(**base_config)

    # German Credit Risk (Regression)
    elif dataset == 'credit-g':
        return datasets.CreditGDataset(**base_config)

    # Diabetes (Regression)
    elif dataset == 'diabetes':
        return datasets.DiabetesDataset(**base_config)
    
    # California Housing Price (Regression)
    elif dataset == 'calhousing':
        return datasets.CalHousingDataset(**base_config)
    
    # Bank Subscription (Classification)
    elif dataset == 'bank':
        return datasets.BankDataset(**base_config)

def main(args):
    start_time = datetime.now()
    dataset, pred = args.datapred.split('/')
    pred_type = DATASET_TO_PREDTYPE[dataset] # Classification/Regression

    # Example: ../results/linear_compare/acs/
    results_dir = osp.join(args.outdir, dataset)
    os.makedirs(results_dir, exist_ok=True)
    results = {} # {method: [[selection_ratios], [avg_auroc], [std_auroc]]}

    for fs_method in args.fs_methods:
        print(f'\nRunning feature selection with method={fs_method}...')

        fs_basename = fs_method.split(':')[0]
        if fs_basename in ['filter-mi', 'mrmr']:
            select = 'all'
        elif fs_basename == 'llm-forward-seq':
            select = 'llm-topk'
        else:
            select = fs_basename
        
        if fs_basename.startswith('llm'):
            _, llm_model, llm_decoding, context_config, example_config = fs_method.split(':')
        else:
            llm_model, llm_decoding, context_config, example_config = '', '', '', ''

        # Forward/backward selection, recursive feature elimination (RFE)
        if fs_basename in ['forward-seq', 'backward-seq', 'rfe']:
            all_fracs = []
            all_test_metrics = []

            for seed in args.seeds:
                train_dataset = load_dataset(args, 'train', 'all', None, seed)
                test_dataset = load_dataset(args, 'test', 'all', None, seed)

                # Manually handle data normalization
                if len(train_dataset.cat_idxs) > 0:
                    onehot = ColumnTransformer([(
                        'onehot', 
                        OneHotEncoder(sparse_output=False, handle_unknown='ignore'), 
                        train_dataset.cat_idxs
                    )], remainder='passthrough')
                    X_train = onehot.fit_transform(train_dataset.X)
                    X_test = onehot.transform(test_dataset.X)
                else:
                    X_train = train_dataset.X
                    X_test = test_dataset.X

                # NOTE: May need to modify below if there are datasets with only categorical features
                # i.e., CustomSequentialFeatureSelection internally standardizes the numerical indices
                D = X_train.shape[-1]
                cont_idxs = np.arange(D-len(train_dataset.cont_idxs), D)

                # Define feature selection model and configs
                if pred_type == 'classification':
                    fs_model = LogisticRegression(class_weight='balanced')
                    param_space = {'clf__C': [0.1, 0.5, 1, 5, 10, 50, 100]}
                    scoring = 'roc_auc'
                    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                
                elif pred_type == 'regression':
                    fs_model = Ridge()
                    param_space = {'clf__alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]}
                    scoring = 'neg_mean_absolute_error'
                    cv = 5

                # Run grid search to find the best hyperparameters for the feature selection model
                print('Running grid search for the feature selection model...')

                fs_scaler = ColumnTransformer([
                    ('z-score', StandardScaler(), cont_idxs)
                ], remainder='passthrough')

                fs_pipe = Pipeline([('normalize', fs_scaler), ('clf', fs_model)])
                fs_grid = GridSearchCV(
                    fs_pipe, param_space, verbose=args.verbose, cv=cv, scoring=scoring, refit=True, n_jobs=64
                ).fit(X_train, train_dataset.Y)
                
                # Recursive feature elimination (RFE)
                if fs_basename == 'rfe':
                    print(f'Running RFE with seed={seed}...')
                    n_features_to_select = 1

                    # Run RFE
                    rfe = RFE(
                        clone(fs_grid.best_estimator_.named_steps['clf']), 
                        n_features_to_select=n_features_to_select,
                        step=1,
                        verbose=args.verbose
                    ).fit(fs_scaler.fit_transform(X_train), train_dataset.Y)

                    order = np.argsort(rfe.ranking_)

                # Forward/backward sequential selection
                else:
                    direction = 'forward' if fs_basename == 'forward-seq' else 'backward'
                    print(f'Running {direction} selection with seed={seed}...')
                    n_features_to_select = D-1 if fs_basename == 'forward-seq' else 1
                    sfs = selection.utils.CustomSequentialFeatureSelector(
                        clone(fs_grid.best_estimator_.named_steps['clf']), 
                        n_features_to_select=n_features_to_select,
                        direction=direction,
                        cv=cv,
                        scoring=scoring,
                        n_jobs=64,
                        cont_idxs=cont_idxs # Normalization applied internally
                    ).fit(X_train, train_dataset.Y)

                    # Retrieve ordered list of selected feature indices
                    order = sfs.feature_idx_order_ # NOTE: Includes one less than D
                    
                    # Add missing feature index
                    missing_idx = (set(np.arange(0,D)) - set(order)).pop()

                    if direction == 'forward':
                        order.append(missing_idx)
                    else:
                        order = order[::-1] # Need to flip the order, since order = order of elimination
                        order.append(missing_idx)

                assert(len(order) == D)

                # Run grid search with ordered features
                seed_fracs = []
                seed_test_metrics = []
                for fs_config in DATASET_TO_FSCONFIG[dataset][fs_basename]:
                    if args.verbose:
                        print(f'\nRunning grid search with ratio={fs_config}...')
                    
                    selected_idxs = order[:int(fs_config * D)]
                    X_train_selected = X_train[:,selected_idxs]
                    X_test_selected = X_test[:,selected_idxs]

                    # Check which indices are numeric
                    selected_cont_idxs = []
                    for i, idx in enumerate(selected_idxs):
                        if idx in cont_idxs:
                            selected_cont_idxs.append(i)

                    scaler = ColumnTransformer([
                        ('z-score', StandardScaler(), selected_cont_idxs)
                    ], remainder='passthrough')
                    
                    # Define downstream prediction model
                    if pred_type == 'classification':
                        model = LogisticRegression(class_weight='balanced')

                    elif pred_type == 'regression':
                        model = Ridge()

                    # Grid Search CV
                    pipe = Pipeline([('normalize', scaler), ('clf', model)])
                    grid = GridSearchCV(
                        pipe, param_space, verbose=args.verbose, cv=cv, scoring=scoring, refit=True, n_jobs=64
                    ).fit(X_train_selected, train_dataset.Y)
                    
                    best_pipe = grid.best_estimator_
                    weights = best_pipe.named_steps['clf'].coef_
                    seed_fracs.append(weights.size / float(DATAPRED_TO_DIMS[args.datapred])) 
                    
                    # Compute the test metric
                    if pred_type == 'classification':
                        Y_prob = best_pipe.predict_proba(X_test_selected)[:,-1]
                        seed_test_metrics.append(roc_auc_score(test_dataset.Y, Y_prob))
                    
                    elif pred_type == 'regression':
                        Y_pred = best_pipe.predict(X_test_selected)
                        seed_test_metrics.append(mean_absolute_error(test_dataset.Y, Y_pred))

                all_fracs.append(seed_fracs)
                all_test_metrics.append(seed_test_metrics)

            avg_fracs = np.mean(all_fracs, axis=0)
            avgs = np.mean(all_test_metrics, axis=0)
            stds = np.std(all_test_metrics, axis=0)

        # LassoNet (Lemhadri et al., 2021)
        elif fs_basename == 'lassonet':
            all_fracs = []
            all_test_metrics = []

            for seed in args.seeds:
                # Load normalized data for fitting LassoNet
                train_dataset = load_dataset(args, 'train', 'all', None, seed, normalize=True, split_train_val=True)
                val_dataset = load_dataset(args, 'val', 'all', None, seed, normalize=True, split_train_val=True)
                test_dataset = load_dataset(args, 'test', 'all', None, seed, normalize=True, split_train_val=True)
                batch_size = 1024 if train_dataset.X.shape[0] >= 10000 else 256

                # Optimizer config
                if args.datapred == 'diabetes/progression':
                    optim = (partial(torch.optim.Adam, lr=1e-4), partial(torch.optim.SGD, lr=1e-4, momentum=0.9))
                else:
                    optim = (partial(torch.optim.Adam, lr=1e-3), partial(torch.optim.SGD, lr=1e-3, momentum=0.9))

                if pred_type == 'classification':
                    lassonet_model = LassoNetClassifier(
                        hidden_dims=(100,),
                        M=10, 
                        optim=optim,
                        batch_size=batch_size,
                        verbose=True,
                        n_iters=(10,3),
                        torch_seed=seed, # Random seed
                        patience=3 # Patience config for both initial model training and regularization path computation
                    )
                elif pred_type == 'regression':
                    lassonet_model = LassoNetRegressor(
                        hidden_dims=(100,),
                        M=10, 
                        optim=optim,
                        batch_size=batch_size,
                        verbose=True,
                        n_iters=(10,3),
                        torch_seed=seed, # Random seed
                        patience=3 # Patience config for both initial model training and regularization path computation
                    )

                # Fit regularization path
                print(f'\nComputing LassoNet regularization path for {args.datapred}...')
                path = lassonet_model.path(train_dataset.X, train_dataset.Y, X_val=val_dataset.X, y_val=val_dataset.Y)
                order = np.argsort(lassonet_model._compute_feature_importances(path).cpu().numpy())[::-1]

                # Reload data without normalization
                train_dataset = load_dataset(args, 'train', 'all', None, seed)
                test_dataset = load_dataset(args, 'test', 'all', None, seed)

                # Manually handle data normalization
                if len(train_dataset.cat_idxs) > 0:
                    onehot = ColumnTransformer([(
                        'onehot',
                        OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
                        train_dataset.cat_idxs
                    )], remainder='passthrough')
                    X_train = onehot.fit_transform(train_dataset.X)
                    X_test = onehot.transform(test_dataset.X)
                else:
                    X_train = train_dataset.X
                    X_test = test_dataset.X

                D = X_train.shape[-1]
                cont_idxs = np.arange(D-len(train_dataset.cont_idxs), D)

                # Run grid search with ordered features
                seed_fracs = []
                seed_test_metrics = []
                for fs_config in DATASET_TO_FSCONFIG[dataset][fs_basename]:
                    if args.verbose:
                        print(f'\nRunning grid search with ratio={fs_config}...')

                    selected_idxs = order[:int(fs_config * D)]
                    X_train_selected = X_train[:,selected_idxs]
                    X_test_selected = X_test[:,selected_idxs]

                    # Check which indices are numeric
                    selected_cont_idxs = []
                    for i, idx in enumerate(selected_idxs):
                        if idx in cont_idxs:
                            selected_cont_idxs.append(i)
                    
                    scaler = ColumnTransformer([
                        ('z-score', StandardScaler(), selected_cont_idxs)
                    ], remainder='passthrough')

                    # Define downstream prediction model
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

                    # Grid Search CV
                    pipe = Pipeline([('normalize', scaler), ('clf', model)])
                    grid = GridSearchCV(
                        pipe, param_space, verbose=args.verbose, cv=cv, scoring=scoring, refit=True, n_jobs=64
                    ).fit(X_train_selected, train_dataset.Y)
                    
                    best_pipe = grid.best_estimator_
                    weights = best_pipe.named_steps['clf'].coef_
                    seed_fracs.append(weights.size / float(DATAPRED_TO_DIMS[args.datapred])) 
                    
                    # Compute the test metric
                    if pred_type == 'classification':
                        Y_prob = best_pipe.predict_proba(X_test_selected)[:,-1]
                        seed_test_metrics.append(roc_auc_score(test_dataset.Y, Y_prob))
                    
                    elif pred_type == 'regression':
                        Y_pred = best_pipe.predict(X_test_selected)
                        seed_test_metrics.append(mean_absolute_error(test_dataset.Y, Y_pred))

                all_fracs.append(seed_fracs)
                all_test_metrics.append(seed_test_metrics)

            avg_fracs = np.mean(all_fracs, axis=0)
            avgs = np.mean(all_test_metrics, axis=0)
            stds = np.std(all_test_metrics, axis=0)
        
        # MRMR selection (Ding and Peng, 2005)
        elif fs_basename == 'mrmr':
            avg_fracs = []
            avgs = []
            stds = []

            fs_configs = DATASET_TO_FSCONFIG[dataset][fs_basename]

            for fs_config in fs_configs:
                fracs = []
                test_metrics = []

                for seed in args.seeds:
                    # Load dataset
                    train_dataset = load_dataset(args, 'train', 'all', fs_config, seed)
                    test_dataset = load_dataset(args, 'test', 'all', fs_config, seed)
                    normalizer = train_dataset.get_normalizer()

                    # Select features based on MRMR
                    if pred_type == 'classification':
                        selected_idxs = mrmr_classif(
                            X=pd.DataFrame(normalizer.fit_transform(train_dataset.X)),
                            y=train_dataset.Y,
                            K=int(fs_config * DATAPRED_TO_DIMS[args.datapred])
                        )
                        model = LogisticRegression(class_weight='balanced')
                        param_space = {'clf__C': [0.1, 0.5, 1, 5, 10, 50, 100]}
                        scoring = 'roc_auc'
                        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                
                    elif pred_type == 'regression':
                        selected_idxs = mrmr_regression(
                            X=pd.DataFrame(normalizer.fit_transform(train_dataset.X)),
                            y=train_dataset.Y,
                            K=int(fs_config * DATAPRED_TO_DIMS[args.datapred])
                        )
                        model = Ridge()
                        param_space = {'clf__alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]}
                        scoring = 'neg_mean_absolute_error'
                        cv = 5

                    selected_idxs = sorted(selected_idxs)
                    feature_filter = ColumnTransformer([
                        ('select', FunctionTransformer(lambda x: x), selected_idxs)
                    ], remainder='drop')

                    # Pipeline config
                    pipe = Pipeline([('normalizer', normalizer), ('select', feature_filter), ('clf', model)])
                    
                    # Grid Search CV
                    grid = GridSearchCV(
                        pipe, param_space, verbose=args.verbose, cv=cv, scoring=scoring, refit=True, n_jobs=64
                    ).fit(train_dataset.X, train_dataset.Y)

                    best_pipe = grid.best_estimator_
                    weights = best_pipe.named_steps['clf'].coef_
                    fracs.append(weights.size / float(DATAPRED_TO_DIMS[args.datapred])) 
                    
                    # Compute the test metric
                    if pred_type == 'classification':
                        Y_prob = best_pipe.predict_proba(test_dataset.X)[:,-1]
                        test_metrics.append(roc_auc_score(test_dataset.Y, Y_prob))
                    
                    elif pred_type == 'regression':
                        Y_pred = best_pipe.predict(test_dataset.X)
                        test_metrics.append(mean_absolute_error(test_dataset.Y, Y_pred))

                avg_fracs.append(np.mean(fracs))
                avgs.append(np.mean(test_metrics))
                stds.append(np.std(test_metrics))
        
        # LLM-based forward sequential selection
        elif fs_basename in ['llm-forward-seq', 'llm-forward-seq-empty']:
            all_fracs = []
            all_test_metrics = []

            # Load all concepts for the dataset
            concepts = selection.utils.load_concepts(dataset=dataset, pred=pred)

            # Check if previous sequential prompt results exist
            seq_prev_exists = False
            if 'empty' not in fs_basename and osp.exists(osp.join(results_dir, f'{llm_model}_forward-seq.yaml')):
                seq_prev_exists = True

                with open(osp.join(results_dir, f'{llm_model}_forward-seq.yaml'), 'r') as fh:
                    concept_orders = yaml.load(fh, Loader=yaml.Loader)

                if args.verbose:
                    print('Loaded existing forward sequential selection results.')

            elif 'empty' in fs_basename and osp.exists(osp.join(results_dir, f'{llm_model}_forward-seq-empty.yaml')):
                seq_prev_exists = True

                with open(osp.join(results_dir, f'{llm_model}_forward-seq-empty.yaml'), 'r') as fh:
                    concept_orders = yaml.load(fh, Loader=yaml.Loader)

                if args.verbose:
                    print('Loaded existing forward sequential selection results.')

            else:
                concept_orders = []

            # Set up LLM
            if llm_model in ['gpt-3.5-turbo', 'gpt-4-0613']:
                llm = chat_models.ChatOpenAI(
                    model_name=llm_model,
                    temperature=0,
                    openai_api_key=openai_api_key,
                    max_tokens=128,
                    n=1,
                    max_retries=5
                )

            elif llm_model.startswith('llama-2'):
                model_size = int(llm_model.split('-')[2][:-1])
                model_id = HF_LLAMA2_DIR.format(model_size)
                llm = llms.VLLM(
                    model=model_id,
                    tensor_parallel_size=4,
                    temperature=0,
                    gpu_memory_utilization=0.85,
                    max_tokens=128
                )

            else:
                raise RuntimeError(f'[Error] Invalid LLM option: {llm_model}.')

            for seed in args.seeds:
                if seq_prev_exists:
                    concept_order = [concept_orders[seed-1][0]]
                    train_dataset = load_dataset(
                        args, 'train', 'all', None, seed, llm_model=llm_model, llm_decoding=llm_decoding,
                        context_config=context_config, example_config=example_config, filtered_concepts=concept_order
                    )
                    test_dataset = load_dataset(
                        args, 'train', 'all', None, seed, llm_model=llm_model, llm_decoding=llm_decoding,
                        context_config=context_config, example_config=example_config, filtered_concepts=concept_order
                    )
                    normalizer = train_dataset.get_normalizer()
                else:
                    # Preload LLM scores
                    with open(f'../prompt_outputs/{dataset}/{llm_model}/{pred}_{llm_decoding}.yaml', 'r') as fh:
                        llm_scores = yaml.load(fh, Loader=yaml.Loader)

                    # Order in which concepts are added
                    concept_order = []

                    # If starting with nonempty set, add top scoring concept according to LLM-Score
                    if not 'empty' in fs_basename:
                        concept_order += list(selection.utils.llm_select_concepts( # np.array
                            llm_scores, topk=1, filtered_concepts=None
                        ))

                        # NOTE: select == 'llm-topk'
                        train_dataset = load_dataset(
                            args, 
                            'train', 
                            select, 
                            None, 
                            seed, 
                            llm_model=llm_model, 
                            llm_decoding=llm_decoding,
                            context_config=context_config, 
                            example_config=example_config, 
                            topk=1,
                            filtered_concepts=concept_order
                        )
                        test_dataset = load_dataset(
                            args, 
                            'test', 
                            select, 
                            None, 
                            seed, 
                            llm_model=llm_model, 
                            llm_decoding=llm_decoding,
                            context_config=context_config, 
                            example_config=example_config, 
                            topk=1,
                            filtered_concepts=concept_order
                        )
                        normalizer = train_dataset.get_normalizer()

                        new_concept_added = True
                    else:
                        new_concept_added = False
                        seed_test_metric = 'N/A'

                    # Set up chat messages
                    template = selection.utils.load_template(dataset=dataset, pred=pred, sequential=True)
                    
                    # Add context if specified
                    if context_config == 'context':
                        context = selection.utils.load_context(dataset=dataset, pred=pred)
                        template = template.format(context=context)
                    else:
                        template = template.format(context='')
                    
                    instruction = dedent(
                        """
                        I used the features {concept_order}, 
                        and the trained model achieved a test {metric} of {value}. 
                        What feature should I add next from: {candidate_concepts}? 
                        Give me just the name of the feature to add (no other text).
                        """
                    )
                    if llm_model.startswith('llama-2'):
                        instruction += ' [/INST]'

                    chat_template = ChatPromptTemplate.from_messages([
                        SystemMessage(content=template),
                        MessagesPlaceholder(variable_name='chat_history'),
                        HumanMessagePromptTemplate.from_template(instruction)
                    ])

                    # Set up chat buffer
                    if args.buffer_chat:
                        memory = ConversationBufferWindowMemory(
                            k=args.chat_buffer_size, 
                            memory_key='chat_history', 
                            input_key='concept_order', 
                            return_messages=True
                        )
                    else:
                        memory = ConversationBufferMemory(
                            memory_key='chat_history',
                            input_key='concept_order',
                            return_messages=True
                        )

                    # Define LLM chain
                    llm_chain = langchain.LLMChain(
                        llm=llm, prompt=chat_template, verbose=args.verbose, memory=memory
                    )

                # Run sequential feature selection
                seed_fracs = []
                seed_test_metrics = []
                all_concepts_selected = False

                while not all_concepts_selected:
                    # Only train a model if a new concept has been added
                    if new_concept_added:
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
                        
                        # Grid Search CV
                        pipe = Pipeline([('normalizer', normalizer), ('clf', model)])
                        grid = GridSearchCV(
                            pipe, param_space, verbose=args.verbose, cv=cv, scoring=scoring, refit=True, n_jobs=64
                        ).fit(train_dataset.X, train_dataset.Y)

                        best_pipe = grid.best_estimator_
                        weights = best_pipe.named_steps['clf'].coef_
                        seed_fracs.append(weights.size / float(DATAPRED_TO_DIMS[args.datapred]))

                        # Compute the test metric
                        if pred_type == 'classification':
                            Y_prob = best_pipe.predict_proba(test_dataset.X)[:,-1]
                            seed_test_metric = roc_auc_score(test_dataset.Y, Y_prob)
                            seed_test_metrics.append(seed_test_metric)
                            
                            if args.verbose:
                                print(f'AUROC={seed_test_metric}')

                        elif pred_type == 'regression':
                            Y_pred = best_pipe.predict(test_dataset.X)
                            seed_test_metric = mean_absolute_error(test_dataset.Y, Y_pred)
                            seed_test_metrics.append(seed_test_metric)
                        
                            if args.verbose:
                                print(f'MAE={seed_test_metric}')

                    # Check if all features have been selected
                    if len(set(concept_order)) == len(concepts):
                        all_concepts_selected = True

                        if args.verbose:
                            print('All features selected.\n')

                        continue
                    
                    # Add new concept if appropriately selected
                    if seq_prev_exists: # Using already existing selection order
                        new_concept = concept_orders[seed-1][len(concept_order)]
                        concept_order.append(new_concept)
                        new_concept_added = True

                        if args.verbose:
                            print(f'Added new concept: {new_concept}')

                        # Update the datasets
                        train_dataset = load_dataset(
                            args, 'train', 'all', None, seed, llm_model=llm_model, llm_decoding=llm_decoding,
                            filtered_concepts=concept_order
                        )
                        test_dataset = load_dataset(
                            args, 'test', 'all', None, seed, llm_model=llm_model, llm_decoding=llm_decoding,
                            filtered_concepts=concept_order
                        )
                        normalizer = train_dataset.get_normalizer()

                    else:
                        candidate_concepts = list(set(concepts) - set(concept_order))

                        # Only consider the candidates with top relevance scores
                        if args.buffer_concepts:
                            candidate_concepts = selection.utils.llm_select_concepts(
                                llm_scores, 
                                topk=args.concept_buffer_size, 
                                filtered_concepts=candidate_concepts
                            )

                        metric_name = 'AUROC' if pred_type == 'classification' else 'MAE'
                        llm_output = llm_chain.predict(
                            concept_order=str(concept_order), 
                            metric=metric_name, 
                            value=seed_test_metric,
                            candidate_concepts=str(candidate_concepts)
                        )

                        # Add Llama-2 prompt tags to LLM response
                        if llm_model.startswith('llama-2'):
                            llm_chain.memory.chat_memory.messages[-1].content += ' </s><s>[INST] '

                        str_scorer = lambda x: fuzz.ratio(x, llm_output)
                        scores = list(map(str_scorer, candidate_concepts))
                        new_concept = candidate_concepts[np.argmax(scores)]

                        if new_concept not in concept_order:
                            concept_order.append(new_concept)
                            new_concept_added = True

                            if args.verbose:
                                print(f'Added new concept: {new_concept}')

                            # Update the datasets
                            train_dataset = load_dataset(
                                args, 'train', 'all', None, seed, llm_model=llm_model, llm_decoding=llm_decoding,
                                filtered_concepts=concept_order
                            )
                            test_dataset = load_dataset(
                                args, 'test', 'all', None, seed, llm_model=llm_model, llm_decoding=llm_decoding,
                                filtered_concepts=concept_order
                            )
                            normalizer = train_dataset.get_normalizer()

                        else:
                            if args.verbose:
                                print('Duplicate concept selected. Reprompting.')

                            new_concept_added = False
                            retry_message = f'I already selected "{new_concept}". ' \
                                'Only choose from the list of candidate features.'
                            
                            if llm_model.startswith('llama-2'):
                                retry_message += ' [/INST] '

                            llm_chain.memory.chat_memory.messages.append(HumanMessage(content=retry_message))

                all_fracs.append(seed_fracs)
                all_test_metrics.append(seed_test_metrics)

                if not seq_prev_exists:
                    concept_orders.append(concept_order)

            all_avg_fracs = np.mean(all_fracs, axis=0)
            all_avgs = np.mean(all_test_metrics, axis=0)
            all_stds = np.std(all_test_metrics, axis=0)

            # Take the results at the prespecified ratios
            avg_fracs, avgs, stds = [], [], []
            concepts = selection.utils.load_concepts(dataset=dataset, pred=pred)

            if len(concepts) < 10:
                avg_fracs = all_avg_fracs
                avgs = all_avgs
                stds = all_stds
            else:
                for ratio in DATASET_TO_FSCONFIG[dataset][fs_basename]:
                    try:
                        ratio_idxs = np.where(all_avg_fracs <= ratio)
                        avg_fracs.append(all_avg_fracs[ratio_idxs][-1])
                        avgs.append(all_avgs[ratio_idxs][-1])
                        stds.append(all_stds[ratio_idxs][-1])
                    except:
                        continue

            # Save the feature selection orders
            if not seq_prev_exists:
                if 'empty' in fs_basename:
                    seq_results_path = osp.join(results_dir, f'{llm_model}_forward-seq-empty.yaml')
                else:
                    seq_results_path = osp.join(results_dir, f'{llm_model}_forward-seq.yaml')

                with open(seq_results_path, 'w') as fh:
                    yaml.dump(concept_orders, fh, default_flow_style=False)

        else:
            avg_fracs = []
            avgs = []
            stds = []

            fs_configs = DATASET_TO_FSCONFIG[dataset][fs_basename]

            for fs_config in fs_configs:
                fracs = []
                test_metrics = []

                for seed in args.seeds:
                    # Load dataset
                    train_dataset = load_dataset(
                        args, 'train', select, fs_config, seed, llm_model=llm_model, llm_decoding=llm_decoding,
                        context_config=context_config, example_config=example_config
                    )
                    test_dataset = load_dataset(
                        args, 'test', select, fs_config, seed, llm_model=llm_model, llm_decoding=llm_decoding,
                        context_config=context_config, example_config=example_config
                    )
                    normalizer = train_dataset.get_normalizer()

                    # Define model and set up pipeline
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

                    # Default pipeline: Normalize and fit model
                    pipe_config = [('normalizer', normalizer), ('clf', model)]
                    
                    # Add LASSO feature selection
                    if fs_basename == 'lasso':
                        if pred_type == 'classification':
                            lasso_model = LogisticRegression(
                                C=fs_config, class_weight='balanced', penalty='l1', solver='saga'
                            )

                        elif pred_type == 'regression':
                            lasso_model = Lasso(alpha=fs_config)

                        pipe_config.insert(1, ('lasso', SelectFromModel(lasso_model)))

                    # Add filtering based on mutual information
                    elif fs_basename == 'filter-mi':
                        if pred_type == 'classification':
                            mutual_info_f = mutual_info_classif
                        
                        elif pred_type == 'regression':
                            mutual_info_f = mutual_info_regression

                        pipe_config.insert(
                            1, ('filter', SelectPercentile(mutual_info_f, percentile=int(fs_config * 100)))
                        )

                    # Grid Search CV
                    pipe = Pipeline(pipe_config)
                    grid = GridSearchCV(
                        pipe, param_space, verbose=args.verbose, cv=cv, scoring=scoring, refit=True, n_jobs=64
                    ).fit(train_dataset.X, train_dataset.Y)
                    
                    best_pipe = grid.best_estimator_
                    weights = best_pipe.named_steps['clf'].coef_
                    fracs.append(weights.size / float(DATAPRED_TO_DIMS[args.datapred])) 
                    
                    # Compute the test metric
                    if pred_type == 'classification':
                        Y_prob = best_pipe.predict_proba(test_dataset.X)[:,-1]
                        test_metrics.append(roc_auc_score(test_dataset.Y, Y_prob))
                    
                    elif pred_type == 'regression':
                        Y_pred = best_pipe.predict(test_dataset.X)
                        test_metrics.append(mean_absolute_error(test_dataset.Y, Y_pred))

                avg_fracs.append(np.mean(fracs))
                avgs.append(np.mean(test_metrics))
                stds.append(np.std(test_metrics))
        
        avg_fracs, avgs, stds = np.array(avg_fracs), np.array(avgs), np.array(stds)
        results[fs_method] = [avg_fracs, avgs, stds]
        
        if ray.is_initialized():
            ray.shutdown()
    
    if args.verbose:
        pprint(results)

    if args.exp_suffix is not None:
        results_path = osp.join(results_dir, f'{pred}_results_{args.exp_suffix}.pkl')
    else:
        results_path = osp.join(results_dir, f'{pred}_results.pkl')

    # Update/append results if other results already exist
    if osp.exists(results_path):
        # Load previous results
        with open(results_path, 'rb') as fh:
            prev_results = pickle.load(fh)

        results = prev_results | results

    with open(results_path, 'wb') as fh:
        pickle.dump(results, fh)

    print(f'\nElapsed: {str(datetime.now() - start_time)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapred', help='Dataset + prediction task pair', default='mimic-icd/hf')
    parser.add_argument(
        '--fs_methods', 
        help='Feature selection methods to compare', 
        nargs='*',
        default=[
            # Baseline feature selection methods
            'lassonet',
            'lasso', 
            'random-concept', 
            'forward-seq',
            'backward-seq',
            'rfe',
            'filter-mi',
            'mrmr',

            # GPT-4
            'llm-forward-seq-empty:gpt-4-0613:greedy::',
            'llm-forward-seq:gpt-4-0613:greedy::',
            'llm-rank:gpt-4-0613:greedy::',
            'llm-ratio:gpt-4-0613:greedy::',
            'llm-ratio:gpt-4-0613:greedy::examples',
            'llm-ratio:gpt-4-0613:greedy::expls',
            'llm-ratio:gpt-4-0613:greedy:context:',
            'llm-ratio:gpt-4-0613:greedy:context:examples',
            'llm-ratio:gpt-4-0613:greedy:context:expls',
            'llm-ratio:gpt-4-0613:consistent::',
            'llm-ratio:gpt-4-0613:consistent::examples',
            'llm-ratio:gpt-4-0613:consistent::expls',
            'llm-ratio:gpt-4-0613:consistent:context:',
            'llm-ratio:gpt-4-0613:consistent:context:examples',
            'llm-ratio:gpt-4-0613:consistent:context:expls',

            # GPT-3.5
            'llm-forward-seq-empty:gpt-3.5-turbo:greedy::',
            'llm-forward-seq:gpt-3.5-turbo:greedy::',
            'llm-rank:gpt-3.5-turbo:greedy::',
            'llm-ratio:gpt-3.5-turbo:greedy::',
            'llm-ratio:gpt-3.5-turbo:greedy::examples',
            'llm-ratio:gpt-3.5-turbo:greedy::expls',
            'llm-ratio:gpt-3.5-turbo:greedy:context:',
            'llm-ratio:gpt-3.5-turbo:greedy:context:examples',
            'llm-ratio:gpt-3.5-turbo:greedy:context:expls',
            'llm-ratio:gpt-3.5-turbo:consistent::',
            'llm-ratio:gpt-3.5-turbo:consistent::examples',
            'llm-ratio:gpt-3.5-turbo:consistent::expls',
            'llm-ratio:gpt-3.5-turbo:consistent:context:',
            'llm-ratio:gpt-3.5-turbo:consistent:context:examples',
            'llm-ratio:gpt-3.5-turbo:consistent:context:expls',

            # Llama-2-70B
            'llm-forward-seq-empty:llama-2-70b-chat:greedy::',
            'llm-forward-seq:llama-2-70b-chat:greedy::', 
            'llm-rank:llama-2-70b-chat:greedy::',
            'llm-ratio:llama-2-70b-chat:greedy::',
            'llm-ratio:llama-2-70b-chat:greedy::examples',
            'llm-ratio:llama-2-70b-chat:greedy::expls',
            'llm-ratio:llama-2-70b-chat:greedy:context:',
            'llm-ratio:llama-2-70b-chat:greedy:context:examples',
            'llm-ratio:llama-2-70b-chat:greedy:context:expls',
            'llm-ratio:llama-2-70b-chat:consistent::',
            'llm-ratio:llama-2-70b-chat:consistent::examples',
            'llm-ratio:llama-2-70b-chat:consistent::expls',
            'llm-ratio:llama-2-70b-chat:consistent:context:',
            'llm-ratio:llama-2-70b-chat:consistent:context:examples',
            'llm-ratio:llama-2-70b-chat:consistent:context:expls',

            # Llama-2-13B
            'llm-forward-seq-empty:llama-2-13b-chat:greedy::',
            'llm-forward-seq:llama-2-13b-chat:greedy::',
            'llm-rank:llama-2-13b-chat:greedy::',
            'llm-ratio:llama-2-13b-chat:greedy::',
            'llm-ratio:llama-2-13b-chat:greedy::examples',
            'llm-ratio:llama-2-13b-chat:greedy::expls',
            'llm-ratio:llama-2-13b-chat:greedy:context:',
            'llm-ratio:llama-2-13b-chat:greedy:context:examples',
            'llm-ratio:llama-2-13b-chat:greedy:context:expls',
            'llm-ratio:llama-2-13b-chat:consistent::',
            'llm-ratio:llama-2-13b-chat:consistent::examples',
            'llm-ratio:llama-2-13b-chat:consistent::expls',
            'llm-ratio:llama-2-13b-chat:consistent:context:',
            'llm-ratio:llama-2-13b-chat:consistent:context:examples',
            'llm-ratio:llama-2-13b-chat:consistent:context:expls',
            
            # Llama-2-7B
            'llm-forward-seq-empty:llama-2-7b-chat:greedy::',
            'llm-forward-seq:llama-2-7b-chat:greedy::',
            'llm-rank:llama-2-7b-chat:greedy::',
            'llm-ratio:llama-2-7b-chat:greedy::',
            'llm-ratio:llama-2-7b-chat:greedy::examples',
            'llm-ratio:llama-2-7b-chat:greedy::expls',
            'llm-ratio:llama-2-7b-chat:greedy:context:',
            'llm-ratio:llama-2-7b-chat:greedy:context:examples',
            'llm-ratio:llama-2-7b-chat:greedy:context:expls',
            'llm-ratio:llama-2-7b-chat:consistent::',
            'llm-ratio:llama-2-7b-chat:consistent::examples',
            'llm-ratio:llama-2-7b-chat:consistent::expls',
            'llm-ratio:llama-2-7b-chat:consistent:context:',
            'llm-ratio:llama-2-7b-chat:consistent:context:examples',
            'llm-ratio:llama-2-7b-chat:consistent:context:expls'
        ],
        type=str
    )
    parser.add_argument('--seeds', help='Random seeds', nargs='*', default=[1,2,3,4,5], type=int)
    parser.add_argument('--outdir', help='Output directory', default='./results/linear_compare', type=str)
    parser.add_argument('--verbose', help='Verbosity', default=False, action='store_true')
    parser.add_argument('--buffer_chat', help='Only track the last K human-LLM interactions', default=False, action='store_true')
    parser.add_argument('--chat_buffer_size', help='Number of interactions to keep in buffer', default=2, type=int)
    parser.add_argument(
        '--buffer_concepts', 
        help='Only select from concepts with highest scores',
        default=False,
        action='store_true'
    )
    parser.add_argument(
        '--concept_buffer_size',
        help='Max number of highest-scoring candidate concepts to consider',
        default=100,
        type=int
    )
    parser.add_argument('--exp_suffix', help='Optional suffix to add to results file', default=None, type=str)
    args = parser.parse_args()
    
    main(args)
import os
import os.path as osp
import argparse
import sys
sys.path.append('../../')

import numpy as np
from datetime import datetime
import pickle
from collections import defaultdict

from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

import torch
from lassonet import LassoNetClassifier

import llm_select.datasets as datasets

DEFAULT_SEED = 42

N_TO_SELECT = {
    # MIMIC HF
    'mimic-icd/hf/0.3': 812,
    
    # MIMIC CKD
    'mimic-icd/ckd/0.3': 788,
    
    # MIMIC COPD
    'mimic-icd/copd/0.3': 860,
    
    # ACS Income
    'acs/income/0.3': 2026,

    # ACS Employment
    'acs/employment/0.3': 499,

    # ACS Public Coverage
    'acs/public_coverage/0.3': 2042,

    # ACS Mobility
    'acs/mobility/0.3': 1853,
}

def load_dataset(
    datapred='mimic-icd/ckd', 
    split='train', 
    seed=DEFAULT_SEED, 
    normalize=True,
    split_train_val=True,
    flatten=True,
    L=24, # For MIMIC datasets
    bin_size=4, # For MIMIC datasets
    **kwargs
):
    '''Data loading function.'''

    dataset, pred = datapred.split('/')

    base_data_config = dict(
        select='all',
        split=split,
        threshold=None,
        seed=seed,
        stratify_split=True,
        split_train_val=split_train_val,
        normalize=normalize,
        return_tensor=True,
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

def main(args):
    start_time = datetime.now()
    device = f'cuda:{args.device_idx}' if torch.cuda.is_available() else 'cpu'

    for datapred in args.datapreds:
        print(f'Computing LassoNet regularization paths for {datapred}...')
        dataset, pred = datapred.split('/')
        os.makedirs(osp.join(args.outdir, dataset), exist_ok=True)

        for seed in args.seeds:
            train_dataset = load_dataset(datapred, 'train', seed, verbose=args.verbose)
            val_dataset = load_dataset(datapred, 'val', seed, verbose=args.verbose)
            test_dataset = load_dataset(datapred, 'test', seed, verbose=args.verbose)

            # Compute class weight
            class_weights = compute_class_weight(
                class_weight='balanced', 
                classes=np.unique(train_dataset.Y), 
                y=train_dataset.Y
            )

            # Get feature groups
            groups = defaultdict(list)
            for i, col in enumerate(train_dataset.columns):
                groups[col.split('_')[0]].append(i)
            groups = list(groups.values())

            # Fit LassoNet regularization path (with group LASSO)
            # NOTE: For n_iters, first number = # epochs for training the initial model without L1 regularization (lambda=0)
            # second number = # epochs for further training the model with a particular lambda value
            # NOTE: When lambda_seq=None, model.path() runs training until lambda hits infinity
            # NOTE: For every new lambda, we initialize the parameters to be whatever we learned from before (warm start)
            
            # Regularization strengths to sweep through
            lambda_seq = [
                10, 100, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 
                5500, 6000, 6500, 6600, 6700, 6800, 6900, 7000, 7100, 7200, 7300, 
                7400, 7500, 8000, 8500, 9000, 9500, 10000, 10500, 11000, 11500, 
                12000, 12500, 13000, 13500, 14000, 14500, 15000
            ]
            model = LassoNetClassifier(
                hidden_dims=(300,), # Adds a single hidden layer with 300 units
                M=10, 
                batch_size=(2048 if dataset == 'acs' else 512),
                verbose=args.verbose, 
                n_iters=(10,5),
                groups=groups, 
                patience=3, 
                lambda_seq=lambda_seq,
                torch_seed=seed,
                #path_multiplier=1.04, # Default is 1.02
                device=device,
                class_weight=class_weights
            )

            path = model.path(
                train_dataset.X, train_dataset.Y, X_val=val_dataset.X, y_val=val_dataset.Y
            )

            # Retrieve the feature indices corresponding to each ratio
            n_selected = []
            aurocs = []
            lambda_ = []
            masks = []

            for p in path:
                model.load(p.state_dict)
                prob = model.predict_proba(test_dataset.X)[:,-1]
                n_selected.append(p.selected.sum().cpu().numpy())
                aurocs.append(roc_auc_score(test_dataset.Y, prob.squeeze()))
                lambda_.append(p.lambda_)
                masks.append(p.selected.cpu().numpy())
            
            for ratio in args.ratios:
                n_to_select = N_TO_SELECT[f'{datapred}/{ratio}']
                n_selected_np = np.array(n_selected)
                n_selected_np = np.where(n_selected_np == 0, -np.inf, n_selected_np) # Exclude cases with no features
                path_idx = np.argmin(np.abs(n_selected_np - n_to_select))
                feature_mask = path[path_idx].selected.cpu().numpy()

                mask_path = osp.join(args.outdir, f'{dataset}/{pred}_mask_{ratio}_{seed}.pkl')
                with open(mask_path, 'wb') as fh:
                    pickle.dump(feature_mask, fh)

                print(f'\nFeature mask for {datapred}/{ratio} saved in: {mask_path}')
                print(f'lambda={lambda_[path_idx]}, n_features={np.sum(feature_mask)}\n')

            # Save regularization path
            reg = dict(n_selected=n_selected, aurocs=aurocs, lambda_=lambda_, masks=masks)
            reg_path = osp.join(args.outdir, f'{dataset}/{pred}_reg_{seed}.pkl')
            with open(reg_path, 'wb') as fh:
                pickle.dump(reg, fh)

            print(f'Regularization path for {datapred} saved in: {reg_path}')

    print(f'Elapsed: {str(datetime.now() - start_time)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapreds', help='Dataset + prediction task pairs', nargs='*', default=['mimic-icd/ckd'], type=str)
    parser.add_argument('--outdir', help='Output directory', default='./lassonet', type=str)
    parser.add_argument('--seeds', help='Seeds for reproducibility', nargs='*', default=[1,2,3,4,5], type=int)
    parser.add_argument('--ratios', help='Selection ratios', nargs='*', default=[0.1,0.3], type=float)
    parser.add_argument('--device_idx', help='CUDA device index', default=0, type=int)
    parser.add_argument('--verbose', help='Verbosity', default=False, action='store_true')
    args = parser.parse_args()

    main(args)
import os
import os.path as osp
import argparse
from tqdm import tqdm
import sys
sys.path.append('../../')

import numpy as np
from datetime import datetime
import pickle

from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

from group_lasso import LogisticGroupLasso
LogisticGroupLasso.LOG_LOSSES = True

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
    'acs/mobility/0.3': 1853
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
    
    for datapred in args.datapreds:
        print(f'Computing gLASSO regularization paths for {datapred}...')
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
            groups = []
            prev = None
            group_idx = 0
            for col in train_dataset.columns:
                current = col.split('_')[0]

                if prev is not None and current != prev:
                    group_idx += 1

                groups.append(group_idx)
                prev = current

            # Fit initial model
            model = LogisticGroupLasso(
                groups=groups,
                group_reg=0,
                l1_reg=0,
                scale_reg='inverse_group_size',
                subsampling_scheme=args.subsampling,
                random_state=seed,
                warm_start=True
            ).fit(train_dataset.X, train_dataset.Y)

            init_lambda_ = 0
            init_auroc = roc_auc_score(
                test_dataset.Y, model.predict_proba(test_dataset.X)[:,-1]
            )
            init_coef = model.coef_[:,1] - model.coef_[:,0]
            init_n_selected = np.count_nonzero(init_coef)
            init_mask = np.abs(init_coef) > 0

            # Fit group LASSO regularization path
            lambda_seq = [
                0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 
                #0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1
            ]
            lambda_, aurocs, n_selected, masks = [], [], [], []
            
            with tqdm(lambda_seq[::-1]) as tgl:
                tgl.set_description('Computing gLASSO path')
                for l in tgl:
                    model.group_reg = l
                    model.fit(train_dataset.X, train_dataset.Y)
                    prob = model.predict_proba(test_dataset.X)[:,-1]
                    auroc = roc_auc_score(test_dataset.Y, prob)
                    aurocs.append(auroc)

                    coef = model.coef_[:,1] - model.coef_[:,0]
                    mask = np.abs(coef) > 0
                    masks.append(mask)
                    n_selected.append(np.sum(mask))

                    tgl.set_postfix(lambda_=l, auroc=auroc, selected=np.sum(mask))
            
            lambda_ = lambda_seq[::-1] + [init_lambda_]
            aurocs.append(init_auroc)
            n_selected.append(init_n_selected)
            masks.append(init_mask)

            for ratio in args.ratios:
                n_to_select = N_TO_SELECT[f'{datapred}/{ratio}']
                n_selected_np = np.array(n_selected)
                n_selected_np = np.where(n_selected_np == 0, -np.inf, n_selected_np) # Exclude cases with no features
                path_idx = np.argmin(np.abs(n_selected_np - n_to_select))
                feature_mask = masks[path_idx]

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
    parser.add_argument('--outdir', help='Output directory', default='./glasso', type=str)
    parser.add_argument('--seeds', help='Seeds for reproducibility', nargs='*', default=[1,2,3,4,5], type=int)
    parser.add_argument('--ratios', help='Selection ratios', nargs='*', default=[0.1,0.3], type=float)
    parser.add_argument('--subsampling', help='Subsampling ratio for stochastic updates', default=0.1, type=float)
    parser.add_argument('--verbose', help='Verbosity', default=False, action='store_true')
    args = parser.parse_args()

    main(args)
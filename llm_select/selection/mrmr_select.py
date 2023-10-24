import os
import os.path as osp
import argparse
from tqdm import tqdm
import sys
sys.path.append('../../')

import numpy as np
import pandas as pd
from datetime import datetime
import pickle

from mrmr import mrmr_classif

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
    
    for datapred in args.datapreds:
        print(f'Selecting features with MRMR for {datapred}...')
        dataset, pred = datapred.split('/')
        os.makedirs(osp.join(args.outdir, dataset), exist_ok=True)

        for seed in args.seeds:
            train_dataset = load_dataset(datapred, 'train', seed, verbose=args.verbose)

            # Select features with MRMR
            max_ratio = args.ratios[np.argmax(args.ratios)]
            max_n_to_select = N_TO_SELECT[f'{datapred}/{max_ratio}']
            max_feature_mask = mrmr_classif(
                X=pd.DataFrame(train_dataset.X),
                y=train_dataset.Y,
                K=max_n_to_select
            )

            for ratio in args.ratios:
                n_to_select = N_TO_SELECT[f'{datapred}/{ratio}']
                feature_mask = sorted(max_feature_mask[:n_to_select])
                mask_path = osp.join(args.outdir, f'{dataset}/{pred}_mask_{ratio}_{seed}.pkl')
                with open(mask_path, 'wb') as fh:
                    pickle.dump(feature_mask, fh)

                print(f'\nFeature mask for {datapred}/{ratio} saved in: {mask_path}')
                print(f'n_features={len(feature_mask)}\n')

    print(f'Elapsed: {str(datetime.now() - start_time)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapreds', help='Dataset + prediction task pairs', nargs='*', default=['mimic-icd/ckd'], type=str)
    parser.add_argument('--outdir', help='Output directory', default='./mrmr', type=str)
    parser.add_argument('--seeds', help='Seeds for reproducibility', nargs='*', default=[1,2,3,4,5], type=int)
    parser.add_argument('--ratios', help='Selection ratios', nargs='*', default=[0.1,0.3], type=float)
    parser.add_argument('--verbose', help='Verbosity', default=False, action='store_true')
    args = parser.parse_args()

    main(args)
import os
import os.path as osp
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

import yaml
from mrmr import mrmr_classif
import scipy
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle

from sklearn.feature_selection import SelectPercentile, mutual_info_classif
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_california_housing, load_diabetes, load_breast_cancer

import torch
torch.set_default_dtype(torch.float32)

import folktables

import llm_select.selection as selection

ABS_DIR = osp.dirname(osp.abspath(__file__))
DATA_DIR = osp.join(ABS_DIR, '../data')
PROMPT_DIR = osp.join(ABS_DIR, '../prompts')
PROMPT_OUTDIR = osp.join(ABS_DIR, '../prompt_outputs')
DEFAULT_SEED = 42

# Set up one-hot/ordinal encoders for MIMIC
time_series_to_cat = {
    'capillary refill': ['abnormal', 'normal'],
    'left dorsalis pedis pulse': ['absent', 'faint', 'doppler', 'strong'],
    'right dorsalis pedis pulse': ['absent', 'faint', 'doppler', 'strong'],
    'left radial pulse': ['absent', 'faint', 'doppler', 'strong'],
    'right radial pulse': ['absent', 'faint', 'doppler', 'strong'],
    'LLE strength / sensation': ['absent', 'impaired', 'intact'],
    'LUE strength / sensation': ['absent', 'impaired', 'intact'],
    'RLE strength / sensation': ['absent', 'impaired', 'intact'],
    'RUE strength / sensation': ['absent', 'impaired', 'intact'],
    'speech': ['aphasic', 'garbled', 'intubated', 'mute', 'other', 'normal', 'slurred', 'dysphasic', 'dysarthric'],
    'cough / gag reflex': ['absent', 'impaired', 'intact'],
    'Braden activity scale': ['1: bedfast', '2: chairfast', '3: walks occasionally', '4: walks frequently'],
    'Braden friction & shear scale': ['1: problem', '2: potential problem', '3: no apparent problem'],
    'Braden mobility scale': ['1: completely immobile', '2: very limited', '3: slightly limited', '4: no limitations'],
    'Braden moisture scale': ['1: consistently moist', '2: moist', '3: occasionally moist', '4: rarely moist'],
    'Braden nutrition scale': ['1: very poor', '2: probably inadequate', '3: adequate', '4: excellent'],
    'Braden sensory perception scale': ['1: completely limited', '2: very limited', '3: slightly limited', '4: no impairment'],
    'Morse ambulatory aid': ['bedrest', 'cane', 'crutches', 'furniture', 'none', 'nurse assist', 'walker', 'wheel chair'],
    'Morse gait / transferring': ['weak', 'normal', 'bedrest', 'immobile', 'impaired'],
    'Morse history of falling': ['yes', 'no'],
    'Morse mental status': ['forgets limitations', 'oriented to own ability'],
    'Morse secondary diagnosis': ['yes', 'no'],
    'delirium assessment': ['yes', 'no'],#, 'unknown'],
    'Glasgow coma scale (GCS) - Eye': ['1', '2', '3', '4'],
    'Glasgow coma scale (GCS) - Verbal': ['1', '2', '3', '4', '5'],
    'Glasgow coma scale (GCS) - Motor': ['1', '2', '3', '4', '5', '6'],
    'level of assistance': ['supervision', 'total assist', 'maximal assist', 'moderate assist', 'minimal assist', 'independent']
}
time_series_to_cat = {k: np.array(v)[:,None] for k,v in time_series_to_cat.items()}
time_series_to_onehot = {k: OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(v) for k,v in time_series_to_cat.items()}
time_series_to_ordinal = {k: OrdinalEncoder().fit(v) for k,v in time_series_to_cat.items()}

static_to_cat = {
    'gender': ['Female', 'Male', 'Other/Unknown'],
    'ethnicity': ['Asian', 'Black', 'Hispanic', 'Native American', 'Other/Unknown', 'White'],
    'ICU unit type': [
        'Cardiac Intensive Care Unit (CICU)', 
        'Cardiac Vascular Intensive Care Unit (CVICU)',
        'Cardiothoracic Intensive Care Unit (CTICU)',
        'Cardiac Care Unit/Cardiothoracic Intensive Care Unit (CCU/CTICU)',
        'Coronary Care Unit (CCU)',
        'Medical Intensive Care Unit (MICU)',
        'Medical/Surgical Intensive Care Unit (MICU/SICU)',
        'Neuro Intensive Care Unit (Neuro ICU)',
        'Neuro Intermediate',
        'Neuro Stepdown',
        'Neuro Surgical Intensive Care Unit (Neuro SICU)',
        'Surgical Intensive Care Unit (SICU)',
        'Trauma SICU (TSICU)',
        'Cardiac Surgical Intensive Care Unit (CSICU)'
    ]
}
static_to_cat = {k: np.array(v)[:,None] for k,v in static_to_cat.items()}
static_to_onehot = {k: OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(v) for k,v in static_to_cat.items()}
static_to_ordinal = {k: OrdinalEncoder().fit(v) for k,v in static_to_cat.items()}

class FlatTransformer(BaseEstimator, TransformerMixin):
    '''Custom sklearn feature transformer for standardizing flattened time-series data.'''

    def __init__(self, X_idxs, S_idxs, T, D_X, D_S, X_scaler, S_scaler):
        self.X_idxs = X_idxs # Indices to standardize
        self.S_idxs = S_idxs # Indices to standardize
        self.T = T
        self.D_X = D_X
        self.D_S = D_S
        self.X_scaler = X_scaler
        self.S_scaler = S_scaler
        
        self.S_mean_ = None
        self.S_var_ = None
        self.S_n_samples_seen_ = None
        self.S_scale_ = None
        
        self.X_mean_ = None # [TN,D_X]
        self.X_var_ = None # []
        self.X_n_samples_seen_ = None
        self.X_scale_ = None
        
    def split_data(self, X, y=None):
        # Split static and time-dependent
        S = X[:,:self.D_S] # [N, D_S]

        if X.shape[-1] == self.D_S:
            X_t = X[:,self.D_S:] # When we have no time-series features
        else:
            X_t = np.reshape(X[:,self.D_S:], (-1,self.T,self.D_X)) # [N,TD_X] -> [N,T,D_X]
        
        return S, X_t
        
    # X: [N, D_S + T*D_X)]
    def fit(self, X, y=None):
        S, X_t = self.split_data(X, y=y)
        
        # Fit on the static features
        if self.S_idxs.size > 0:
            self.S_scaler.fit(S[:,self.S_idxs], y=y)
            self.S_mean_ = self.S_scaler.mean_
            self.S_var_ = self.S_scaler.var_
            self.S_n_samples_seen_ = self.S_scaler.n_samples_seen_
            self.S_scale_ = self.S_scaler.scale_
        
        # Fit on the time-dependent features
        if self.X_idxs.size > 0:
            X_t = np.transpose(X_t, axes=[1,0,2]) # [T,N,D_X]
            X_t = np.reshape(X_t, (-1,self.D_X)) # [TN,D_X], T blocks of size N
            self.X_scaler.fit(X_t[:,self.X_idxs], y=y)
            self.X_mean_ = self.X_scaler.mean_
            self.X_var_ = self.X_scaler.var_
            self.X_n_samples_seen_ = self.X_scaler.n_samples_seen_
            self.X_scale_ = self.X_scaler.scale_
        
        return self
    
    def transform(self, X, y=None):
        S, X_t = self.split_data(X, y=y)
        
        # Transform the static features
        if self.S_idxs.size > 0:
            S[:,self.S_idxs] -= self.S_mean_
            S[:,self.S_idxs] /= self.S_scale_
        
        # Tranform the time-dependent features
        if self.X_idxs.size > 0:
            X_t = np.transpose(X_t, axes=[1,0,2]) # [T,N,D_X]
            X_t = np.reshape(X_t, (-1,self.D_X)) # [TN,D_X], T blocks of size N
            X_t[:,self.X_idxs] -= self.X_mean_
            X_t[:,self.X_idxs] /= self.X_scale_
            X_t = np.reshape(X_t, (self.T,-1,self.D_X)) # [T,N,D_X]
            X_t = np.transpose(X_t, axes=[1,0,2])
            X_t = np.reshape(X_t, (-1,self.T*self.D_X)) # [N,TD_X]
        
        # Need to recombine the static and time-dependent features
        X = np.concatenate([S, X_t], axis=-1)

        # Substitute NaNs with 0
        X = np.nan_to_num(X)
        
        return X

class TimeSeriesTransformer(BaseEstimator, TransformerMixin):
    '''Custom sklearn feature transformer for standardizing time-series data.'''
    
    def __init__(self, X_idxs, S_idxs, T, D_X, D_S, X_scaler, S_scaler):
        self.X_idxs = X_idxs # Indices to standardize
        self.S_idxs = S_idxs # Indices to standardize
        self.T = T
        self.D_X = D_X
        self.D_S = D_S
        self.X_scaler = X_scaler
        self.S_scaler = S_scaler
        
        self.S_mean_ = None
        self.S_var_ = None
        self.S_n_samples_seen_ = None
        self.S_scale_ = None
        
        self.X_mean_ = None # [TN,D_X]
        self.X_var_ = None # []
        self.X_n_samples_seen_ = None
        self.X_scale_ = None
        
    def split_data(self, X, y=None):
        # Split time-invariant and time-dependent
        S = X[:,0,:self.D_S] # [N, D_S]
        
        if X.shape[-1] == self.D_S:
            X_t = X[:,:,self.D_S:] # [N,T,0]
        else:
            X_t = np.reshape(X[:,:,self.D_S:], (-1,self.D_X)) # [N,T,D_X] -> [NT,D_X]

        return S, X_t
    
    # X: [N, D_S + T*D_X)]
    def fit(self, X, y=None):
        S, X_t = self.split_data(X, y=y) # [N,D_S], [NT,D_X]
        
        # Fit on the time-invariant features
        if len(self.S_idxs) > 0:
            self.S_scaler.fit(S[:,self.S_idxs], y=y)
            self.S_mean_ = self.S_scaler.mean_
            self.S_var_ = self.S_scaler.var_
            self.S_n_samples_seen_ = self.S_scaler.n_samples_seen_
            self.S_scale_ = self.S_scaler.scale_
        
        # Fit on the time-dependent features
        if len(self.X_idxs) > 0:
            self.X_scaler.fit(X_t[:,self.X_idxs], y=y)
            self.X_mean_ = self.X_scaler.mean_
            self.X_var_ = self.X_scaler.var_
            self.X_n_samples_seen_ = self.X_scaler.n_samples_seen_
            self.X_scale_ = self.X_scaler.scale_
        
        return self
    
    def transform(self, X, y=None):
        S, X_t = self.split_data(X, y=y) # [N,D_S], [NT,D_X]
        
        # Transform the time-invariant features
        if len(self.S_idxs) > 0:
            S[:,self.S_idxs] -= self.S_mean_
            S[:,self.S_idxs] /= self.S_scale_
        
        S = np.repeat(S[:,None,:], self.T, axis=1)

        # Tranform the time-dependent features
        if len(self.X_idxs) > 0:
            X_t[:,self.X_idxs] -= self.X_mean_
            X_t[:,self.X_idxs] /= self.X_scale_
            X_t = np.reshape(X_t, (-1,self.T,self.D_X)) # [N,T,D_X]

        # Need to recombine the time-invariant and time-dependent features
        X = np.concatenate([S, X_t], axis=-1) # [N,T,D_S+D_X]
        
        # Substitute NaNs with 0
        X = np.nan_to_num(X)

        return X

class BaseDataset(torch.utils.data.Dataset):
    '''Base dataset class for classification datasets.'''

    def __init__(
        self,
        dataset='',
        pred='',
        select='all',
        split='train',
        split_train_val=True,
        llm_model='gpt-4-0613',
        llm_decoding='greedy',
        with_context=False,
        with_examples=False,
        with_expls=False,
        ratio=None,
        threshold=None,
        topk=None,
        seed=DEFAULT_SEED,
        normalize=True,
        stratify_split=True, # Classification only
        return_tensor=True,
        filtered_concepts=None,
        verbose=False,
        **kwargs
    ):
        self.data_dir = osp.join(DATA_DIR, dataset)
        self.prompt_dir = osp.join(PROMPT_DIR, dataset)
        self.prompt_outdir = osp.join(PROMPT_OUTDIR, f'{dataset}/{llm_model}')

        self.dataset = dataset
        self.pred = pred
        self.select = select
        self.split = split
        self.split_train_val = split_train_val
        self.llm_model = llm_model
        self.llm_decoding = llm_decoding
        self.with_context = with_context
        self.with_examples = with_examples
        self.with_expls = with_expls
        self.ratio = ratio
        self.threshold = threshold
        self.topk = topk
        self.seed = seed
        self.normalize = normalize
        self.stratify_split = stratify_split
        self.return_tensor = return_tensor
        self.filtered_concepts = filtered_concepts
        self.verbose = verbose

        if self.filtered_concepts is not None and self.verbose:
            print('[Warning] Using the prespecified concepts. '\
                  'Other concept selection options are ignored.\n')
    
    def select_concepts(self, concepts):
        # Preload the LLM prompt outputs
        if self.select.startswith('llm-rank'):
            llm_outpath = osp.join(self.prompt_outdir, f'{self.pred}_rank.yaml')
            with open(llm_outpath, 'r') as fh:
                llm_output = yaml.load(fh, Loader=yaml.Loader)

        elif self.select.startswith('llm'):
            llm_outname = osp.join(self.prompt_outdir, f'{self.pred}_{self.llm_decoding}')
            
            # Fetch LLM output from prompt with context
            if self.with_context:
                llm_outname += '_context'

            # Fetch LLM output from prompt with examples + explanations
            if self.with_expls:
                llm_outname += '_expls'

            # Fetch LLM output from prompt with examples
            elif self.with_examples:
                llm_outname += '_examples'

            llm_outpath = f'{llm_outname}.yaml'
            
            with open(llm_outpath, 'r') as fh:
                llm_output = yaml.load(fh, Loader=yaml.Loader)

        # Randomly select features at the concept level
        if self.select == 'random-concept':
            np.random.seed(self.seed)
            selected_concepts = np.random.choice(
                concepts,
                size=int(self.ratio * len(concepts)),
                replace=False
            ).tolist()

        # LLM concept selection by threshold
        elif self.select == 'llm-threshold':
            selected_concepts = selection.utils.llm_select_concepts(
                llm_output, threshold=self.threshold, filtered_concepts=concepts, seed=self.seed
            )

        # LLM concept selection by fraction
        elif self.select == 'llm-ratio':
            selected_concepts = selection.utils.llm_select_concepts(
                llm_output, ratio=self.ratio, filtered_concepts=concepts, seed=self.seed
            )

        # LLM concept selection by number of elements
        elif self.select == 'llm-topk':
            selected_concepts = selection.utils.llm_select_concepts(
                llm_output, topk=self.topk, filtered_concepts=concepts, seed=self.seed
            )

        # LLM concept selection by direct ranking (Default: by ratio)
        elif self.select == 'llm-rank':
            selected_concepts = selection.utils.llm_select_concepts(
                llm_output, ratio=self.ratio, filtered_concepts=concepts, seed=self.seed, rank=True
            )

        else:
            raise RuntimeError(f'[Error] Invalid concept selection mode provided: {self.select}.')
        
        return selected_concepts
    
    def select_features(self, X_train, Y_train, Xs):
        # Random feature selection
        if self.select == 'random-feature':
            np.random.seed(self.seed)
            idxs = np.random.choice(
                np.arange(0, X_train.shape[-1]),
                size=int(self.ratio * X_train.shape[-1]),
                replace=False
            ).tolist()

            self.columns = self.columns[idxs]

            X_train = X_train[...,idxs]

            if isinstance(Xs, list) or isinstance(Xs, tuple):
                new_Xs = [X[...,idxs] for X in Xs]
            else:
                new_Xs = Xs[...,idxs]

        elif self.select == 'filter-mi':
            mi_filter = SelectPercentile(
                mutual_info_classif, percentile=int(self.ratio * 100)
            )
            X_train = mi_filter.fit_transform(X_train, Y_train)

            if isinstance(Xs, list) or isinstance(Xs, tuple):
                new_Xs = [mi_filter.transform(X) for X in Xs]
            else:
                new_Xs = mi_filter.transform(Xs)

            self.columns = self.columns[mi_filter.get_support()]

        elif self.select == 'mrmr':
            if self.get_pred_type() == 'classification':
                mrmr_selector = mrmr_classif
            elif self.get_pred_type() == 'regression':
                mrmr_selector = mrmr_classif

            # Select features using MRMR
            idxs = mrmr_selector(X=pd.DataFrame(X_train), y=pd.DataFrame(Y_train), K=int(self.ratio * X_train.shape[-1]))
            X_train = X_train[...,idxs]
            
            if isinstance(Xs, list) or isinstance(Xs, tuple):
                new_Xs = [X[...,idxs] for X in Xs]
            else:
                new_Xs = Xs[...,idxs]

        return (X_train, *new_Xs) if (isinstance(Xs, list) or isinstance(Xs, tuple)) else (X_train, new_Xs)
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        if self.return_tensor:
            batch = (
                torch.Tensor(self.X[idx]),
                torch.Tensor(self.Y[:,None][idx])
            )
        else:
            batch = (self.X[idx], self.Y[idx])

        return batch

class NumericMixin(object):
    '''Base class for datasets with only numeric features.'''

    def __init__(self):
        pass

    def get_normalizer(self):
        normalizer = StandardScaler()

        return normalizer
    
class CategoricalMixin(object):
    '''Base class for datasets with categorical (and numeric) features.'''

    def __init__(self):
        pass

    def get_normalizer(self):
        if len(self.cont_idxs) > 0 and len(self.cat_idxs) > 0:
            normalizer = ColumnTransformer([
                ('z-score', StandardScaler(), self.cont_idxs),
                ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), self.cat_idxs)
            ])
        elif len(self.cont_idxs) > 0:
            normalizer = StandardScaler()
        else:
            normalizer = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

        return normalizer

class CarsDataset(BaseDataset, CategoricalMixin):
    '''Used car price prediction dataset (Kaggle).'''

    def __init__(self, **kwargs):
        super(CarsDataset, self).__init__(dataset='cars', pred='price', **kwargs)
        self.load()

    @classmethod
    def get_pred_type(cls):
        return 'regression'
    
    def load(self):
        # Load the dataset
        # Kaggle: https://www.kaggle.com/datasets/vijayaadithyanvg/car-price-predictionused-cars
        df = pd.read_csv(osp.join(self.data_dir, 'cars.csv'))
        df['Age'] = 2023 - df['Year']
        df.drop(columns=['Car_Name','Year'], inplace=True)
        concept_map = dict(
            Present_Price='Present price',
            Driven_kms='Driven kilometers',
            Fuel_Type='Fuel type',
            Selling_type='Selling type (dealer/individual)',
            Transmission='Transmission (manual/automatic)',
            Owner='Number of previous owners',
            Age='Age'
        )
        concepts = list(concept_map.values())
        df.rename(columns=concept_map, inplace=True)
        
        all_cont_concepts = [
            'Present price', 
            'Driven kilometers', 
            'Number of previous owners',
            'Age'
        ]
        all_cat_concepts = [
            'Fuel type', 
            'Selling type (dealer/individual)', 
            'Transmission (manual/automatic)'
        ]

        # Choose subset of feature concepts to use
        if self.filtered_concepts is not None:
            selected_concepts = self.filtered_concepts
        elif self.select == 'random-concept' or self.select.startswith('llm'):
            selected_concepts = self.select_concepts(concepts)
        else:
            selected_concepts = concepts

        # Check number of concepts selected
        if self.verbose:
            print(f'Concepts: {len(selected_concepts)}/{len(concepts)} selected.')

        if len(selected_concepts) == 0:
            raise RuntimeError('[Error] No concepts selected with provided setting!')

        Y = df['Selling_Price'].to_numpy().astype(float)
        selected_df = df[selected_concepts]
        X_selected = selected_df.to_numpy()
        cont_concepts = []
        cont_idxs = []
        cat_concepts = []

        for i, concept in enumerate(selected_concepts):
            if concept in all_cont_concepts:
                cont_idxs.append(i)
                cont_concepts.append(concept)
            elif concept in all_cat_concepts:
                cat_concepts.append(concept)

        self.cont_idxs = cont_idxs
        self.cat_idxs = [i for i in range(len(selected_concepts)) if i not in cont_idxs]

        # Generate the train-val-test split
        X_train, X_test, Y_train, Y_test = train_test_split(
            X_selected, Y, test_size=0.2, random_state=DEFAULT_SEED
        )

        if self.split_train_val:
            X_train, X_val, Y_train, Y_val = train_test_split(
                X_train, Y_train, test_size=0.2, random_state=self.seed
            )

        # Standardize the features according to the training data
        if self.normalize:
            normalizer = self.get_normalizer()
            X_train = normalizer.fit_transform(X_train)
            X_test = normalizer.transform(X_test)

            if self.split_train_val:
                X_val = normalizer.transform(X_val)

            # Gather column names
            self.columns = cont_concepts
            for i, concept in enumerate(cat_concepts):
                if isinstance(normalizer, ColumnTransformer):
                    categories = normalizer.named_transformers_['onehot'].categories_
                elif isinstance(normalizer, (OneHotEncoder, OrdinalEncoder)):
                    categories = normalizer.categories_
                
                self.columns += [f'{concept}_{cat}' for cat in categories[i]]
        else:
            self.columns = selected_concepts

        # Feature selection at the feature level
        if self.select in ['random-feature', 'filter-mi', 'mrmr']:
            if self.split_train_val:
                X_train, X_test = self.select_features(X_train, Y_train, X_test)
            else:
                X_train, X_val, X_test = self.select_features(X_train, Y_train, (X_val, X_test))

        # Store the relevant split
        if self.split == 'train':
            self.X = X_train
            self.Y = Y_train

        elif self.split == 'val':
            self.X = X_val
            self.Y = Y_val
        
        elif self.split == 'test':
            self.X = X_test
            self.Y = Y_test

        elif self.split == 'all':
            if self.split_train_val:
                self.X = np.concatenate([X_train, X_val, X_test], axis=0)
                self.Y = np.concatenate([Y_train, Y_val, Y_test], axis=0)
            else:
                self.X = np.concatenate([X_train, X_test], axis=0)
                self.Y = np.concatenate([Y_train, Y_test], axis=0)

        if self.verbose:
            print(f'X: {self.X.shape}, Y: {self.Y.shape}')

class PimaDataset(BaseDataset, NumericMixin):
    '''Pima Indians diabetes dataset (Smith et al., 1988).'''

    def __init__(self, **kwargs):
        super(PimaDataset, self).__init__(dataset='pima', pred='diabetes', **kwargs)
        self.load()

    @classmethod
    def get_pred_type(cls):
        return 'classification'

    def load(self):
        # Load the dataset
        df = pd.read_csv(osp.join(self.data_dir, 'diabetes.csv'))
        concept_map = dict(
            Pregnancies='Number of times pregnant',
            Glucose='Plasma glucose concentration at 2 hours in an oral glucose tolerance test',
            BloodPressure='Diastolic blood pressure (mmHg)',
            SkinThickness='Triceps skin fold thickness (mm)',
            Insulin='2-hour serum insulin (muU/ml)',
            BMI='Body mass index (weight in kg/(height in m)^2)',
            DiabetesPedigreeFunction='Diabetes pedigree function',
            Age='Age'
        )
        concepts = list(concept_map.values())
        df.rename(columns=concept_map, inplace=True)

        # Choose subset of feature concepts to use
        if self.filtered_concepts is not None:
            selected_concepts = self.filtered_concepts
        elif self.select == 'random-concept' or self.select.startswith('llm'):
            selected_concepts = self.select_concepts(concepts)
        else:
            selected_concepts = concepts

        # Check number of concepts selected
        if self.verbose:
            print(f'Concepts: {len(selected_concepts)}/{len(concepts)} selected.')

        if len(selected_concepts) == 0:
            raise RuntimeError('[Error] No concepts selected with provided setting!')

        Y = df['Outcome'].to_numpy().astype(float)
        selected_df = df[selected_concepts]
        X_selected = selected_df.to_numpy()

        # Generate the train-val-test split
        if self.stratify_split:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=DEFAULT_SEED)
            train_idx, test_idx = next(sss.split(X_selected, Y))
            X_train, Y_train = X_selected[train_idx], Y[train_idx]
            X_test, Y_test = X_selected[test_idx], Y[test_idx]

            if self.split_train_val:
                sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=self.seed)
                train_idx, val_idx = next(sss.split(X_train, Y_train))
                X_val, Y_val = X_train[val_idx], Y_train[val_idx]
                X_train, Y_train = X_train[train_idx], Y_train[train_idx]
        else:
            X_train, X_test, Y_train, Y_test = train_test_split(
                X_selected, Y, test_size=0.2, random_state=DEFAULT_SEED
            )

            if self.split_train_val:
                X_train, X_val, Y_train, Y_val = train_test_split(
                    X_train, Y_train, test_size=0.2, random_state=self.seed
                )

        # Standardize the features according to the training data
        if self.normalize:
            normalizer = self.get_normalizer()
            X_train = normalizer.fit_transform(X_train)
            X_test = normalizer.transform(X_test)

            if self.split_train_val:
                X_val = normalizer.transform(X_val)
       
        self.columns = selected_concepts

        # Feature selection at the feature level
        if self.select in ['random-feature', 'filter-mi', 'mrmr']:
            if self.split_train_val:
                X_train, X_test = self.select_features(X_train, Y_train, X_test)
            else:
                X_train, X_val, X_test = self.select_features(X_train, Y_train, (X_val, X_test))

        # Fixed
        self.cont_idxs = np.arange(0,X_train.shape[-1])
        self.cat_idxs = []

        # Store the relevant split
        if self.split == 'train':
            self.X = X_train
            self.Y = Y_train

        elif self.split == 'val':
            self.X = X_val
            self.Y = Y_val
        
        elif self.split == 'test':
            self.X = X_test
            self.Y = Y_test

        elif self.split == 'all':
            if self.split_train_val:
                self.X = np.concatenate([X_train, X_val, X_test], axis=0)
                self.Y = np.concatenate([Y_train, Y_val, Y_test], axis=0)
            else:
                self.X = np.concatenate([X_train, X_test], axis=0)
                self.Y = np.concatenate([Y_train, Y_test], axis=0)

        if self.verbose:
            print(f'X: {self.X.shape}, Y: {self.Y.shape}')

class GiveMeCreditDataset(BaseDataset, NumericMixin):
    '''Kaggle `Give Me Some Credit` deliquency prediction dataset.'''

    def __init__(self, remove_outliers=False, balance_classes=False, **kwargs):
        super(GiveMeCreditDataset, self).__init__(dataset='give-me-credit', pred='delinquency', **kwargs)
        self.remove_outliers = remove_outliers
        self.balance_classes = balance_classes
        self.load()

    @classmethod
    def get_pred_type(cls):
        return 'classification'

    def load(self):
        # Load the dataset
        # Used source: https://github.com/dtak/ocbnn-public/
        # Preprocessing steps in: https://github.com/dtak/ocbnn-public/blob/master/data/dataloader.py#L155
        # Original Kaggle source: https://www.kaggle.com/c/GiveMeSomeCredit
        df = pd.read_csv(
            'https://raw.githubusercontent.com/dtak/ocbnn-public/master/data/give_me_some_credit.csv'
        )
        df.dropna(inplace=True) # Remove rows with NaN's
        df.drop(columns=[df.columns[0]], inplace=True) # First column = index

        # Optional: Remove outliers
        if self.remove_outliers:
            df = df[
                (df['NumberOfTime30-59DaysPastDueNotWorse'] <= np.quantile(df['NumberOfTime30-59DaysPastDueNotWorse'].to_numpy(), 0.95)) &
                (df['DebtRatio'] <= np.quantile(df['DebtRatio'].to_numpy(), 0.95)) &
                (df['MonthlyIncome'] <= np.quantile(df['MonthlyIncome'].to_numpy(), 0.95)) &
                (df['RevolvingUtilizationOfUnsecuredLines'] <= np.quantile(df['RevolvingUtilizationOfUnsecuredLines'].to_numpy(), 0.998)) &
                (df['NumberOfTimes90DaysLate'] <= np.quantile(df['NumberOfTimes90DaysLate'].to_numpy(), 0.998)) &
                (df['NumberRealEstateLoansOrLines'] <= np.quantile(df['NumberRealEstateLoansOrLines'].to_numpy(), 0.998)) &
                (df['NumberOfTime60-89DaysPastDueNotWorse'] <= np.quantile(df['NumberOfTime60-89DaysPastDueNotWorse'].to_numpy(), 0.998)) &
                (df['NumberOfDependents'] <= np.quantile(df['NumberOfDependents'].to_numpy(), 0.998))
            ].reset_index(drop=True)

        # Optional: Downsample negative examples
        if self.balance_classes:
            df_pos = df[df['SeriousDlqin2yrs'] == 1]
            n_pos = df_pos.shape[0]
            df_neg = df[df['SeriousDlqin2yrs'] == 0].sample(n=n_pos, random_state=DEFAULT_SEED)
            df = pd.concat([df_pos, df_neg], axis=0).sample(frac=1, random_state=DEFAULT_SEED).reset_index(drop=True)

        # All concepts are numerical
        concept_map = {
            'RevolvingUtilizationOfUnsecuredLines': 'Total balance on credit cards and personal lines of credit ' \
                '(except real estate and no installment debt like car loans, divided by monthly gross income)', 
            'age': 'Age',
            'DebtRatio': 'Monthly debt payments, alimony, and living costs, divided by monthly gross income',
            'MonthlyIncome': 'Monthly income',
            'NumberOfOpenCreditLinesAndLoans': 'Number of open loans and lines of credit', 
            'NumberRealEstateLoansOrLines': 'Number of mortgage and real estate loans, including home equity lines of credit', 
            'NumberOfTime30-59DaysPastDueNotWorse': 'Number of times borrower has been 30 to 59 days past due ' \
                '(but no worse) in the last 2 years', 
            'NumberOfTime60-89DaysPastDueNotWorse': 'Number of times borrower has been 60 to 89 days past due ' \
                '(but no worse) in the last 2 years',
            'NumberOfTimes90DaysLate': 'Number of times borrower has been 90 days or more past due in the last 2 years',
            'NumberOfDependents': 'Number of dependents in family, excluding themselves'
        }
        concepts = list(concept_map.values())
        df.rename(columns=concept_map, inplace=True)

        # Choose subset of feature concepts to use
        if self.filtered_concepts is not None:
            selected_concepts = self.filtered_concepts
        elif self.select == 'random-concept' or self.select.startswith('llm'):
            selected_concepts = self.select_concepts(concepts)
        else:
            selected_concepts = concepts

        # Check number of concepts selected
        if self.verbose:
            print(f'Concepts: {len(selected_concepts)}/{len(concepts)} selected.')

        if len(selected_concepts) == 0:
            raise RuntimeError('[Error] No concepts selected with provided setting!')

        Y = df['SeriousDlqin2yrs'].to_numpy().astype(float)
        selected_df = df[selected_concepts]
        X_selected = selected_df.to_numpy()

        # Generate the train-val-test split
        if self.stratify_split:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=DEFAULT_SEED)
            train_idx, test_idx = next(sss.split(X_selected, Y))
            X_train, Y_train = X_selected[train_idx], Y[train_idx]
            X_test, Y_test = X_selected[test_idx], Y[test_idx]

            if self.split_train_val:
                sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=self.seed)
                train_idx, val_idx = next(sss.split(X_train, Y_train))
                X_val, Y_val = X_train[val_idx], Y_train[val_idx]
                X_train, Y_train = X_train[train_idx], Y_train[train_idx]
        else:
            X_train, X_test, Y_train, Y_test = train_test_split(
                X_selected, Y, test_size=0.2, random_state=DEFAULT_SEED
            )

            if self.split_train_val:
                X_train, X_val, Y_train, Y_val = train_test_split(
                    X_train, Y_train, test_size=0.2, random_state=self.seed
                )

        # Standardize the features according to the training data
        if self.normalize:
            normalizer = self.get_normalizer()
            X_train = normalizer.fit_transform(X_train)
            X_test = normalizer.transform(X_test)

            if self.split_train_val:
                X_val = normalizer.transform(X_val)
       
        self.columns = selected_concepts

        # Feature selection at the feature level
        if self.select in ['random-feature', 'filter-mi', 'mrmr']:
            if self.split_train_val:
                X_train, X_test = self.select_features(X_train, Y_train, X_test)
            else:
                X_train, X_val, X_test = self.select_features(X_train, Y_train, (X_val, X_test))

        # Fixed
        self.cont_idxs = np.arange(0,X_train.shape[-1])
        self.cat_idxs = []

        # Store the relevant split
        if self.split == 'train':
            self.X = X_train
            self.Y = Y_train

        elif self.split == 'val':
            self.X = X_val
            self.Y = Y_val
        
        elif self.split == 'test':
            self.X = X_test
            self.Y = Y_test

        elif self.split == 'all':
            if self.split_train_val:
                self.X = np.concatenate([X_train, X_val, X_test], axis=0)
                self.Y = np.concatenate([Y_train, Y_val, Y_test], axis=0)
            else:
                self.X = np.concatenate([X_train, X_test], axis=0)
                self.Y = np.concatenate([Y_train, Y_test], axis=0)

        if self.verbose:
            print(f'X: {self.X.shape}, Y: {self.Y.shape}')

class COMPASDataset(BaseDataset, CategoricalMixin):
    '''COMPAS dataset (Larson et al., 2016).'''

    def __init__(self, **kwargs):
        super(COMPASDataset, self).__init__(dataset='compas', pred='recid', **kwargs)
        self.load()

    @classmethod
    def get_pred_type(cls):
        return 'classification'

    def load(self):
        # Load the dataset
        # Source: https://github.com/propublica/compas-analysis/
        # Preprocessing steps in: https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb
        df = pd.read_csv(
            'https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv'
        )
        df = df.loc[
            (df['days_b_screening_arrest'] <= 30) &
            (df['days_b_screening_arrest'] >= -30) &
            (df['is_recid'] != -1) &
            (df['c_charge_degree'] != 'O') &
            (df['score_text'] != 'N/A')
        ].reset_index(drop=True)

        # Add jail length-of-stay (in days) column
        df['c_jail_in'] = pd.to_datetime(df['c_jail_in'])
        df['c_jail_out'] = pd.to_datetime(df['c_jail_out'])
        df['c_jail_los'] = (df['c_jail_out'] - df['c_jail_in']).dt.days

        # Add jail years
        df['c_jail_in_year'] = df['c_jail_in'].dt.year
        df['c_jail_out_year'] = df['c_jail_out'].dt.year

        # Add column for year of birth
        df['dob'] = pd.to_datetime(df['dob'])
        df['dob_year'] = df['dob'].dt.year

        concept_map = dict(
            sex='Sex',
            age='Age',
            age_cat='Age Group (<25, 25-45, >45)',
            race='Race',
            juv_fel_count='Juvenile Felony Count',
            juv_misd_count='Juvenile Misdemeanor Count',
            juv_other_count='Juvenile Other Count (Not Felony or Misdemeanor)',
            priors_count='Number of Prior Criminal Offenses',
            c_charge_degree='Charge Degree (Felony or Misdemeanor)',
            two_year_recid='Recidivism within Two Years After Release (Yes/No)',
            c_jail_los='Length of Stay in Jail',
            c_jail_in_year='Year Sent to Jail',
            c_jail_out_year='Year Out of Jail',
            dob_year='Year of Birth'
        )
        concepts = list(concept_map.values())
        df.rename(columns=concept_map, inplace=True)
        
        all_cont_concepts = [
            'Age', 
            'Juvenile Felony Count', 
            'Juvenile Misdemeanor Count', 
            'Juvenile Other Count (Not Felony or Misdemeanor)',
            'Number of Prior Criminal Offenses',
            'Recidivism within Two Years After Release (Yes/No)',
            'Length of Stay in Jail',
            'Year Sent to Jail',
            'Year Out of Jail',
            'Year of Birth'
        ]
        all_cat_concepts = [
            'Sex',
            'Age Group (<25, 25-45, >45)',
            'Race',
            'Charge Degree (Felony or Misdemeanor)'
        ]

        # Choose subset of feature concepts to use
        if self.filtered_concepts is not None:
            selected_concepts = self.filtered_concepts
        elif self.select == 'random-concept' or self.select.startswith('llm'):
            selected_concepts = self.select_concepts(concepts)
        else:
            selected_concepts = concepts

        # Check number of concepts selected
        if self.verbose:
            print(f'Concepts: {len(selected_concepts)}/{len(concepts)} selected.')

        if len(selected_concepts) == 0:
            raise RuntimeError('[Error] No concepts selected with provided setting!')

        Y = (df['score_text'] != 'Low').to_numpy().astype(float)
        selected_df = df[selected_concepts]
        X_selected = selected_df.to_numpy()
        cont_concepts = []
        cont_idxs = []
        cat_concepts = []

        for i, concept in enumerate(selected_concepts):
            if concept in all_cont_concepts:
                cont_idxs.append(i)
                cont_concepts.append(concept)
            elif concept in all_cat_concepts:
                cat_concepts.append(concept)

        self.cont_idxs = cont_idxs
        self.cat_idxs = [i for i in range(len(selected_concepts)) if i not in cont_idxs]

        # Generate the train-val-test split
        if self.stratify_split:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=DEFAULT_SEED)
            train_idx, test_idx = next(sss.split(X_selected, Y))
            X_train, Y_train = X_selected[train_idx], Y[train_idx]
            X_test, Y_test = X_selected[test_idx], Y[test_idx]

            if self.split_train_val:
                sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=self.seed)
                train_idx, val_idx = next(sss.split(X_train, Y_train))
                X_val, Y_val = X_train[val_idx], Y_train[val_idx]
                X_train, Y_train = X_train[train_idx], Y_train[train_idx]
        else:
            X_train, X_test, Y_train, Y_test = train_test_split(
                X_selected, Y, test_size=0.2, random_state=DEFAULT_SEED
            )

            if self.split_train_val:
                X_train, X_val, Y_train, Y_val = train_test_split(
                    X_train, Y_train, test_size=0.2, random_state=self.seed
                )

        # Standardize the features according to the training data
        if self.normalize:
            normalizer = self.get_normalizer()
            X_train = normalizer.fit_transform(X_train)
            X_test = normalizer.transform(X_test)

            if self.split_train_val:
                X_val = normalizer.transform(X_val)

            # Gather column names
            self.columns = cont_concepts
            for i, concept in enumerate(cat_concepts):
                if isinstance(normalizer, ColumnTransformer):
                    categories = normalizer.named_transformers_['onehot'].categories_
                elif isinstance(normalizer, (OneHotEncoder, OrdinalEncoder)):
                    categories = normalizer.categories_
                
                self.columns += [f'{concept}_{cat}' for cat in categories[i]]
        else:
            self.columns = selected_concepts

        # Feature selection at the feature level
        if self.select in ['random-feature', 'filter-mi', 'mrmr']:
            if self.split_train_val:
                X_train, X_test = self.select_features(X_train, Y_train, X_test)
            else:
                X_train, X_val, X_test = self.select_features(X_train, Y_train, (X_val, X_test))

        # Store the relevant split
        if self.split == 'train':
            self.X = X_train
            self.Y = Y_train

        elif self.split == 'val':
            self.X = X_val
            self.Y = Y_val
        
        elif self.split == 'test':
            self.X = X_test
            self.Y = Y_test

        elif self.split == 'all':
            if self.split_train_val:
                self.X = np.concatenate([X_train, X_val, X_test], axis=0)
                self.Y = np.concatenate([Y_train, Y_val, Y_test], axis=0)
            else:
                self.X = np.concatenate([X_train, X_test], axis=0)
                self.Y = np.concatenate([Y_train, Y_test], axis=0)

        if self.verbose:
            print(f'X: {self.X.shape}, Y: {self.Y.shape}')

class MiamiHousingDataset(BaseDataset, NumericMixin):
    '''Miami housing price dataset (Grinsztajn et al., 2022).'''

    def __init__(self, **kwargs):
        super(MiamiHousingDataset, self).__init__(dataset='miami-housing', pred='price', **kwargs)
        self.load()

    @classmethod
    def get_pred_type(cls):
        return 'regression'

    def load(self):
        # Load the dataset
        # OpenML Source: https://www.openml.org/search?type=data&sort=runs&id=43093&status=active
        # HuggingFace Source: https://huggingface.co/datasets/inria-soda/tabular-benchmark/viewer/reg_num_MiamiHousing2016
        arff = scipy.io.arff.loadarff(osp.join(self.data_dir, 'miami2016.arff'))
        df = pd.DataFrame(arff[0])
        concept_map = dict(
            LND_SQFOOT='land area (square feet)',
            TOT_LVG_AREA='floor area (square feet)',
            SPEC_FEAT_VAL='value of special features (e.g., swimming pools)',
            RAIL_DIST='distance to the nearest rail line (an indicator of noise) (feet)',
            OCEAN_DIST='distance to the ocean (feet)',
            WATER_DIST='distance to the nearest body of water (feet)',
            CNTR_DIST='distance to the Miami central business district (feet)',
            SUBCNTR_DI='distance to the nearest subcenter (feet)',
            HWY_DIST='distance to the nearest highway (an indicator of noise) (feet)',
            age='age of the structure',
            avno60plus='dummy variable for airplane noise exceeding an acceptable level',
            structure_quality='quality of the structure',
            month_sold='sale month in 2016 (1 = january)',
            LATITUDE='latitude',
            LONGITUDE='longitude'
        )
        concepts = list(concept_map.values())
        df.rename(columns=concept_map, inplace=True)
        df.drop(columns=['PARCELNO'], inplace=True) # Remove unique structure identifier

        # Choose subset of feature concepts to use
        if self.filtered_concepts is not None:
            selected_concepts = self.filtered_concepts
        elif self.select == 'random-concept' or self.select.startswith('llm'):
            selected_concepts = self.select_concepts(concepts)
        else:
            selected_concepts = concepts

        # Check number of concepts selected
        if self.verbose:
            print(f'Concepts: {len(selected_concepts)}/{len(concepts)} selected.')

        if len(selected_concepts) == 0:
            raise RuntimeError('[Error] No concepts selected with provided setting!')

        Y = np.log(df['SALE_PRC'].to_numpy())
        selected_df = df[selected_concepts]
        X_selected = selected_df.to_numpy()

        # Generate the train-val-test split
        X_train, X_test, Y_train, Y_test = train_test_split(
            X_selected, Y, test_size=0.2, random_state=DEFAULT_SEED
        )

        if self.split_train_val:
            X_train, X_val, Y_train, Y_val = train_test_split(
                X_train, Y_train, test_size=0.2, random_state=self.seed
            )

        # Standardize the features according to the training data
        if self.normalize:
            normalizer = self.get_normalizer()
            X_train = normalizer.fit_transform(X_train)
            X_test = normalizer.transform(X_test)

            if self.split_train_val:
                X_val = normalizer.transform(X_val)

        self.columns = selected_concepts

        # Feature selection at the feature level
        if self.select in ['random-feature', 'filter-mi', 'mrmr']:
            if self.split_train_val:
                X_train, X_test = self.select_features(X_train, Y_train, X_test)
            else:
                X_train, X_val, X_test = self.select_features(X_train, Y_train, (X_val, X_test))

        # Fixed
        self.cont_idxs = np.arange(0,X_train.shape[-1])
        self.cat_idxs = []

        # Store the relevant split
        if self.split == 'train':
            self.X = X_train
            self.Y = Y_train

        elif self.split == 'val':
            self.X = X_val
            self.Y = Y_val
        
        elif self.split == 'test':
            self.X = X_test
            self.Y = Y_test

        elif self.split == 'all':
            if self.split_train_val:
                self.X = np.concatenate([X_train, X_val, X_test], axis=0)
                self.Y = np.concatenate([Y_train, Y_val, Y_test], axis=0)
            else:
                self.X = np.concatenate([X_train, X_test], axis=0)
                self.Y = np.concatenate([Y_train, Y_test], axis=0)

        if self.verbose:
            print(f'X: {self.X.shape}, Y: {self.Y.shape}')

class BreastCancerDataset(BaseDataset, NumericMixin):
    '''Wisconsin breast cancer dataset (Street et al., 1993).'''

    def __init__(self, **kwargs):
        super(BreastCancerDataset, self).__init__(dataset='breast-cancer', pred='diagnosis', **kwargs)
        self.load()

    @classmethod
    def get_pred_type(cls):
        return 'classification'

    def load(self):
        # Load the dataset
        data_source = load_breast_cancer(as_frame=True)
        df = data_source.data
        Y = data_source.target.to_numpy()
        concept_map = {
            'mean radius': 'Mean Radius',
            'mean texture': 'Mean Texture (SE in Gray-Scale Values)',
            'mean perimeter': 'Mean Perimeter',
            'mean area': 'Mean Area',
            'mean smoothness': 'Mean Smoothness (Local Variation in Radius Lengths)',
            'mean compactness': 'Mean Compactness (Perimeter^2 / Area - 1)', 
            'mean concavity': 'Mean Concavity (Severity of Concave Portions of the Contour)',
            'mean concave points': 'Mean Concave Points (Number of Concave Portions of the Contour)',
            'mean symmetry': 'Mean Symmetry',
            'mean fractal dimension': 'Mean Fractal Dimension ("Coastline Approximation" - 1)',
            'radius error': 'SE in Radius',
            'texture error': 'SE in Texture (SE in Gray-Scale Values)',
            'perimeter error': 'SE in Perimeter',
            'area error': 'SE in Area',
            'smoothness error': 'SE in Smoothness (Local Variation in Radius Lengths)',
            'compactness error': 'SE in Compactness (Perimeter^2 / Area - 1)',
            'concavity error': 'SE in Concavity (Severity of Concave Portions of the Contour)',
            'concave points error': 'SE in Concave Points (Number of Concave Portions of the Contour)',
            'symmetry error': 'SE in Symmetry',
            'fractal dimension error': 'SE in Fractal Dimension ("Coastline Approximation" - 1)',
            'worst radius': 'Worst Radius',
            'worst texture': 'Worst Texture (SE in Gray-Scale Values)',
            'worst perimeter': 'Worst Perimeter',
            'worst area': 'Worst Area',
            'worst smoothness': 'Worst Smoothness (Local Variation in Radius Lengths)',
            'worst compactness': 'Worst Compactness (Perimeter^2 / Area - 1)',
            'worst concavity': 'Worst Concavity (Severity of Concave Portions of the Contour)',
            'worst concave points': 'Worst Concave Points (Number of Concave Portions of the Contour)',
            'worst symmetry': 'Worst Symmetry',
            'worst fractal dimension': 'Worst Fractal Dimension ("Coastline Approximation" - 1)'
        }
        concepts = list(concept_map.values())
        df.rename(columns=concept_map, inplace=True)

        # Choose subset of feature concepts to use
        if self.filtered_concepts is not None:
            selected_concepts = self.filtered_concepts
        elif self.select == 'random-concept' or self.select.startswith('llm'):
            selected_concepts = self.select_concepts(concepts)
        else:
            selected_concepts = concepts

        # Check number of concepts selected
        if self.verbose:
            print(f'Concepts: {len(selected_concepts)}/{len(concepts)} selected.')

        if len(selected_concepts) == 0:
            raise RuntimeError('[Error] No concepts selected with provided setting!')

        selected_df = df[selected_concepts]
        X_selected = selected_df.to_numpy()

        # Generate the train-val-test split
        X_train, X_test, Y_train, Y_test = train_test_split(
            X_selected, Y, test_size=0.2, random_state=DEFAULT_SEED
        )

        if self.split_train_val:
            X_train, X_val, Y_train, Y_val = train_test_split(
                X_train, Y_train, test_size=0.2, random_state=self.seed
            )

        # Standardize the features according to the training data
        if self.normalize:
            normalizer = self.get_normalizer()
            X_train = normalizer.fit_transform(X_train)
            X_test = normalizer.transform(X_test)

            if self.split_train_val:
                X_val = normalizer.transform(X_val)
        
        self.columns = selected_concepts

        # Feature selection at the feature level
        if self.select in ['random-feature', 'filter-mi', 'mrmr']:
            if self.split_train_val:
                X_train, X_test = self.select_features(X_train, Y_train, X_test)
            else:
                X_train, X_val, X_test = self.select_features(X_train, Y_train, (X_val, X_test))

        # Fixed
        self.cont_idxs = np.arange(0,X_train.shape[-1])
        self.cat_idxs = []

        # Store the relevant split
        if self.split == 'train':
            self.X = X_train
            self.Y = Y_train

        elif self.split == 'val':
            self.X = X_val
            self.Y = Y_val
        
        elif self.split == 'test':
            self.X = X_test
            self.Y = Y_test

        elif self.split == 'all':
            if self.split_train_val:
                self.X = np.concatenate([X_train, X_val, X_test], axis=0)
                self.Y = np.concatenate([Y_train, Y_val, Y_test], axis=0)
            else:
                self.X = np.concatenate([X_train, X_test], axis=0)
                self.Y = np.concatenate([Y_train, Y_test], axis=0)

        if self.verbose:
            print(f'X: {self.X.shape}, Y: {self.Y.shape}')

class WineDataset(BaseDataset, NumericMixin):
    '''Wine quality dataset (Cortez et al., 2009).'''

    def __init__(self, **kwargs):
        super(WineDataset, self).__init__(dataset='wine', pred='quality', **kwargs)
        self.load()

    @classmethod
    def get_pred_type(cls):
        return 'regression'

    def load(self):
        # Load the dataset
        red_df = pd.read_csv(osp.join(self.data_dir, 'winequality-red.csv'), sep=';')
        white_df = pd.read_csv(osp.join(self.data_dir, 'winequality-white.csv'), sep=';')
        df = pd.concat([red_df, white_df], axis=0)
        concept_map = {
            'fixed acidity': 'Fixed acidity (g(tartaric acid)/dm^3)',
            'volatile acidity': 'Volatile acidity (g(acetic acid)/dm^3)',
            'citric acid': 'Citric acid (g/dm^3)',
            'residual sugar': 'Residual sugar (g/dm^3)',
            'chlorides': 'Chlorides (g(sodium chloride)/dm^3)',
            'free sulfur dioxide': 'Free sulfur dioxide (mg/dm^3)',
            'total sulfur dioxide': 'Total sulfur dioxide (mg/dm^3)',
            'density': 'Density (g/cm^3)',
            'pH': 'pH',
            'sulphates': 'Sulphates (g(potassium sulphate)/dm^3)',
            'alcohol': 'Alcohol (vol.%)'
        }
        concepts = list(concept_map.values())
        concept_map['quality'] = 'label'
        df.rename(columns=concept_map, inplace=True)
            
        Y = df['label'].to_numpy().astype(float)
        
        # Choose subset of feature concepts to use
        if self.filtered_concepts is not None:
            selected_concepts = self.filtered_concepts
        elif self.select == 'random-concept' or self.select.startswith('llm'):
            selected_concepts = self.select_concepts(concepts)
        else:
            selected_concepts = concepts

        # Check number of concepts selected
        if self.verbose:
            print(f'Concepts: {len(selected_concepts)}/{len(concepts)} selected.')

        if len(selected_concepts) == 0:
            raise RuntimeError('[Error] No concepts selected with provided setting!')

        selected_df = df[selected_concepts]
        X_selected = selected_df.to_numpy()

        # Generate the train-val-test split
        X_train, X_test, Y_train, Y_test = train_test_split(
            X_selected, Y, test_size=0.2, random_state=DEFAULT_SEED
        )

        if self.split_train_val:
            X_train, X_val, Y_train, Y_val = train_test_split(
                X_train, Y_train, test_size=0.2, random_state=self.seed
            )

        # Standardize the features according to the training data
        if self.normalize:
            normalizer = self.get_normalizer()
            X_train = normalizer.fit_transform(X_train)
            X_test = normalizer.transform(X_test)

            if self.split_train_val:
                X_val = normalizer.transform(X_val)
        
        self.columns = selected_concepts

        # Feature selection at the feature level
        if self.select in ['random-feature', 'filter-mi', 'mrmr']:
            if self.split_train_val:
                X_train, X_test = self.select_features(X_train, Y_train, X_test)
            else:
                X_train, X_val, X_test = self.select_features(X_train, Y_train, (X_val, X_test))

        # Fixed
        self.cont_idxs = np.arange(0,X_train.shape[-1])
        self.cat_idxs = []

        # Store the relevant split
        if self.split == 'train':
            self.X = X_train
            self.Y = Y_train

        elif self.split == 'val':
            self.X = X_val
            self.Y = Y_val
        
        elif self.split == 'test':
            self.X = X_test
            self.Y = Y_test

        elif self.split == 'all':
            if self.split_train_val:
                self.X = np.concatenate([X_train, X_val, X_test], axis=0)
                self.Y = np.concatenate([Y_train, Y_val, Y_test], axis=0)
            else:
                self.X = np.concatenate([X_train, X_test], axis=0)
                self.Y = np.concatenate([Y_train, Y_test], axis=0)

        if self.verbose:
            print(f'X: {self.X.shape}, Y: {self.Y.shape}')

class CreditGDataset(BaseDataset, CategoricalMixin):
    '''German credit dataset (Hegselmann et al., 2023; Hofmann, 1994).'''

    def __init__(self, **kwargs):
        super(CreditGDataset, self).__init__(dataset='credit-g', pred='risk', **kwargs)
        self.load()

    @classmethod
    def get_pred_type(cls):
        return 'classification'

    def load(self):
        # Load the dataset
        # UCI Reference: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
        # OpenML Source: https://openml.org/search?type=data&sort=runs&status=active&id=31
        arff = scipy.io.arff.loadarff(osp.join(self.data_dir, 'dataset_31_credit-g.arff'))
        df = pd.DataFrame(arff[0])
        concept_map = dict(
            checking_status='Status of existing checking account',
            duration='Duration, in months',
            credit_history='Credit history (credits taken, paid back duly, delays, critical accounts)',
            purpose='Purpose of the credit (e.g., car, television, education)',
            credit_amount='Credit amount',
            savings_status='Status of savings accounts/bonds, in Deutsche Mark',
            employment='Number of years spent in current employment',
            installment_commitment='Installment rate in percentage of disposable income',
            personal_status='Sex and marital status',
            other_parties='Other debtors/guarantors (none/co-applicant/guarantor)',
            residence_since='Number of years spent in current residence',
            property_magnitude='Property (e.g., real estate, life insurance)',
            age='Age',
            other_payment_plans='Other installment plans (bank/stores/none)',
            housing='Housing (rent/own/for free)',
            existing_credits='Number of existing credits at the bank',
            job='Job',
            num_dependents='Number of people being liable to provide maintenance for',
            own_telephone="Telephone (none/registered under customer's name)",
            foreign_worker='Is a foreign worker (yes/no)'
        )
        concepts = list(concept_map.values())
        concept_map['class'] = 'label'
        df.rename(columns=concept_map, inplace=True)

        # Update byte objects to string
        for col, dtype in df.dtypes.items():
            if dtype == object:
                df[col] = df[col].apply(lambda x: x.decode('utf-8'))

        # Format target label
        df['label'] = (df['label'] == 'good')

        # Choose subset of feature concepts to use
        if self.filtered_concepts is not None:
            selected_concepts = self.filtered_concepts
        elif self.select == 'random-concept' or self.select.startswith('llm'):
            selected_concepts = self.select_concepts(concepts)
        else:
            selected_concepts = concepts

        # Check number of concepts selected
        if self.verbose:
            print(f'Concepts: {len(selected_concepts)}/{len(concepts)} selected.')

        if len(selected_concepts) == 0:
            raise RuntimeError('[Error] No concepts selected with provided setting!')

        Y = df['label'].to_numpy().astype(float)
        selected_df = df[selected_concepts]
        X_selected = selected_df.to_numpy()
        cont_concepts = []
        cont_idxs = []
        cat_concepts = []

        for i, (concept, dtype) in enumerate(selected_df.dtypes.items()):
            # NOTE: iufc = int/unsigned int/float/complex
            if dtype.kind in 'iufc':
                cont_idxs.append(i)
                cont_concepts.append(concept)
            else:
                cat_concepts.append(concept)

        self.cont_idxs = cont_idxs
        self.cat_idxs = [i for i in range(len(selected_concepts)) if i not in cont_idxs]

        # Generate the train-val-test split
        if self.stratify_split:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=DEFAULT_SEED)
            train_idx, test_idx = next(sss.split(X_selected, Y))
            X_train, Y_train = X_selected[train_idx], Y[train_idx]
            X_test, Y_test = X_selected[test_idx], Y[test_idx]

            if self.split_train_val:
                sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=self.seed)
                train_idx, val_idx = next(sss.split(X_train, Y_train))
                X_val, Y_val = X_train[val_idx], Y_train[val_idx]
                X_train, Y_train = X_train[train_idx], Y_train[train_idx]
        else:
            X_train, X_test, Y_train, Y_test = train_test_split(
                X_selected, Y, test_size=0.2, random_state=DEFAULT_SEED
            )

            if self.split_train_val:
                X_train, X_val, Y_train, Y_val = train_test_split(
                    X_train, Y_train, test_size=0.2, random_state=self.seed
                )

        # Standardize the features according to the training data
        if self.normalize:
            normalizer = self.get_normalizer()
            X_train = normalizer.fit_transform(X_train)
            X_test = normalizer.transform(X_test)

            if self.split_train_val:
                X_val = normalizer.transform(X_val)

            # Gather column names
            self.columns = cont_concepts
            for i, concept in enumerate(cat_concepts):
                if isinstance(normalizer, ColumnTransformer):
                    categories = normalizer.named_transformers_['onehot'].categories_
                elif isinstance(normalizer, (OneHotEncoder, OrdinalEncoder)):
                    categories = normalizer.categories_
                
                self.columns += [f'{concept}_{cat}' for cat in categories[i]]
        else:
            self.columns = selected_concepts

        # Feature selection at the feature level
        if self.select in ['random-feature', 'filter-mi', 'mrmr']:
            if self.split_train_val:
                X_train, X_test = self.select_features(X_train, Y_train, X_test)
            else:
                X_train, X_val, X_test = self.select_features(X_train, Y_train, (X_val, X_test))

        # Store the relevant split
        if self.split == 'train':
            self.X = X_train
            self.Y = Y_train

        elif self.split == 'val':
            self.X = X_val
            self.Y = Y_val
        
        elif self.split == 'test':
            self.X = X_test
            self.Y = Y_test

        elif self.split == 'all':
            if self.split_train_val:
                self.X = np.concatenate([X_train, X_val, X_test], axis=0)
                self.Y = np.concatenate([Y_train, Y_val, Y_test], axis=0)
            else:
                self.X = np.concatenate([X_train, X_test], axis=0)
                self.Y = np.concatenate([Y_train, Y_test], axis=0)

        if self.verbose:
            print(f'X: {self.X.shape}, Y: {self.Y.shape}')

class DiabetesDataset(BaseDataset, NumericMixin):
    '''Diabetes regression dataset (Efron et al., 2004).'''

    def __init__(self, **kwargs):
        super(DiabetesDataset, self).__init__(dataset='diabetes', pred='progression', **kwargs)
        self.load()

    @classmethod
    def get_pred_type(cls):
        return 'regression'

    def load(self):
        # Load the dataset
        data_source = load_diabetes(as_frame=True, scaled=False)
        df = data_source.data
        Y = data_source.target.to_numpy()
        concept_map = dict(
            age='Age',
            sex='Sex',
            bmi='Body Mass Index (BMI)',
            bp='Average Blood Pressure',
            s1='Total Serum Cholesterol (TC)',
            s2='Low-Density Lipoproteins (LDL)',
            s3='High-Density Lipoproteins (HDL)',
            s4='Total Cholesterol / HDL (TCH)',
            s5='Log of Serum Triglycerides Level (LTG)',
            s6='Blood Sugar Level'
        )
        concepts = list(concept_map.values())
        df.rename(columns=concept_map, inplace=True)

        # Choose subset of feature concepts to use
        if self.filtered_concepts is not None:
            selected_concepts = self.filtered_concepts
        elif self.select == 'random-concept' or self.select.startswith('llm'):
            selected_concepts = self.select_concepts(concepts)
        else:
            selected_concepts = concepts

        # Check number of concepts selected
        if self.verbose:
            print(f'Concepts: {len(selected_concepts)}/{len(concepts)} selected.')

        if len(selected_concepts) == 0:
            raise RuntimeError('[Error] No concepts selected with provided setting!')

        selected_df = df[selected_concepts]
        X_selected = selected_df.to_numpy()

        # Generate the train-val-test split
        X_train, X_test, Y_train, Y_test = train_test_split(
            X_selected, Y, test_size=0.2, random_state=DEFAULT_SEED
        )

        if self.split_train_val:
            X_train, X_val, Y_train, Y_val = train_test_split(
                X_train, Y_train, test_size=0.2, random_state=self.seed
            )

        # Standardize the features according to the training data
        if self.normalize:
            normalizer = self.get_normalizer()
            X_train = normalizer.fit_transform(X_train)
            X_test = normalizer.transform(X_test)

            if self.split_train_val:
                X_val = normalizer.transform(X_val)
        
        self.columns = selected_concepts

        # Feature selection at the feature level
        if self.select in ['random-feature', 'filter-mi', 'mrmr']:
            if self.split_train_val:
                X_train, X_test = self.select_features(X_train, Y_train, X_test)
            else:
                X_train, X_val, X_test = self.select_features(X_train, Y_train, (X_val, X_test))

        # Fixed
        self.cont_idxs = np.arange(0,X_train.shape[-1])
        self.cat_idxs = []

        # Store the relevant split
        if self.split == 'train':
            self.X = X_train
            self.Y = Y_train

        elif self.split == 'val':
            self.X = X_val
            self.Y = Y_val
        
        elif self.split == 'test':
            self.X = X_test
            self.Y = Y_test

        elif self.split == 'all':
            if self.split_train_val:
                self.X = np.concatenate([X_train, X_val, X_test], axis=0)
                self.Y = np.concatenate([Y_train, Y_val, Y_test], axis=0)
            else:
                self.X = np.concatenate([X_train, X_test], axis=0)
                self.Y = np.concatenate([Y_train, Y_test], axis=0)

        if self.verbose:
            print(f'X: {self.X.shape}, Y: {self.Y.shape}')

class CalHousingDataset(BaseDataset, NumericMixin):
    '''California housing price dataset (Pace and Barry, 1997).'''

    def __init__(self, **kwargs):
        super(CalHousingDataset, self).__init__(dataset='calhousing', pred='price', **kwargs)
        self.load()

    @classmethod
    def get_pred_type(cls):
        return 'regression'

    def load(self):
        # Load the dataset
        data_source = fetch_california_housing(as_frame=True)
        df = data_source.data
        Y = data_source.target.to_numpy()
        concept_map = dict(
            MedInc='Median Income in U.S. Census Block Group',
            HouseAge='Median House Age in U.S. Census Block Group',
            AveRooms='Average Number of Rooms Per Household',
            AveBedrms='Average Number of Bedrooms Per Household',
            Population='U.S. Census Block Group Population',
            AveOccup='Average Number of Household Members',
            Latitude='Latitude of U.S. Census Block Group',
            Longitude='Longitude of U.S. Census Block Group'
        )
        concepts = list(concept_map.values())
        df.rename(columns=concept_map, inplace=True)

        # Choose subset of feature concepts to use
        if self.filtered_concepts is not None:
            selected_concepts = self.filtered_concepts
        elif self.select == 'random-concept' or self.select.startswith('llm'):
            selected_concepts = self.select_concepts(concepts)
        else:
            selected_concepts = concepts

        # Check number of concepts selected
        if self.verbose:
            print(f'Concepts: {len(selected_concepts)}/{len(concepts)} selected.')

        if len(selected_concepts) == 0:
            raise RuntimeError('[Error] No concepts selected with provided setting!')

        selected_df = df[selected_concepts]
        X_selected = selected_df.to_numpy()

        # Generate the train-val-test split
        X_train, X_test, Y_train, Y_test = train_test_split(
            X_selected, Y, test_size=0.2, random_state=DEFAULT_SEED
        )

        if self.split_train_val:
            X_train, X_val, Y_train, Y_val = train_test_split(
                X_train, Y_train, test_size=0.2, random_state=self.seed
            )

        # Standardize the features according to the training data
        if self.normalize:
            normalizer = self.get_normalizer()
            X_train = normalizer.fit_transform(X_train)
            X_test = normalizer.transform(X_test)

            if self.split_train_val:
                X_val = normalizer.transform(X_val)
        
        self.columns = selected_concepts

        # Feature selection at the feature level
        if self.select in ['random-feature', 'filter-mi', 'mrmr']:
            if self.split_train_val:
                X_train, X_test = self.select_features(X_train, Y_train, X_test)
            else:
                X_train, X_val, X_test = self.select_features(X_train, Y_train, (X_val, X_test))

        # Fixed
        self.cont_idxs = np.arange(0,X_train.shape[-1])
        self.cat_idxs = []

        # Store the relevant split
        if self.split == 'train':
            self.X = X_train
            self.Y = Y_train

        elif self.split == 'val':
            self.X = X_val
            self.Y = Y_val
        
        elif self.split == 'test':
            self.X = X_test
            self.Y = Y_test

        elif self.split == 'all':
            if self.split_train_val:
                self.X = np.concatenate([X_train, X_val, X_test], axis=0)
                self.Y = np.concatenate([Y_train, Y_val, Y_test], axis=0)
            else:
                self.X = np.concatenate([X_train, X_test], axis=0)
                self.Y = np.concatenate([Y_train, Y_test], axis=0)

        if self.verbose:
            print(f'X: {self.X.shape}, Y: {self.Y.shape}')

class BankDataset(BaseDataset, CategoricalMixin):
    '''Bank dataset (Hegselmann et al., 2023; Kadra et al., 2021).'''

    def __init__(self, **kwargs):
        super(BankDataset, self).__init__(dataset='bank', pred='subscription', **kwargs)
        self.load()

    @classmethod
    def get_pred_type(cls):
        return 'classification'

    def load(self):
        # Load the dataset
        # UCI Reference: https://archive.ics.uci.edu/dataset/222/bank+marketing
        # OpenML Source: https://openml.org/search?type=data&status=active&id=45065
        arff = scipy.io.arff.loadarff(osp.join(self.data_dir, 'phpkIxskf.arff'))
        df = pd.DataFrame(arff[0])
        concepts = [
            'Age', 
            'Occupation', 
            'Marital Status', 
            'Education Level', 
            'Has Credit in Default', 
            'Average Yearly Account Balance', 
            'Has Housing Loan', 
            'Has Personal Loan', 
            'Contact Communication Type (e.g. Cellular, Telephone)', 
            'Last Contact Day of Month',
            'Last Contact Month of Year', 
            'Last Contact Duration (in Seconds)',
            'Number of Contacts Performed During This Campaign and for This Client', 
            'Number of Days Passed After the Client was Last Contacted from a Previous Campaign', 
            'Number of Contacts Performed Before This Campaign and for This Client', 
            'Outcome of the Previous Marketing Campaign'
        ]
        col_map = {f'V{i+1}': c for i,c in enumerate(concepts)}
        col_map['Class'] = 'label' # Target label
        df.rename(columns=col_map, inplace=True)

        # Update byte objects to string
        for col, dtype in df.dtypes.items():
            if dtype == object:
                df[col] = df[col].apply(lambda x: x.decode('utf-8'))

        # Format target label
        df['label'] = (df['label'] == '2')

        # Choose subset of feature concepts to use
        if self.filtered_concepts is not None:
            selected_concepts = self.filtered_concepts
        elif self.select == 'random-concept' or self.select.startswith('llm'):
            selected_concepts = self.select_concepts(concepts)
        else:
            selected_concepts = concepts

        # Check number of concepts selected
        if self.verbose:
            print(f'Concepts: {len(selected_concepts)}/{len(concepts)} selected.')

        if len(selected_concepts) == 0:
            raise RuntimeError('[Error] No concepts selected with provided setting!')

        Y = df['label'].to_numpy().astype(float)
        selected_df = df[selected_concepts]
        X_selected = selected_df.to_numpy()
        cont_concepts = []
        cont_idxs = []
        cat_concepts = []

        for i, (concept, dtype) in enumerate(selected_df.dtypes.items()):
            # NOTE: iufc = int/unsigned int/float/complex
            if dtype.kind in 'iufc':
                cont_idxs.append(i)
                cont_concepts.append(concept)
            else:
                cat_concepts.append(concept)

        self.cont_idxs = cont_idxs
        self.cat_idxs = [i for i in range(len(selected_concepts)) if i not in cont_idxs]

        # Generate the train-val-test split
        if self.stratify_split:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=DEFAULT_SEED)
            train_idx, test_idx = next(sss.split(X_selected, Y))
            X_train, Y_train = X_selected[train_idx], Y[train_idx]
            X_test, Y_test = X_selected[test_idx], Y[test_idx]

            if self.split_train_val:
                sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=self.seed)
                train_idx, val_idx = next(sss.split(X_train, Y_train))
                X_val, Y_val = X_train[val_idx], Y_train[val_idx]
                X_train, Y_train = X_train[train_idx], Y_train[train_idx]
        else:
            X_train, X_test, Y_train, Y_test = train_test_split(
                X_selected, Y, test_size=0.2, random_state=DEFAULT_SEED
            )

            if self.split_train_val:
                X_train, X_val, Y_train, Y_val = train_test_split(
                    X_train, Y_train, test_size=0.2, random_state=self.seed
                )

        # Standardize the features according to the training data
        if self.normalize:
            normalizer = self.get_normalizer()
            X_train = normalizer.fit_transform(X_train)
            X_test = normalizer.transform(X_test)

            if self.split_train_val:
                X_val = normalizer.transform(X_val)

            # Gather column names
            self.columns = cont_concepts
            for i, concept in enumerate(cat_concepts):
                if isinstance(normalizer, ColumnTransformer):
                    categories = normalizer.named_transformers_['onehot'].categories_
                elif isinstance(normalizer, (OneHotEncoder, OrdinalEncoder)):
                    categories = normalizer.categories_
                
                self.columns += [f'{concept}_{cat}' for cat in categories[i]]
        else:
            self.columns = selected_concepts

        # Feature selection at the feature level
        if self.select in ['random-feature', 'filter-mi', 'mrmr']:
            if self.split_train_val:
                X_train, X_test = self.select_features(X_train, Y_train, X_test)
            else:
                X_train, X_val, X_test = self.select_features(X_train, Y_train, (X_val, X_test))

        # Store the relevant split
        if self.split == 'train':
            self.X = X_train
            self.Y = Y_train

        elif self.split == 'val':
            self.X = X_val
            self.Y = Y_val
        
        elif self.split == 'test':
            self.X = X_test
            self.Y = Y_test

        elif self.split == 'all':
            if self.split_train_val:
                self.X = np.concatenate([X_train, X_val, X_test], axis=0)
                self.Y = np.concatenate([Y_train, Y_val, Y_test], axis=0)
            else:
                self.X = np.concatenate([X_train, X_test], axis=0)
                self.Y = np.concatenate([Y_train, Y_test], axis=0)

        if self.verbose:
            print(f'X: {self.X.shape}, Y: {self.Y.shape}')

class ACSDataset(BaseDataset, CategoricalMixin):
    '''ACS datasets from folktables.'''

    def __init__(
        self,
        pred='income',
        survey_year='2018',
        horizon='1-Year',
        survey='person',
        states=['CA'],
        eval_parity=False,
        **kwargs
    ):
        super(ACSDataset, self).__init__(dataset='acs', pred=pred, **kwargs)
        self.survey_year = survey_year
        self.horizon = horizon
        self.survey = survey
        self.states = states
        self.eval_parity = eval_parity
        self.load()

    @classmethod
    def get_pred_type(cls):
        return 'classification'

    def load(self):
        # Load the dataset
        data_source = folktables.ACSDataSource(
            survey_year=self.survey_year,
            horizon=self.horizon, 
            survey=self.survey,
            root_dir=self.data_dir
        )
        data = data_source.get_data(states=self.states, download=True)
        columns = data.columns.tolist()
        defs = data_source.get_definitions(download=True)
        concept_defs = defs[defs[0] == 'NAME'].drop_duplicates()

        ACSFull = self.get_pred_problem(columns)
        concept_codes = ACSFull.df_to_pandas(data)[0].columns.tolist() # Feature concept codes
        concepts = [
            f'{concept_defs[concept_defs[1] == cc][4].values[0]} ({cc})'
            for cc in concept_codes
        ] # Map to full name along with the PUMS code
        concept_dtypes = [
            'categorical' if concept_defs[concept_defs[1] == cc][2].values[0] == 'C' else 'numeric'
            for cc in concept_codes
        ]
        X, Y, G = ACSFull.df_to_numpy(data)
        Y = Y.astype(float)
        
        # Choose subset of feature concepts to use
        if self.filtered_concepts is not None:
            selected_concepts = self.filtered_concepts
        elif self.select == 'random-concept' or self.select.startswith('llm'):
            selected_concepts = self.select_concepts(concepts)
        else:
            selected_concepts = concepts
        
        # Check number of concepts selected
        if self.verbose:
            print(f'Concepts: {len(selected_concepts)}/{len(concepts)} selected.')

        if len(selected_concepts) == 0:
            raise RuntimeError('[Error] No concepts selected with provided setting!')

        X_selected = np.zeros((X.shape[0], len(selected_concepts)))
        cont_concepts = []
        cat_concepts = []
        cont_idxs = []
        current = 0
        for i, concept in enumerate(concepts):
            if concept in selected_concepts:
                X_selected[:,current] = X[:,i]
                
                if concept_dtypes[i] == 'numeric':
                    cont_concepts.append(concept)
                    cont_idxs.append(current)
                else:
                    cat_concepts.append(concept)

                current += 1

        self.cont_idxs = cont_idxs
        self.cat_idxs = [i for i in range(len(selected_concepts)) if i not in cont_idxs]

        # Generate the train-val-test split
        if self.stratify_split:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=DEFAULT_SEED)
            train_idx, test_idx = next(sss.split(X_selected, Y))
            X_train, Y_train, G_train = X_selected[train_idx], Y[train_idx], G[train_idx]
            X_test, Y_test, G_test = X_selected[test_idx], Y[test_idx], G[test_idx]

            if self.split_train_val:
                sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=self.seed)
                train_idx, val_idx = next(sss.split(X_train, Y_train))
                X_val, Y_val, G_val = X_train[val_idx], Y_train[val_idx], G_train[val_idx]
                X_train, Y_train, G_train = X_train[train_idx], Y_train[train_idx], G_train[train_idx]
        else:
            X_train, X_test, Y_train, Y_test, G_train, G_test = train_test_split(
                X_selected, Y, G, test_size=0.2, random_state=DEFAULT_SEED
            )

            if self.split_train_val:
                X_train, X_val, Y_train, Y_val, G_train, G_val = train_test_split(
                    X_train, Y_train, G_train, test_size=0.2, random_state=self.seed
                )

        # Standardize the features according to the training data
        if self.normalize:
            normalizer = self.get_normalizer()
            X_train = normalizer.fit_transform(X_train)
            X_test = normalizer.transform(X_test)

            if self.split_train_val:
                X_val = normalizer.transform(X_val)

            # Gather column names
            self.columns = cont_concepts
            for i, concept in enumerate(cat_concepts):
                if isinstance(normalizer, ColumnTransformer):
                    categories = normalizer.named_transformers_['onehot'].categories_
                elif isinstance(normalizer, (OneHotEncoder, OrdinalEncoder)):
                    categories = normalizer.categories_

                self.columns += [f'{concept}_{cat}' for cat in categories[i]]
        else:
            self.columns = selected_concepts

        # Feature selection at the feature level
        if self.select in ['random-feature', 'filter-mi', 'mrmr']:
            if self.split_train_val:
                X_train, X_test = self.select_features(X_train, Y_train, X_test)
            else:
                X_train, X_val, X_test = self.select_features(X_train, Y_train, (X_val, X_test))

        # Store the relevant split
        if self.split == 'train':
            self.X = X_train
            self.Y = Y_train
            self.G = G_train

        elif self.split == 'val':
            self.X = X_val
            self.Y = Y_val
            self.G = G_val
        
        elif self.split == 'test':
            self.X = X_test
            self.Y = Y_test
            self.G = G_test

        elif self.split == 'all':
            if self.split_train_val:
                self.X = np.concatenate([X_train, X_val, X_test], axis=0)
                self.Y = np.concatenate([Y_train, Y_val, Y_test], axis=0)
                self.G = np.concatenate([G_train, G_val, G_test], axis=0)
            else:
                self.X = np.concatenate([X_train, X_test], axis=0)
                self.Y = np.concatenate([Y_train, Y_test], axis=0)
                self.G = np.concatenate([G_train, G_test], axis=0)

        if self.verbose:
            print(f'X: {self.X.shape}, Y: {self.Y.shape}, G: {self.G.shape}')

    # Returns a folktables.BasicProblem class for given prediction task
    def get_pred_problem(self, columns):
        # Concept codes to exclude
        exc_codes = ['RT', 'SERIALNO', 'NAICSP', 'SOCP']
        
        if self.pred == 'income':
            # Excluding label and income-related features
            # Keywords: income, insurance (low-income)
            exc_codes += [
                'PINCP', 'ADJINC', 'FINCP', 'HINCP', 'FFINCP', 'FHINCP', 
                'HINS1', 'HINS2', 'HINS3', 'HINS4', 'HINS5', 'HINS6', 
                'HINS7', 'INTP', 'OIP', 'PAP', 'RETP', 'SEMP', 'SSIP', 
                'SSP', 'WAGP', 'HICOV', 'PERNP', 'POVPIP', 'PRIVCOV', 
                'PUBCOV', 'FHICOVP', 'FHINS1P', 'FHINS2P', 'FHINS3C', 
                'FHINS3P', 'FHINS4C', 'FHINS4P', 'FHINS5C', 'FHINS5P', 
                'FHINS6P', 'FHINS7P', 'FINTP', 'FOIP', 'FPAP', 'FPINCP', 
                'FPRIVCOVP', 'FPUBCOVP', 'FRETP', 'FSEMP', 'FSSIP', 'FSSP', 
                'FWAGP', 'GRPIP', 'OCPIP'
            ]

            if self.eval_parity:
                exc_codes.append('RAC1P')

            config = dict(
                features=[c for c in columns if c not in exc_codes],
                target='PINCP',
                target_transform=lambda x: x > 50000,
                group='RAC1P',
                preprocess=folktables.acs.adult_filter,
                postprocess=lambda x: np.nan_to_num(x, -1)
            )

        elif self.pred == 'employment':
            # Keywords: work, retirement, occupation, employment
            exc_codes += [
                'ESR', 'FES', 'NWAB', 'NWAV', 'NWLA', 'NWLK', 'NWRE',
                'SEMP', 'FESRP', 'FSEMP', 'FWRKP', 'FWKWP', 'FWKLP', 
                'FWKHP', 'POWSP', 'POWPUMA', 'OCCP', 'JWDP', 'JWAP',
                'WRK', 'WKW', 'WKL', 'WKHP', 'WAGP', 'SSP', 'SSIP',
                'WORKSTAT', 'FRETP', 'FJWTRP', 'FOCCP', 'FPOWSP',
                'FCOWP', 'FJWDP', 'FJWMNP', 'SOCP', 'RETP', 'JWTR',
                'JWMNP', 'COW', 'WKEXREL', 'WIF', 'INDP', 'PERNP',
                'DRIVESP', 'JWRIP', 'PINCP'
            ]

            if self.eval_parity:
                exc_codes.append('RAC1P')

            config = dict(
                features=[c for c in columns if c not in exc_codes],
                target='ESR',
                target_transform=lambda x: x == 1,
                group='RAC1P',
                preprocess=folktables.acs.employment_filter,
                postprocess=lambda x: np.nan_to_num(x, -1)
            )

        elif self.pred == 'public_coverage':
            # Excluding label and features related to income and health insurance
            exc_codes += [
                'PUBCOV', 'ADJINC', 'FINCP', 'GRPIP', 'HINCP', 'OCPIP', 
                'FFINCP', 'FHINCP', 'HINS1', 'HINS2', 'HINS3', 'HINS4', 
                'HINS5', 'HINS6', 'HINS7', 'INTP', 'OIP', 'PAP', 'RETP', 
                'SEMP', 'SSIP', 'SSP', 'WAGP', 'HICOV', 'PERNP', 'POVPIP', 
                'PRIVCOV', 'PUBCOV', 'FHICOVP', 'FHINS1P', 'FHINS2P', 
                'FHINS3C', 'FHINS3P', 'FHINS4C', 'FHINS4P', 'FHINS5C', 
                'FHINS5P', 'FHINS6P', 'FHINS7P', 'FINTP', 'FOIP', 'FPAP', 
                'FPINCP', 'FPRIVCOVP', 'FPUBCOVP', 'FRETP', 'FSEMP', 'FSSIP', 
                'FSSP', 'FWAGP', #'PINCP',
            ]

            if self.eval_parity:
                exc_codes.append('RAC1P')

            config = dict(
                features=[c for c in columns if c not in exc_codes],
                target='PUBCOV',
                target_transform=lambda x: x == 1,
                group='RAC1P',
                preprocess=folktables.acs.public_coverage_filter,
                postprocess=lambda x: np.nan_to_num(x, -1)
            )

        elif self.pred == 'mobility':
            # Keywords: mobility, migration
            exc_codes += [
                'MIG', 'FMIGP', 'MIGPUMA', 'MIGSP', 'FMIGSP'
            ]

            if self.eval_parity:
                exc_codes.append('RAC1P')

            config = dict(
                features=[c for c in columns if c not in exc_codes],
                target='MIG',
                target_transform=lambda x: x == 1,
                group='RAC1P',
                preprocess=lambda x: x.drop(x.loc[(x['AGEP'] <= 18) | (x['AGEP'] >= 35)].index),
                postprocess=lambda x: np.nan_to_num(x, -1)
            )
        
        else:
            raise RuntimeError(f'[Error] {self.pred} not supported.')

        ACSFull = folktables.BasicProblem(**config)

        return ACSFull
    
    def __getitem__(self, idx):
        if self.return_tensor:
            batch = (
                torch.Tensor(self.X[idx]),
                torch.Tensor(self.Y[:,None][idx]),
                torch.Tensor(self.G[:,None][idx])
            )
        else:
            batch = (self.X[idx], self.Y[idx], self.G[idx])

        return batch

class MIMICBaseDataset(BaseDataset):
    '''Base dataset class for MIMIC-IV and eICU datasets.'''

    def __init__(self, pred, L, bin_size, flatten=False, **kwargs):
        super(MIMICBaseDataset, self).__init__(**kwargs)
        self.pred = pred
        self.L = L
        self.bin_size = bin_size
        self.flatten = flatten
        self.load()

    # Loads the dataset
    def load(self):
        # Static data
        with open(osp.join(self.data_dir, 'static_data.pkl'), 'rb') as fh:
            static = pickle.load(fh)
        
        static_features = static['features']
        static_masks = static['masks']
        
        with open(osp.join(self.data_dir, 'static_concepts.yaml'), 'r') as fh:
            static_concepts = yaml.load(fh, Loader=yaml.FullLoader)

        # Time-series data
        if self.dataset in ['mimic-icd', 'eicu-icd']:
            with open(osp.join(self.data_dir, f'time_series_data_{self.L}_{self.bin_size}.pkl'), 'rb') as fh:
                time_series = pickle.load(fh)
        
        elif self.dataset in ['mimic', 'eicu']:
            with open(osp.join(self.data_dir, 'time_series_data.pkl'), 'rb') as fh:
                time_series = pickle.load(fh)

        time_series_features = time_series['features']
        time_series_masks = time_series['masks']
        time_series_deltas = time_series['deltas']

        with open(osp.join(self.data_dir, 'time_series_concepts.yaml'), 'r') as fh:
            time_series_concepts = yaml.load(fh, Loader=yaml.FullLoader)

        # Feature mapping
        mapping = pd.read_csv(osp.join(self.data_dir, 'mapping.csv'))

        # Choose subset of feature concepts to retrieve
        if self.filtered_concepts is not None:
            S_concepts = [c for c in self.filtered_concepts if c in static_concepts]
            X_concepts = [c for c in self.filtered_concepts if c in time_series_concepts]
        elif self.select == 'random-concept' or self.select.startswith('llm'):
            S_concepts, X_concepts = self.select_concepts(static_concepts, time_series_concepts)
        else:
            S_concepts = static_concepts
            X_concepts = time_series_concepts

        # Check number of concepts selected
        if self.verbose:
            print(f'Static Concepts: {len(S_concepts)}/{len(static_concepts)} selected.')
            print(f'Time-Series Concepts: {len(X_concepts)}/{len(time_series_concepts)} selected.')

        if len(S_concepts) == 0 and len(X_concepts) == 0:
            raise RuntimeError('[Error] No concepts selected with provided setting!')

        # Append all static features together
        S = None
        S_mask = None
        S_idxs = []
        S_columns = []
        S_mask_columns = []
        current_idx = 0

        with tqdm(S_concepts, leave=False) as tconcepts:
            for concept in tconcepts:
                tconcepts.set_description(f'Appending {concept}')
                
                if S is None:
                    S = static_features[concept]
                    S_mask = static_masks[concept]
                else:
                    S = np.concatenate([S, static_features[concept]], axis=-1)
                    S_mask = np.concatenate([S_mask, static_masks[concept]], axis=-1)

                # Numeric features
                if concept in ['age', 'height', 'weight']:
                    S_idxs.append(current_idx)
                    S_columns.append(concept)
                    S_mask_columns.append(f'{concept}_mask')
                    current_idx += 1
                else:
                    S_columns += [f'{concept}_{cat}' for cat in static_to_onehot[concept].categories_[0]]
                    S_mask_columns.append(f'{concept}_mask')
                    current_idx += static_to_onehot[concept].categories_[0].size
        
        # Concatenate the static features and masks
        if len(S_concepts) > 0:
            S = np.concatenate([S, S_mask], axis=-1)
            self.D_S = S.shape[-1]

            S_columns += S_mask_columns
            assert(S.shape[-1] == len(S_columns))
            
            if self.verbose:
                print(f'Static data: {S.shape}')
        else:
            self.D_S = 0
            
            if self.verbose:
                print('Static data: N/A')

        self.S_idxs = np.array(S_idxs)

        # Append all time-series features together
        X = None
        X_mask = None
        X_delta = None
        X_idxs = []
        X_columns = []
        X_mask_columns = []
        current_idx = 0

        with tqdm(X_concepts, leave=False) as tconcepts:
            for concept in tconcepts:
                tconcepts.set_description(f'Appending {concept}')

                if X is None:
                    X = time_series_features[concept]
                    X_mask = time_series_masks[concept]
                    X_delta = time_series_deltas[concept]
                else:
                    X = np.concatenate([X, time_series_features[concept]], axis=-1)
                    X_mask = np.concatenate([X_mask, time_series_masks[concept]], axis=-1)
                    X_delta = np.concatenate([X_delta, time_series_deltas[concept]], axis=-1)

                dtype = mapping[mapping['feature'] == concept].iloc[0]['dtype']
                if dtype == 'numeric':
                    X_idxs.append(current_idx)
                    X_columns.append(concept)
                    X_mask_columns.append(f'{concept}_mask')
                    current_idx += 1
                elif dtype == 'categorical':
                    X_columns += [f'{concept}_{cat}' for cat in time_series_to_onehot[concept].categories_[0]]
                    X_mask_columns.append(f'{concept}_mask')
                    current_idx += time_series_to_onehot[concept].categories_[0].size
                else:
                    X_columns.append(concept)
                    X_mask_columns.append(f'{concept}_mask')
                    current_idx += 1
        
        # Concatenate the time-series features, masks, and delta-time vectors
        if len(X_concepts) > 0:
            X = np.concatenate([X, X_mask, X_delta], axis=-1)
            self.D_X = X.shape[-1]
            
            X_delta_columns = [f'{col.split("_")[0]}_delta' for col in X_mask_columns]
            X_columns += X_mask_columns + X_delta_columns
            assert(X.shape[-1] == len(X_columns))

            X_idxs += list(range(-len(X_delta_columns),0)) # Include the delta-time vector indices for standardization

            if self.verbose:
                print(f'Time-series data: {X.shape}\n')
        else:
            self.D_X = 0
            
            if self.verbose:
                print(f'Time-series data: (N/A)\n')

        self.X_idxs = np.array(X_idxs)

        # Combine the static and time-series features
        if self.flatten:
            if len(X_columns) > 0:
                # Add suffix with hour information
                X_columns = [f'{c}_{i*self.bin_size}hr' for c in X_columns for i in range(self.L//self.bin_size)]

            if S is None: # Edge case: No static concepts selected
                X_full = X.reshape(X.shape[0],-1) # [N,T*D_X]
            elif X is None: # Edge case: No time-series concepts selected
                X_full = S # [N,D_S]
            else:
                X_full = np.hstack([S, X.reshape(X.shape[0],-1)]) # [N,D_S+T*D_X]
        else:
            if S is None:
                X_full = X # [N,T,D_X]
            elif X is None:
                X_full = np.repeat(S[:,None,:], self.L//self.bin_size, axis=1) # [N,T,D_S]
            else:
                S_repeated = np.repeat(S[:,None,:],self.L//self.bin_size, axis=1) # [N,T,D_S]
                X_full = np.concatenate([S_repeated, X], axis=-1) # [N,T,D_S+D_X]

        # Save column names
        self.columns = np.array(S_columns + X_columns)

        # Load labels
        static_df = pd.read_csv(osp.join(self.data_dir, 'static_df.csv'))
        Y = static_df[self.pred].to_numpy() # [N,]
        ids = static_df['stay_id'].to_numpy() # [N,]

        # Generate the train-val-test split
        if self.stratify_split:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=DEFAULT_SEED)
            train_idx, test_idx = next(sss.split(X_full, Y))
            X_train, Y_train, train_ids = X_full[train_idx], Y[train_idx], ids[train_idx]
            X_test, Y_test, test_ids = X_full[test_idx], Y[test_idx], ids[test_idx]

            if self.split_train_val:
                sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=self.seed)
                train_idx, val_idx = next(sss.split(X_train, Y_train))
                X_val, Y_val, val_ids = X_train[val_idx], Y_train[val_idx], train_ids[val_idx]
                X_train, Y_train, train_ids = X_train[train_idx], Y_train[train_idx], train_ids[train_idx]
        else:
            X_train, X_test, Y_train, Y_test, train_ids, test_ids = train_test_split(
                X_full, Y, ids, test_size=0.2, random_state=DEFAULT_SEED
            )

            if self.split_train_val:
                X_train, X_val, Y_train, Y_val, train_ids, val_ids = train_test_split(
                    X_train, Y_train, train_ids, test_size=0.2, random_state=self.seed
                )

        # Standardize the features according to the training data
        if self.normalize:
            normalizer = self.get_normalizer()
            X_train = normalizer.fit_transform(X_train)
            X_test = normalizer.transform(X_test)

            if self.split_train_val:
                X_val = normalizer.transform(X_val)

        # Feature selection at the feature level
        if self.select in ['random-feature', 'filter-mi', 'mrmr']:
            if self.split_train_val:
                X_train, X_test = self.select_features(X_train, Y_train, X_test)
            else:
                X_train, X_val, X_test = self.select_features(X_train, Y_train, (X_val, X_test))

        # Store the relevant split
        if self.split == 'train':
            self.X = X_train
            self.Y = Y_train
            self.ids = train_ids

        elif self.split == 'val':
            self.X = X_val
            self.Y = Y_val
            self.ids = val_ids

        elif self.split == 'test':
            self.X = X_test
            self.Y = Y_test
            self.ids = test_ids

        elif self.split == 'all':
            if self.split_train_val:
                self.X = np.concatenate([X_train, X_val, X_test], axis=0)
                self.Y = np.concatenate([Y_train, Y_val, Y_test], axis=0)
                self.ids = np.concatenate([train_ids, val_ids, test_ids], axis=0)
            else:
                self.X = np.concatenate([X_train, X_test], axis=0)
                self.Y = np.concatenate([Y_train, Y_test], axis=0)
                self.ids = np.concatenate([train_ids, test_ids], axis=0)

        if self.verbose:
            print(f'X: {self.X.shape}, Y: {self.Y.shape}, ids: {self.ids.shape}')

    # Selects features at the concept level
    def select_concepts(self, static_concepts, time_series_concepts):
        # Preload the LLM prompt outputs
        if self.select.startswith('llm-rank'):
            llm_outpath = osp.join(self.prompt_outdir, f'{self.pred}_rank.yaml')
            with open(llm_outpath, 'r') as fh:
                llm_output = yaml.load(fh, Loader=yaml.Loader)
        
        elif self.select.startswith('llm'):
            llm_outname = osp.join(self.prompt_outdir, f'{self.pred}_{self.llm_decoding}')

            # Fetch LLM output from prompt with context
            if self.with_context:
                llm_outname += '_context'

            # Fetch LLM output from prompt with examples + explanations
            if self.with_expls:
                llm_outname += '_expls'

            # Fetch LLM output from prompt with examples
            elif self.with_examples:
                llm_outname += '_examples'

            #llm_outpath = f'{llm_outname}.pkl'
            llm_outpath = f'{llm_outname}.yaml'

            #with open(llm_outpath, 'rb') as fh:
            #    llm_output = pickle.load(fh)
            with open(llm_outpath, 'r') as fh:
                llm_output = yaml.load(fh, Loader=yaml.Loader)
                
        # Randomly select features at the concept level
        if self.select == 'random-concept':
            np.random.seed(self.seed)
            selected_concepts = np.random.choice(
                static_concepts + time_series_concepts,
                size=int(self.ratio * len(static_concepts + time_series_concepts)),
                replace=False
            ).tolist()
        
        # LLM concept selection by threshold
        elif self.select == 'llm-threshold':
            selected_concepts = selection.utils.llm_select_concepts(
                llm_output, threshold=self.threshold, seed=self.seed
            )

        # LLM concept selection by fraction
        elif self.select == 'llm-ratio':
            selected_concepts = selection.utils.llm_select_concepts(
                llm_output, ratio=self.ratio, seed=self.seed
            )

        # LLM concept selection by number of elements
        elif self.select == 'llm-topk':
            selected_concepts = selection.utils.llm_select_concepts(
                llm_output, topk=self.topk, seed=self.seed
            )

        # LLM concept selection by direct ranking (Default: by ratio)
        elif self.select == 'llm-rank':
            selected_concepts = selection.utils.llm_select_concepts(
                llm_output, ratio=self.ratio, seed=self.seed, rank=True
            )

        else:
            raise RuntimeError(f'[Error] Invalid concept selection mode provided: {self.select}.')

        S_concepts = [c for c in selected_concepts if c in static_concepts]
        X_concepts = [c for c in selected_concepts if c in time_series_concepts]

        return S_concepts, X_concepts
    
    # Selects features at the feature level
    def select_features(self, X_train, Y_train, Xs):
        # Random feature selection
        if self.select == 'random-feature':
            np.random.seed(self.seed)
            idxs = np.random.choice(
                np.arange(0, X_train.shape[-1]), 
                size=int(self.ratio * X_train.shape[-1]), 
                replace=False
            ).tolist()

            self.columns = self.columns[idxs]

            X_train = X_train[...,idxs]
            
            if isinstance(Xs, list) or isinstance(Xs, tuple):
                new_Xs = [X[...,idxs] for X in Xs]
            else:
                new_Xs = Xs[...,idxs]

        # Filtering based on mutual information
        # NOTE: Only applicable for flattened data
        elif self.select == 'filter-mi':
            if self.flatten:
                # NOTE: percentile = % of features to *keep*
                mi_filter = SelectPercentile(
                    mutual_info_classif, percentile=int(self.ratio * 100)
                )
                X_train = mi_filter.fit_transform(X_train, Y_train)

                if isinstance(Xs, list) or isinstance(Xs, tuple):
                    new_Xs = [mi_filter.transform(X) for X in Xs]
                else:
                    new_Xs = mi_filter.transform(Xs)

                self.columns = self.columns[mi_filter.get_support()] # Boolean mask on features
            else:
                print('[filter-mi] Option only applicable to flattened data.')

        return (X_train, *new_Xs) if (isinstance(Xs, list) or isinstance(Xs, tuple)) else (X_train, new_Xs)

    def get_normalizer(self):
        config = [self.X_idxs, self.S_idxs, self.L//self.bin_size, self.D_X, self.D_S, StandardScaler(), StandardScaler()]
        normalizer = FlatTransformer(*config) if self.flatten else TimeSeriesTransformer(*config)

        return normalizer
    
    def __getitem__(self, idx):
        if self.return_tensor:
            batch = (
                torch.Tensor(self.X[idx]),
                torch.Tensor(self.Y[:,None][idx]),
                torch.Tensor(self.ids[:,None][idx])
            )
        else:
            batch = (self.X[idx], self.Y[idx], self.ids[idx])

        return batch

class MIMICICDDataset(MIMICBaseDataset):
    '''MIMIC-IV ICD code prediction dataset.'''

    def __init__(self, pred, L=24, bin_size=4, **kwargs):
        super(MIMICICDDataset, self).__init__(
            pred=pred,
            L=L,
            bin_size=bin_size,
            dataset='mimic-icd', 
            **kwargs
        )
        self.load()

    @classmethod
    def get_pred_type(cls):
        return 'classification'
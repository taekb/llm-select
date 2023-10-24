import os
import os.path as osp
import re
import yaml
import openai
import pickle
from pprint import pprint
from datetime import datetime
import random
import numpy as np
import ranky as rk
import pickle
import pandas as pd
from tqdm import tqdm
from thefuzz import fuzz
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from numbers import Integral, Real
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import _fit_context, is_classifier, clone
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import cross_val_score, check_cv

import vllm

import langchain
import langchain.chat_models as chat_models
import langchain.llms as llms
import langchain.output_parsers as output_parsers
from langchain.prompts import HumanMessagePromptTemplate
import pydantic
from typing import Optional

import huggingface_hub

import folktables

# Default directories
ABS_DIR = osp.dirname(osp.abspath(__file__))
DATA_DIR = osp.join(ABS_DIR, '../../data')
CONFIG_DIR = osp.join(ABS_DIR, '../../config')
PROMPT_DIR = osp.join(ABS_DIR, '../../prompts')
PROMPT_OUTDIR = osp.join(ABS_DIR, '../../prompt_outputs')

DEFAULT_SEED = 1

# Llama-2 path
HF_LLAMA2_DIR = 'meta-llama/Llama-2-{}b-chat-hf'

# Load OpenAI API key
with open(osp.join(CONFIG_DIR, 'openai_api_key.txt'), 'r') as fh:
    openai_api_key = fh.read().strip()
    openai.api_key = openai_api_key

# Load HugginFace API token and authenticate
with open(osp.join(CONFIG_DIR, 'hf_api_key.txt'), 'r') as fh:
    hf_api_key = fh.read()
    huggingface_hub.login(token=hf_api_key)

'''
    Customized feature selection utilities.

'''

class CustomSequentialFeatureSelector(SequentialFeatureSelector):
    '''Customized sklearn SFS that keeps track of intermediate scores.'''

    # NOTE: Added `cont_idxs` to dynamically configure the normalizer for each candidate feature set
    def __init__(self, estimator, n_features_to_select=None, direction='forward',
                 scoring=None, cv=5, n_jobs=None, cont_idxs=None):
        super().__init__(
            estimator=estimator, 
            n_features_to_select=n_features_to_select,
            direction=direction,
            scoring=scoring, 
            cv=cv, 
            n_jobs=n_jobs
        )
        self.cont_idxs = cont_idxs
        self.intermediate_scores_ = []
        self.feature_idx_order_ = []

    @_fit_context(
        # SequentialFeatureSelector.estimator is not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(self, X, y=None):
        """Learn the features to select from X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of predictors.

        y : array-like of shape (n_samples,), default=None
            Target values. This parameter may be ignored for
            unsupervised learning.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        tags = self._get_tags()
        X = self._validate_data(
            X,
            accept_sparse="csc",
            ensure_min_features=2,
            force_all_finite=not tags.get("allow_nan", True),
        )
        n_features = X.shape[1]

        if self.n_features_to_select == "auto":
            if self.tol is not None:
                # With auto feature selection, `n_features_to_select_` will be updated
                # to `support_.sum()` after features are selected.
                self.n_features_to_select_ = n_features - 1
            else:
                self.n_features_to_select_ = n_features // 2
        elif isinstance(self.n_features_to_select, Integral):
            if self.n_features_to_select >= n_features:
                raise ValueError("n_features_to_select must be < n_features.")
            self.n_features_to_select_ = self.n_features_to_select
        elif isinstance(self.n_features_to_select, Real):
            self.n_features_to_select_ = int(n_features * self.n_features_to_select)

        if self.tol is not None and self.tol < 0 and self.direction == "forward":
            raise ValueError("tol must be positive when doing forward selection")

        cv = check_cv(self.cv, y, classifier=is_classifier(self.estimator))

        cloned_estimator = clone(self.estimator)

        # the current mask corresponds to the set of features:
        # - that we have already *selected* if we do forward selection
        # - that we have already *excluded* if we do backward selection
        current_mask = np.zeros(shape=n_features, dtype=bool)
        n_iterations = (
            self.n_features_to_select_
            if self.n_features_to_select == "auto" or self.direction == "forward"
            else n_features - self.n_features_to_select_
        )

        old_score = -np.inf
        is_auto_select = self.tol is not None and self.n_features_to_select == "auto"

        with tqdm(range(n_iterations), unit='feature') as tfeature:
            for _ in tfeature:
                new_feature_idx, new_score = self._get_best_new_feature_score(
                    cloned_estimator, X, y, cv, current_mask
                )
                tfeature.set_postfix(added=new_feature_idx)

                if is_auto_select and ((new_score - old_score) < self.tol):
                    break

                old_score = new_score
                current_mask[new_feature_idx] = True

        if self.direction == "backward":
            current_mask = ~current_mask

        self.support_ = current_mask
        self.n_features_to_select_ = self.support_.sum()

        return self

    def _get_best_new_feature_score(self, estimator, X, y, cv, current_mask):
        # Return the best new feature and its score to add to the current_mask,
        # i.e. return the best new feature and its score to add (resp. remove)
        # when doing forward selection (resp. backward selection).
        # Feature will be added if the current score and past score are greater
        # than tol when n_feature is auto,
        
        candidate_feature_indices = np.flatnonzero(~current_mask)
        scores = {}
        
        # Iterate through all candidate features not yet selected
        for feature_idx in candidate_feature_indices:
            candidate_mask = current_mask.copy()
            candidate_mask[feature_idx] = True # Include index of candidate feature

            if self.direction == "backward":
                candidate_mask = ~candidate_mask

            # Subselect already selected + candidate features
            X_new = X[:, candidate_mask]

            # NOTE: Need to make sure that the reordering of the feature indices in the
            # transformer doesn't mess up what feature index gets added to bookkeeping!
            if self.cont_idxs is not None:
                scaler_idxs = []
                for i, idx in enumerate(np.where(candidate_mask)[0]):
                    if idx in self.cont_idxs:
                        scaler_idxs.append(i)

                # Update estimator
                ct = ColumnTransformer([
                    ('z-score', StandardScaler(), scaler_idxs)
                ], remainder='passthrough')
                
                pipe = Pipeline([
                    ('normalize', ct), ('clf', estimator)
                ])
                
                scores[feature_idx] = cross_val_score(
                    pipe,
                    X_new,
                    y,
                    cv=cv,
                    scoring=self.scoring,
                    n_jobs=self.n_jobs,
                ).mean()

            else:
                scores[feature_idx] = cross_val_score(
                    estimator,
                    X_new,
                    y,
                    cv=cv,
                    scoring=self.scoring,
                    n_jobs=self.n_jobs,
                ).mean()

        new_feature_idx = max(scores, key=lambda feature_idx: scores[feature_idx])
        self.intermediate_scores_.append(scores[new_feature_idx])
        self.feature_idx_order_.append(new_feature_idx)

        return new_feature_idx, scores[new_feature_idx]


'''
    Utility functions for LLM-based feature selection.

'''

class FeatureImportance(pydantic.BaseModel):
    '''Langchain Pydantic output parsing structure.'''

    reasoning: Optional[str] = pydantic.Field(
        description='Logical reasoning behind feature importance score'
    )
    score: float = pydantic.Field(
        description='Feature importance score'
    )
    
def load_template(
    dataset='mimic-icd',
    pred='ckd',
    prompt_dir=PROMPT_DIR,
    sequential=False,
    rank=False,
    llm_model='gpt-3.5-turbo'
):
    '''Prompt template loading function.'''

    if sequential:
        template_path = osp.join(prompt_dir, f'{dataset}/{pred}_template_seq.txt')
        with open(template_path, 'r') as fh:
            template = fh.read()

        # Add specialized system and instruction tags for Llama-2
        if llm_model.startswith('llama-2'):
            template = '<s>[INST] <<SYS>>\n' + template + '<</SYS>>\n\n'

    elif rank:
        template_path = osp.join(prompt_dir, f'{dataset}/{pred}_template_rank.txt')
        with open(template_path, 'r') as fh:
            template = fh.read()

        instruction = 'Rank all {n_concepts} features in the following list:\n{concepts}.'
        #instruction += '\nOnly output the feature names (no explanations) in your answer.'
            
        if llm_model.startswith('llama-2'):
            template = '<s>[INST] <<SYS>>\n' + template + '<</SYS>>\n\n' + \
                instruction + ' [/INST]\n'
        else:
            template += '\n' + instruction + '\n'
    else:
        template_path = osp.join(prompt_dir, f'{dataset}/{pred}_template.txt')
        with open(template_path, 'r') as fh:
            template = fh.read()

        instruction = 'Provide a score and reasoning for "{concept}" ' \
            'formatted according to the output schema above:'
        
        if llm_model.startswith('llama-2'):
            template = '<s>[INST] <<SYS>>\n' + template + '<</SYS>>\n\n' + \
                instruction + ' [/INST]\n'

        else:
            template += '\n' + instruction + '\n'

    return template

def load_context(
    dataset='mimic-icd',
    pred='ckd',
    prompt_dir=PROMPT_DIR
):
    '''Prompt context loading function.'''

    context_path = osp.join(prompt_dir, f'{dataset}/{pred}_context.txt')
    with open(context_path, 'r') as fh:
        context = fh.read()

    return context

def load_examples(
    dataset='mimic-icd',
    pred='ckd',
    prompt_dir=PROMPT_DIR,
    add_expls=False
):
    '''Prompt example loading function.'''

    examples_path = osp.join(prompt_dir, f'{dataset}/{pred}_examples.txt')
    with open(examples_path, 'r') as fh:
        examples = fh.read()

    # (Optional) Filter out reasoning
    if not add_expls:
        examples = '\n'.join([
            e for e in examples.split('\n') if 'reasoning' not in e
        ])

    return examples

def load_concepts(dataset='mimic-icd', pred=None, data_dir=DATA_DIR):
    '''Concept loading function.'''

    if dataset == 'mimic-icd':
        # Static concepts
        with open(osp.join(data_dir, f'{dataset}/static_concepts.yaml'), 'r') as fh:
            static_concepts = yaml.load(fh, Loader=yaml.FullLoader)
        
        # Time-series concepts
        with open(osp.join(data_dir, f'{dataset}/time_series_concepts.yaml'), 'r') as fh:
            time_series_concepts = yaml.load(fh, Loader=yaml.FullLoader)

        concepts = static_concepts + time_series_concepts

    elif dataset == 'acs':
        # Default ACS config
        data_source = folktables.ACSDataSource(
            survey_year='2018', 
            horizon='1-Year', 
            survey='person',
            root_dir=osp.join(DATA_DIR, 'acs')
        )
        data = data_source.get_data(states=['CA'], download=True)
        columns = data.columns.tolist()
        defs = data_source.get_definitions(download=True)
        concept_defs = defs[defs[0] == 'NAME'].drop_duplicates()

        # Codes to exclude
        exc_codes = ['RT', 'SERIALNO', 'NAICSP', 'SOCP']

        if pred == 'income':
            exc_codes.append('PINCP')
        elif pred == 'employment':
            exc_codes.append('ESR')
        elif pred == 'public_coverage':
            exc_codes.append('PUBCOV')
        elif pred == 'travel_time':
            exc_codes.append('JWMNP')
        elif pred == 'mobility':
            exc_codes.append('MIG')

        concept_codes = [c for c in columns if c not in exc_codes]
        concepts = [
            f'{concept_defs[concept_defs[1] == cc][4].values[0]} ({cc})'
            for cc in concept_codes
        ]

    elif dataset == 'bank':
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

    elif dataset == 'calhousing':
        concepts = [
            'Median Income in U.S. Census Block Group',
            'Median House Age in U.S. Census Block Group',
            'Average Number of Rooms Per Household',
            'Average Number of Bedrooms Per Household',
            'U.S. Census Block Group Population',
            'Average Number of Household Members',
            'Latitude of U.S. Census Block Group',
            'Longitude of U.S. Census Block Group'
        ]

    elif dataset == 'diabetes':
        concepts = [
            'Age',
            'Sex',
            'Body Mass Index (BMI)',
            'Average Blood Pressure',
            'Total Serum Cholesterol (TC)',
            'Low-Density Lipoproteins (LDL)',
            'High-Density Lipoproteins (HDL)',
            'Total Cholesterol / HDL (TCH)',
            'Log of Serum Triglycerides Level (LTG)',
            'Blood Sugar Level'
        ]

    elif dataset == 'credit-g':
        concepts = [
            'Status of existing checking account',
            'Duration, in months',
            'Credit history (credits taken, paid back duly, delays, critical accounts)',
            'Purpose of the credit (e.g., car, television, education)',
            'Credit amount',
            'Status of savings accounts/bonds, in Deutsche Mark',
            'Number of years spent in current employment',
            'Installment rate in percentage of disposable income',
            'Sex and marital status',
            'Other debtors/guarantors (none/co-applicant/guarantor)',
            'Number of years spent in current residence',
            'Property (e.g., real estate, life insurance)',
            'Age',
            'Other installment plans (bank/stores/none)',
            'Housing (rent/own/for free)',
            'Number of existing credits at the bank',
            'Job',
            'Number of people being liable to provide maintenance for',
            "Telephone (none/registered under customer's name)",
            'Is a foreign worker (yes/no)'
        ]

    elif dataset == 'wine':
        concepts = [
            'Fixed acidity (g(tartaric acid)/dm^3)',
            'Volatile acidity (g(acetic acid)/dm^3)',
            'Citric acid (g/dm^3)',
            'Residual sugar (g/dm^3)',
            'Chlorides (g(sodium chloride)/dm^3)',
            'Free sulfur dioxide (mg/dm^3)',
            'Total sulfur dioxide (mg/dm^3)',
            'Density (g/cm^3)',
            'pH',
            'Sulphates (g(potassium sulphate)/dm^3)',
            'Alcohol (vol.%)'
        ]

    elif dataset == 'miami-housing':
        concepts = [
            'land area (square feet)',
            'floor area (square feet)',
            'value of special features (e.g., swimming pools)',
            'distance to the nearest rail line (an indicator of noise) (feet)',
            'distance to the ocean (feet)',
            'distance to the nearest body of water (feet)',
            'distance to the Miami central business district (feet)',
            'distance to the nearest subcenter (feet)',
            'distance to the nearest highway (an indicator of noise) (feet)',
            'age of the structure',
            'dummy variable for airplane noise exceeding an acceptable level',
            'quality of the structure',
            'sale month in 2016 (1 = january)',
            'latitude',
            'longitude'
        ]

    elif dataset == 'compas':
        concepts = [
            'Sex',
            'Age',
            'Age Group (<25, 25-45, >45)',
            'Race',
            'Juvenile Felony Count',
            'Juvenile Misdemeanor Count',
            'Juvenile Other Count (Not Felony or Misdemeanor)',
            'Number of Prior Criminal Offenses',
            'Charge Degree (Felony or Misdemeanor)',
            'Recidivism within Two Years After Release (Yes/No)',
            'Length of Stay in Jail',
            'Year Sent to Jail',
            'Year Out of Jail',
            'Year of Birth'
        ]

    elif dataset == 'give-me-credit':
        concepts = [
            'Total balance on credit cards and personal lines of credit ' \
                '(except real estate and no installment debt like car loans, divided by monthly gross income)', 
            'Age',
            'Monthly debt payments, alimony, and living costs, divided by monthly gross income',
            'Monthly income',
            'Number of open loans and lines of credit', 
            'Number of mortgage and real estate loans, including home equity lines of credit', 
            'Number of times borrower has been 30 to 59 days past due ' \
                '(but no worse) in the last 2 years', 
            'Number of times borrower has been 60 to 89 days past due ' \
                '(but no worse) in the last 2 years',
            'Number of times borrower has been 90 days or more past due in the last 2 years',
            'Number of dependents in family, excluding themselves'
        ]
    
    elif dataset == 'pima':
        concepts = [
            'Number of times pregnant',
            'Plasma glucose concentration at 2 hours in an oral glucose tolerance test',
            'Diastolic blood pressure (mmHg)',
            'Triceps skin fold thickness (mm)',
            '2-hour serum insulin (muU/ml)',
            'Body mass index (weight in kg/(height in m)^2)',
            'Diabetes pedigree function',
            'Age'
        ]
    
    elif dataset == 'cars':
        concepts = [
            'Present price', 
            'Driven kilometers', 
            'Fuel type', 
            'Selling type (dealer/individual)', 
            'Transmission (manual/automatic)',
            'Number of previous owners',
            'Age'
        ]
        
    return concepts

# Helper function for parsing concept outputs
def parse_and_aggregate(concept, concept_outputs, parser, verbose, fix_with_llm):
    scores = []
    expls = []
    n_samples = len(concept_outputs)

    # Check parser type
    if isinstance(parser, output_parsers.PydanticOutputParser):
        parser_type = 'pydantic'
    
    elif isinstance(parser, output_parsers.StructuredOutputParser):
        parser_type = 'structured' 

    for i in range(n_samples):
        output = concept_outputs[i]

        try:
            parsed = parser.parse(output)
            
            if parser_type == 'pydantic':
                score, expl = parsed.score, parsed.reasoning
            
            elif parser_type == 'structured':
                score, expl = parsed['score'], parsed['reasoning']
        
        except Exception as e1:
            # Try to reformat output with higher-capacity LLM
            if fix_with_llm:
                try:
                    fix_parser = output_parsers.OutputFixingParser.from_llm(
                        parser=parser,
                        llm=chat_models.ChatOpenAI(
                            model='gpt-3.5-turbo',
                            temperature=0,
                            openai_api_key=openai_api_key,
                            verbose=verbose
                        )
                    )
                    retry_p = [
                        langchain.schema.HumanMessage(
                            content=fix_parser.retry_chain.prompt.format(
                                instructions=parser.get_format_instructions(),
                                completion=output,
                                error=str(e1)
                            )
                        )
                    ]
                    fixed_output = fix_parser.retry_chain.llm(retry_p).content

                    if verbose:
                        print(f'\n[{concept}] Attempting to parse LLM-fixed output:')
                        print(f'{fixed_output}\n')

                    parsed = parser.parse(fixed_output)

                    if parser_type == 'pydantic':
                        score, expl = parsed.score, parsed.reasoning
                    
                    elif parser_type == 'structured':
                        score, expl = parsed['score'], parsed['reasoning']
                
                # Try manual parsing
                except:
                    if verbose:
                        print(f'[{concept}] Attempting manual parsing...')

                    match = re.search('{.*}', output, re.DOTALL)
                    if match:
                        try:
                            parsed = parser.parse(match.group(0))

                            if parser_type == 'pydantic':
                                score, expl = parsed.score, parsed.reasoning

                            elif parser_type == 'structured':
                                score, expl = parsed['score'], parsed['reasoning']
                        except:
                            parsed = None
                            score, expl = np.nan, match.group(0)
                    else:
                        parsed = None
                        score, expl = np.nan, output
            else:
                if verbose:
                    print(f'[{concept}] Attempting manual parsing...')

                match = re.search('{.*}', output, re.DOTALL)
                if match:
                    try:
                        parsed = parser.parse(match.group(0))

                        if parser_type == 'pydantic':
                            score, expl = parsed.score, parsed.reasoning

                        elif parser_type == 'structured':
                            score, expl = parsed['score'], parsed['reasoning']
                    except:
                        parsed = None
                        score, expl = np.nan, match.group(0)
                else:
                    parsed = None
                    score, expl = np.nan, output

        if parsed is None and verbose:
            print(f'[{concept}] Parsing failed with the following output:')
            print(f'{output}\n')

        scores.append(score)
        expls.append(expl)
    
    # Average score
    try:
        scores = list(map(lambda x: x if isinstance(x,float) else np.nan, scores))
        final_score = np.nanmean(scores)
    except:
        final_score = np.nan
    
    # Print all generated explanations
    print(f'\nVariable: {concept}')
    print(f'Score: {final_score:.2f}')
    
    for i, expl in enumerate(expls):
        print(f'Explanation {i+1}: {expl}')

    return concept, dict(score=float(final_score), expl=expls)

def prompt_llm(
    dataset='mimic-icd',
    pred='ckd',
    llm_model='llama-2-13b-chat',
    data_dir=DATA_DIR,
    prompt_dir=PROMPT_DIR,
    prompt_outdir=PROMPT_OUTDIR, 
    min_score=0,
    max_score=1,
    temperature=0, 
    n_samples=1,
    max_tokens=256,
    verbose=False,
    add_context=False,
    add_examples=False,
    add_expls=False,
    opt_suffix=None, # Optional suffix
    fix_with_llm=True,
    n_gpus=8,
    gpu_memory_utilization=0.75,
    query=None, # Specific concept(s) to selectively query 
    save_output=True
):
    '''Submits the feature selection prompts to a pretrained LLM.'''

    start_time = datetime.now()
    os.makedirs(osp.join(prompt_outdir, f'{dataset}/{llm_model}'), exist_ok=True)

    print(f'\nSubmitting feature concept selection prompts.')
    print(f'- Model: {llm_model}')
    print(f'- Dataset: {dataset}')
    print(f'- Prediction Task: {pred}')
    print(f'- Min Score: {min_score}')
    print(f'- Max Score: {max_score}')
    print(f'- Self-Consistency: {True if temperature > 0 else False}')
    print(f'- Number of Samples: {n_samples}')

    # Load concepts
    if query is None:
        concepts = load_concepts(dataset=dataset, pred=pred, data_dir=data_dir)
    else:
        if isinstance(query, (list,tuple,np.array)):
            concepts = query
        else:
            concepts = [query]

    # Load prompt template and examples
    template = load_template(dataset=dataset, pred=pred, prompt_dir=prompt_dir, llm_model=llm_model)
    context = load_context(dataset=dataset, pred=pred, prompt_dir=prompt_dir)
    examples = load_examples(dataset=dataset, pred=pred, prompt_dir=prompt_dir, add_expls=add_expls)

    # Handle GPT and Llama-2 models separately
    if llm_model in ['gpt-3.5-turbo', 'gpt-4-0613']:
        # Define output parser
        parser = output_parsers.PydanticOutputParser(pydantic_object=FeatureImportance)
        helper = partial(
            parse_and_aggregate, parser=parser, verbose=verbose, fix_with_llm=fix_with_llm
        )

        # Set up prompts
        input_template = HumanMessagePromptTemplate.from_template(template)
        prompt = langchain.prompts.ChatPromptTemplate(
            messages=[input_template],
            input_variables=['concept'],
            partial_variables={
                'min_score': min_score,
                'max_score': max_score,
                'context': (context if add_context else ''),
                'format_instructions': parser.get_format_instructions(),
                'examples': (examples if (add_examples or add_expls) else '')
            }
        )

        # Define LLMChain
        # Reference: https://github.com/langchain-ai/langchain/blob/b5a74fb/libs/langchain/langchain/llms/openai.py#L134
        llm = chat_models.ChatOpenAI(
            model_name=llm_model,
            temperature=temperature,
            openai_api_key=openai_api_key,
            max_tokens=max_tokens,
            n=n_samples, # Number of samples to generate for each concept
            max_retries=15 # Exponential backoff
        )
        llm_chain = langchain.LLMChain(llm=llm, prompt=prompt, verbose=verbose)

        # Generate outputs
        inputs = [{'concept': c} for c in concepts]
        outputs = llm_chain.generate(inputs)
        outputs = [
            [outputs.generations[i][j].message.content for j in range(n_samples)] 
            for i in range(len(outputs.generations))
        ]

        # Parse and aggregate outputs
        out_dict = {c: out for (c, out) in map(helper, concepts, outputs)}

    elif 'llama-2' in llm_model:
        print(f'\nLoading pretrained {llm_model}...')
        
        # NOTE: Fixing n=1, n_samples > 1 handled at the input level
        model_size = int(llm_model.split('-')[2][:-1])
        model_id = HF_LLAMA2_DIR.format(model_size)
        model = vllm.LLM(model_id, tensor_parallel_size=n_gpus, gpu_memory_utilization=gpu_memory_utilization)
        sampling_params = vllm.SamplingParams(n=1, temperature=temperature, max_tokens=max_tokens)

        # Define parser
        response_schemas = [
            output_parsers.ResponseSchema(
                name='reasoning', description='Logical reasoning behind feature importance score', type='str'
            ),
            output_parsers.ResponseSchema(
                name='score', description='Feature importance score', type='float'
            )
        ]
        parser = output_parsers.StructuredOutputParser.from_response_schemas(response_schemas)
        helper = partial(
            parse_and_aggregate, parser=parser, verbose=verbose, fix_with_llm=fix_with_llm
        )

        # Set up prompts
        prompt = langchain.prompts.PromptTemplate(
            template=template, 
            input_variables=['concept'],
            partial_variables={
                'min_score': min_score,
                'max_score': max_score,
                'context': (context if add_context else ''),
                'format_instructions': parser.get_format_instructions(),
                'examples': (examples if (add_examples or add_expls) else '')
            }
        )

        # Generate outputs
        inputs = [
            prompt.format_prompt(concept=c).to_string()
            for c in concepts 
            for _ in range(n_samples)
        ]
        
        outputs = model.generate(inputs, sampling_params)
        outputs = [output.outputs[0].text for output in outputs]
        outputs = [outputs[i:i+n_samples] for i in range(0, len(outputs), n_samples)]

        # Parse and aggregate outputs
        out_dict = {c: out for (c, out) in map(helper, concepts, outputs)}
    else:
        raise RuntimeError(f'[Error] {llm_model} not supported.')

    # Only save when querying for all features
    if query is None and save_output:
        out_suffix = 'greedy' if temperature == 0 else 'consistent'
        out_suffix += '_context' if add_context else ''
        
        if add_expls:
            out_suffix += '_expls'
            
        elif add_examples:
            out_suffix += '_examples'

        # Add optional suffix
        if opt_suffix is not None:
            out_suffix += f'_{opt_suffix}'

        # Default score range is between 0 and 1
        if min_score == 0 and max_score == 1:
            out_name = f'{pred}_{out_suffix}'
        else:
            out_name = f'{pred}_{out_suffix}_{min_score}_{max_score}'

        out_path = osp.join(prompt_outdir, f'{dataset}/{llm_model}/{out_name}.yaml')
        with open(out_path, 'w') as fh:
            yaml.dump(out_dict, fh, default_flow_style=False)
            
        print(f'\nLLM prompt results (.yaml) saved in: {out_path}\n')
            
    print(f'Elapsed: {str(datetime.now() - start_time)}')

    return out_dict

def prompt_llm_rank(
    dataset='mimic-icd',
    pred='ckd',
    llm_model='gpt-3.5-turbo',
    data_dir=DATA_DIR,
    prompt_dir=PROMPT_DIR,
    prompt_outdir=PROMPT_OUTDIR,
    temperature=0,
    n_samples=1,
    max_tokens=1024,
    verbose=False,
    n_gpus=8
):
    '''Submits the feature ranking prompts to a pretrained LLM.'''

    start_time = datetime.now()
    os.makedirs(osp.join(prompt_outdir, f'{dataset}/{llm_model}'), exist_ok=True)

    print(f'\nSubmitting feature ranking prompts.')
    print(f'- Model: {llm_model}')
    print(f'- Dataset: {dataset}')
    print(f'- Prediction Task: {pred}')
    print(f'- Number of Samples: {n_samples}')
    
    # Load concepts
    concepts = load_concepts(dataset=dataset, pred=pred, data_dir=data_dir)

    # Load rank prompt template
    template = load_template(dataset=dataset, pred=pred, prompt_dir=prompt_dir, llm_model=llm_model, rank=True)

    # Handle GPT and Llama-2 models separately
    if llm_model in ['gpt-3.5-turbo', 'gpt-4-0613']:
        # Define output parser
        parser = output_parsers.NumberedListOutputParser()

        # Set up prompts
        input_template = HumanMessagePromptTemplate.from_template(template)
        prompt = langchain.prompts.ChatPromptTemplate(
            messages=[input_template],
            input_variables=['n_concepts', 'concepts'],
            partial_variables={
                'format_instructions': parser.get_format_instructions()
            }
        )

        # Define LLMChain
        # Reference: https://github.com/langchain-ai/langchain/blob/b5a74fb/libs/langchain/langchain/llms/openai.py#L134
        llm = chat_models.ChatOpenAI(
            model_name=llm_model,
            temperature=temperature,
            openai_api_key=openai_api_key,
            max_tokens=max_tokens,
            n=n_samples,
            max_retries=15 # Exponential backoff
        )
        llm_chain = langchain.LLMChain(llm=llm, prompt=prompt, verbose=verbose)

        # Generate outputs
        inputs = [{
            'n_concepts': len(concepts),
            'concepts': '\n'.join([f"{i+1}. {c}" for (i,c) in enumerate(concepts)])
        }]
        outputs = llm_chain.generate(inputs)
        outputs = [
            outputs.generations[i][j].message.content 
            for j in range(n_samples)
            for i in range(len(outputs.generations))
        ]

        # Parse and aggregate ranks
        parsed = [parser.parse(output) for output in outputs] # NOTE: List of lists
        
        ranks = []
        for p in parsed:
            # Ensure outputs match to concepts
            rank = []
            for c_out in p:
                fuzz_scores = [fuzz.ratio(c_out, c) for c in concepts]
                rank.append(concepts[np.argmax(fuzz_scores)])

            ranks.append(rank)

        print(ranks)
        rank_dict = {f'rank_{i+1}': [len(concepts) - r.index(c) for c in concepts] for i,r in enumerate(ranks)}
        rank_df = pd.DataFrame(rank_dict, index=concepts)
        agg_rank = pd.DataFrame(rk.borda(rank_df)).sort_values(by=0).index.to_list()
        
        out_path = osp.join(prompt_outdir, f'{dataset}/{llm_model}/{pred}_rank')
        out_path += '_consistent' if temperature > 0 else ''
        out_path += '.yaml'
        
        with open(out_path, 'w') as fh:
            yaml.dump(agg_rank, fh, default_flow_style=False)

        print(f'LLM prompt results (.yaml) saved in: {out_path}\n')

    elif 'llama-2' in llm_model:
        print(f'\nLoading pretrained {llm_model}...')
        
        # NOTE: Fixing n=1, n_samples > 1 handled at the input level
        model_size = int(llm_model.split('-')[2][:-1])
        model_id = HF_LLAMA2_DIR.format(model_size)
        model = vllm.LLM(model_id, tensor_parallel_size=n_gpus, gpu_memory_utilization=0.6) # trust_remote_code=False
        sampling_params = vllm.SamplingParams(n=1, temperature=temperature, max_tokens=max_tokens)

        # Define parser
        parser = output_parsers.NumberedListOutputParser()

        # Set up prompts
        prompt = langchain.prompts.PromptTemplate(
            template=template, 
            input_variables=['n_concepts', 'concepts'],
            partial_variables={
                'format_instructions': parser.get_format_instructions(),
            }
        )

        # Generate output
        input_p = prompt.format_prompt(
            n_concepts=len(concepts),
            concepts='\n'.join([f"{i+1}. {c}" for (i,c) in enumerate(concepts)])
        ).to_string()
        output = model.generate([input_p], sampling_params)[0].outputs[0].text

        try:
            parsed = parser.parse(output)

            # Ensure outputs match to concepts
            rank = []
            for c_out in parsed:
                fuzz_scores = [fuzz.ratio(c_out, c) for c in concepts]
                rank.append(concepts[np.argmax(fuzz_scores)])

            out_path = osp.join(prompt_outdir, f'{dataset}/{llm_model}/{pred}_rank.yaml')
            with open(out_path, 'w') as fh:
                yaml.dump(rank, fh, default_flow_style=False)

            if verbose and len(rank) != len(concepts):
                print('[Warning] Number of concepts do not match.')
                print(f'Expected={len(concepts)}, Parsed={len(rank)}')
                print(f'Missing: {set(concepts) - set(rank)}')

            print(f'LLM prompt results (.yaml) saved in: {out_path}\n')
        except:
            parsed = None
            out_path = osp.join(prompt_outdir, f'{dataset}/{llm_model}/{pred}_rank.txt')
            with open(out_path, 'w') as fh:
                fh.write(output)

            print(f'Parsing failed. LLM prompt results (.txt) saved in: {out_path}\n')
    else:
        raise RuntimeError(f'[Error] {llm_model} not supported.')
            
    print(f'Elapsed: {str(datetime.now() - start_time)}')
    
    if verbose and parsed is None:
        print('[Warning] Returning LLM unparsed output.')
        
    return parsed if parsed is not None else output
    
def llm_select_concepts(
    llm_output, 
    threshold=None, 
    ratio=None, 
    topk=None, 
    filtered_concepts=None, 
    rank=False,
    seed=DEFAULT_SEED
):
    '''Returns a list of concepts selected by an LLM.'''

    if sum(map(bool, [threshold, ratio, topk])) >= 2:
        raise RuntimeError('[Error] Multiple concept selection options specified.')
    elif sum(map(bool, [threshold, ratio, topk])) == 0:
        raise RuntimeError('[Error] No concept selection option specified.')
    
    if topk is not None:
        if not isinstance(topk, int):
            raise RuntimeError(f'[Error] topk argument must be int. Got: {type(topk)}.')

    if filtered_concepts is not None:
        concepts = np.array(filtered_concepts)
    else:
        concepts = np.array(list(llm_output.keys()))
    
    # Using direct LLM ranking
    if rank:
        # Use fuzzy string matching to ensure correct concept string
        fuzzy_matched = []
        for c in llm_output:
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

        # If there are missing concepts in the ranking, randomly add according to the seed
        missing = list(set(concepts) - set(rank_output))
        random.seed(seed)
        random.shuffle(missing)
        rank_output += missing

        # Ensure that the features match 
        selected_concepts = rank_output[:int(ratio*len(concepts))]
    
    # Selecting concepts with the highest importance scores
    else:
        selected_concepts = []
        scores = []
        for concept in concepts:
            out = llm_output[concept]
            scores.append(out['score'])
            
            if threshold is not None and out['score'] >= threshold:
                selected_concepts.append(concept)

        if ratio is not None:            
            # Random tie-breaking in argsort
            scores = np.array(scores)
            np.random.seed(seed)
            sorted_idxs = np.argsort(scores + np.random.uniform(-1e-10, 1e-10, scores.shape))[::-1]
            top_idxs = sorted_idxs[:int(ratio * len(concepts))]
            selected_concepts = concepts[top_idxs]

        elif topk is not None:
            # Random tie-breaking in argsort
            scores = np.array(scores)
            np.random.seed(seed)
            sorted_idxs = np.argsort(scores + np.random.uniform(-1e-10, 1e-10, scores.shape))[::-1]
            top_idxs = sorted_idxs[:topk]
            selected_concepts = concepts[top_idxs]

    return selected_concepts
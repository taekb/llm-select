# LLM-Select: Feature Selection with Large Language Models
This is the source code repository for the paper: "LLM-Select: Feature Selection with Large Language Models". 

This repository includes implementations of our proposed LLM-Score, LLM-Rank, and LLM-Seq feature selection methods based on GPT-4, GPT-3.5, and Llama-2 (70B/13B/7B), along with all of the prompt templates and Python & bash scripts used for running all of the experiments. In particular, we use the following versions for each model:

- GPT-4: `gpt-4-0613`
- GPT-3.5: `gpt-3.5-turbo`
- Llama-2 (70B): `meta-llama/llama-2-70b-chat-hf`
- Llama-2 (13B): `meta-llama/llama-2-13b-chat-hf`
- Llama-2 (7B): `meta-llama/llama-2-7b-chat-hf`

In our code, we access GPT-4 and GPT-3.5 via the [OpenAI API](https://platform.openai.com/) and the Llama-2 models via [HuggingFace](https://huggingface.co/meta-llama). As such, please make sure to add your OpenAI and HuggingFace API keys in `.txt` format under `./config` according to the `./config/README.txt` file before running the LLM-based feature selection methods.

<br>

## 1. Setup
We provide the exported [conda](https://docs.conda.io/en/latest/) environment `.yaml` file (tested in the Linux environment) to replicate the setting under which all implementations and experiments were run.

### 1.1. Install Conda
If you do not already have conda installed on your local machine, please install conda following the instructions [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

### 1.2. Import Conda Environments
The exported [conda](https://docs.conda.io/en/latest/) environment `.yaml` file is provided under `./conda`. To import and create a new conda environment, run:
```
conda env create -f ./conda/llm-select-env.yaml
```

For additional information, you can also refer to conda's [documentation on managing environments](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

### 1.3. Activate the Conda Environment
After Step 1.2, you can check whether the environment (named `llm-select`) was successfully created by running:
```
conda env list
```
which lists all of the conda environments available on your local machine. If `llm-select` is also listed, then you can activate it by running:
```
conda activate llm-select
```

We note that the [scikit-feature](https://github.com/jundongl/scikit-feature) library, which is a dependency for our code, needs to be installed manually. As described in the linked GitHub repository, this involves cloning the [scikit-feature](https://github.com/jundongl/scikit-feature) repository and running `python3 setup.py install` while the `llm-select` conda environment is activated.

<br><br>

## 2. Running the Code

Below, we provide instructions for running LLM-Score, LLM-Rank, and LLM-Seq for feature selection and replicating the small-scale dataset experiments in Section 4.1 and the large-scale dataset experiments in Section 4.2 of the paper. 

Using LLM-Score and LLM-Rank for feature selection involves a *two-step process*: (i) prompting an LLM to generate the feature importance scores or feature rankings and then (ii) using the highest scoring/ranked features for training a downstream prediction model. So before running the bash scripts for running the [small-scale](#21-smallscale-dataset-experiments) and large-scale dataset experiments, please follow the instructions for running [LLM-Score](#201-running-llmscore) and [LLM-Rank](#202-running-llmrank) first.

For LLM-Seq, which selects features sequentially, training the downstream model occurs simultaneously in an iterative manner: at each iteration of the selection loop, we add a new feature by prompting the LLM and train a downstream model with the updated feature subset. 

See Section 3 of the paper for more details on LLM-Score, LLM-Rank, and LLM-Seq. All of the main utilities for LLM-based feature selection are included under `./llm_select/selection`. All of the prompt templates used for the three methods are included under `./prompts`.

### 2.0.1. Running LLMScore
To run LLM-Score for any of the small-scale or large-scale datasets used in our experiments, move to `./llm_select/selection` and run the following bash script:
```
./get_scores.sh <llm_model> <*datapreds>
```
where <llm_model> should be replaced with one of `gpt-4-0613`, `gpt-3.5-turbo`, `llama-2-70b-chat`, `llama-2-13b-chat`, and `llama-2-7b-chat`, and <*datapreds> should be replaced with a variable number of dataset & prediction task pairs (in the format `<dataset>/<prediction>`), which can take the following values listed below.

Small-Scale Datasets:
- `credit-g/risk`
- `bank/subscription`
- `give-me-credit/delinquency`
- `compas/recid`
- `pima/diabetes`
- `calhousing/price`
- `diabetes/progression`
- `wine/quality`
- `miami-housing/price`
- `cars/price`

Large-Scale Datasets:
- `acs/income`
- `acs/employment`
- `acs/public_coverage`
- `acs/mobility`
- `mimic-icd/ckd`
- `mimic-icd/copd`
- `mimic-icd/hf`

We note that by default, the `./get_scores.sh` bash script runs LLM-Score for all of the 12 prompt design and decoding strategy pairs discussed in Section 4.1, *which can take a long time and incur a lot of cost financially (from OpenAI API usage)*. 

To only use the default prompt template (which contains no dataset-specific context and few-shot examples) and greedy decoding, please comment out all of the lines that run the `prompt_llm.py` script inside `./get_scores.sh` except for the one marked with "Greedy: Default".

All of the parsed LLM-Score results will then be saved under the `./prompt_outputs` folder, which will be automatically created when the script is run for the first time.

### 2.0.2. Running LLM-Rank
To run LLM-Score for any of the small-scale or large-scale datasets used in our experiments, move to `./llm_select/selection` and run the following bash script:
```
./get_ranks.sh <llm_model> <*datapreds>
```
where <llm_model> should be replaced with one of `gpt-4-0613`, `gpt-3.5-turbo`, `llama-2-70b-chat`, `llama-2-13b-chat`, and `llama-2-7b-chat`, and <*datapreds> should be replaced with a variable number of dataset & prediction task pairs (in the format `<dataset>/<prediction>`, which can take the following values listed below.

All of the parsed LLM-Rank results will then be saved under the `./prompt_outputs` folder, which will be automatically created when the script is run for the first time.

### 2.1. Small-Scale Dataset Experiments
To run all of the small-scale dataset experiments, move to `./llm_select` and run the following bash script:
```
./run_linear_compare.sh <*datapreds>
```
where <*datapreds> should be replaced with a variable number of dataset & prediction task pairs (in the format `<dataset>/<prediction>`), taking one of the values listed in the instructions for [LLM-Score](#201-running-llmscore).

All of the results will then be saved under the `./llm_select/results/linear_compare` folder, which will be automatically created when the script is run for the first time.

### 2.2. Large-Scale Dataset Experiments
In the large-scale dataset experiments, we compare our LLM-based feature selection methods against [LassoNet](https://github.com/lasso-net/lassonet) (Lemhadri et al., 2021), [group LASSO](https://github.com/yngvem/group-lasso) (gLASSO; Yuan and Lin, 2005), and [maximum relevance minimum redundancy selection](https://github.com/smazzanti/mrmr) (MRMR; Ding and Peng, 2005). 

Given that regularization path computation and feature selection based on these methods can take a long time for datasets with a large number of features, we precompute the binary feature masks before running the main experiments. To precompute the binary feature masks, move to `./llm_select/selection` and run the following bash scripts:
```
./run_lassonet.sh
./run_glasso.sh
./run_mrmr.sh
```
All of the feature masks (and the computed regularization paths, if applicable) will then be saved under `./llm_select/selection/<method>` (e.g., `./llm_select/selection/lassonet`). These feature masks will then be automatically used by the large-scale dataset experiment scripts.

To run the large-scale dataset experiments, move to `./llm_select` and run the following bash script:
```
./run_large_exp.sh 
```
All of the results will then be saved under `./llm_select/results/benchmark/`, which will be automatically created when the script is run for the first time.

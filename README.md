# LLM-Select: Feature Selection with Large Language Models
This is the source code repository for the paper: "LLM-Select: Feature Selection with Large Language Models". 

This repository includes implementations of our proposed LLM-Score, LLM-Rank, and LLM-Seq feature selection methods based on GPT-4, GPT-3.5, and Llama-2, along with all of the prompts and the Python scripts used for running all of the experiments.

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

We note that the [scikit-feature](https://github.com/jundongl/scikit-feature) library, which is a dependency for our code, needs to be installed manually. As described in the linked GitHub repository, this involves cloning the [scikit-feature](https://github.com/jundongl/scikit-feature) repository and running `python3 setup.py install` while the `llm-select` conda environment activated.

<br><br>

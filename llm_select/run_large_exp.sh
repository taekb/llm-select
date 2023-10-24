#!/bin/bash

start_time=$(date +%s)

#export CUDA_VISIBLE_DEVICES=0,1,2,3

# Create log directories
mkdir -p ./logs/large_exp

# MIMIC-ICD HF 
python3 -u large_exp.py --datapred "mimic-icd/hf" --models linear mlp lgb --fs_methods lassonet glasso llm-ratio:gpt-4-0613:greedy:: --run_locally 2>&1 | tee "./logs/large_exp/mimic-icd_hf.log"

# MIMIC-ICD CKD
python3 -u large_exp.py --datapred "mimic-icd/ckd" --models linear mlp lgb --fs_methods lassonet glasso llm-ratio:gpt-4-0613:greedy:: --run_locally 2>&1 | tee "./logs/large_exp/mimic-icd_ckd.log"

# MIMIC-ICD COPD
python3 -u large_exp.py --datapred "mimic-icd/copd" --models linear mlp lgb --fs_methods lassonet glasso llm-ratio:gpt-4-0613:greedy:: --run_locally 2>&1 | tee "./logs/large_exp/mimic-icd_copd.log"

# ACS Income
python3 -u large_exp.py --datapred "acs/income" --models linear mlp lgb --fs_methods lassonet glasso llm-ratio:gpt-4-0613:greedy:: --run_locally 2>&1 | tee "./logs/large_exp/acs_income.log"

# ACS Employment
python3 -u large_exp.py --datapred "acs/employment" --models linear mlp lgb --fs_methods lassonet glasso llm-ratio:gpt-4-0613:greedy:: --run_locally 2>&1 | tee "./logs/large_exp/acs_employment.log"

# ACS Mobility
python3 -u large_exp.py --datapred "acs/mobility" --models linear mlp lgb --fs_methods lassonet glasso llm-ratio:gpt-4-0613:greedy:: --run_locally 2>&1 | tee "./logs/large_exp/acs_mobility.log"

# ACS Public Coverage
python3 -u large_exp.py --datapred "acs/public_coverage" --models linear mlp lgb --fs_methods lassonet glasso llm-ratio:gpt-4-0613:greedy:: --run_locally 2>&1 | tee "./logs/large_exp/acs_public_coverage.log"

end_time=$(date +%s)
elapsed_seconds=$((end_time - start_time))
hours=$((elapsed_seconds / 3600))
minutes=$(( (elapsed_seconds % 3600) / 60 ))
seconds=$((elapsed_seconds % 60))

printf "\nTotal elapsed time: ${hours} hours, ${minutes} minutes, ${seconds} seconds.\n\n"
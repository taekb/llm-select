#!/bin/bash

if [ "$#" -lt 2 ]; then
    echo "Required arguments missing: arg1={llm_model}, arg2={*datapreds}"
    exit 1
fi

# LLM model to use for extracting the scores
llm_model=$1

# Iterate through all of the dataset + prediction task pairs
for datapred in "${@:2}"; do
    #dataset="${datapred%%/*}"
    IFS='/' read -r dataset pred <<< "$datapred"

    printf "Running prompts with ${llm_model}...\n"
    printf "Dataset: ${dataset}\n"
    printf "Prediction: ${pred}\n"

    start_time=$(date +%s)

    # Create log directory
    logdir="../logs/${dataset}/${llm_model}"
    mkdir -p "$logdir"

    # Greedy with score range 0-10
    python3 -u prompt_llm.py --min_score 0 --max_score 10 --datapreds "$datapred" --llm_model "$llm_model" --n_samples 5 2>&1 | tee "${logdir}/${pred}_greedy_0_10.log"

    # Greedy with score range 8-24
    python3 -u prompt_llm.py --min_score 8 --max_score 24 --datapreds "$datapred" --llm_model "$llm_model" --n_samples 5 2>&1 | tee "${logdir}/${pred}_greedy_8_24.log"

    end_time=$(date +%s)
    elapsed_seconds=$((end_time - start_time))
    hours=$((elapsed_seconds / 3600))
    minutes=$(( (elapsed_seconds % 3600) / 60 ))
    seconds=$((elapsed_seconds % 60))

    printf "\nTotal elapsed time for ${datapred}: ${hours} hours, ${minutes} minutes, ${seconds} seconds.\n\n"
done
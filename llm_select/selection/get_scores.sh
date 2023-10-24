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

    # Greedy: Default
    python3 -u prompt_llm.py --datapreds "$datapred" --llm_model "$llm_model" --n_samples 5 2>&1 | tee "${logdir}/${pred}_greedy.log"

    # Greedy: Default + Examples
    python3 -u prompt_llm.py --datapreds "$datapred" --llm_model "$llm_model" --n_samples 5 --add_examples 2>&1 | tee "${logdir}/${pred}_greedy_examples.log"

    # Greedy: Default + Examples with CoT
    python3 -u prompt_llm.py --datapreds "$datapred" --llm_model "$llm_model" --n_samples 5 --add_expls 2>&1 | tee "${logdir}/${pred}_greedy_expls.log"

    # Greedy: Default + Context
    python3 -u prompt_llm.py --datapreds "$datapred" --llm_model "$llm_model" --n_samples 5 --add_context 2>&1 | tee "${logdir}/${pred}_greedy_context.log"

    # Greedy: Default + Context + Examples
    python3 -u prompt_llm.py --datapreds "$datapred" --llm_model "$llm_model" --n_samples 5 --add_context --add_examples 2>&1 | tee "${logdir}/${pred}_greedy_context_examples.log"

    # Greedy: Default + Context + Examples with CoT
    python3 -u prompt_llm.py --datapreds "$datapred" --llm_model "$llm_model" --n_samples 5 --add_context --add_expls 2>&1 | tee "${logdir}/${pred}_greedy_context_expls.log"
    
    # Self-Consistency: Default
    python3 -u prompt_llm.py --datapreds "$datapred" --llm_model "$llm_model" --temperature 0.5 --n_samples 5 2>&1 | tee "${logdir}/${pred}_consistent.log"

    # Self-Consistency: Default + Examples
    python3 -u prompt_llm.py --datapreds "$datapred" --llm_model "$llm_model" --temperature 0.5 --add_examples --n_samples 5 2>&1 | tee "${logdir}/${pred}_consistent_examples.log"

    # Self-Consistency: Default + Examples with CoT
    python3 -u prompt_llm.py --datapreds "$datapred" --llm_model "$llm_model" --temperature 0.5 --add_expls --n_samples 5 2>&1 | tee "${logdir}/${pred}_consistent_expls.log"

    # Self-Consistency: Default + Context
    python3 -u prompt_llm.py --datapreds "$datapred" --llm_model "$llm_model" --temperature 0.5 --add_context --n_samples 5 2>&1 | tee "${logdir}/${pred}_consistent_context.log"

    # Self-Consistency: Default + Context + Examples
    python3 -u prompt_llm.py --datapreds "$datapred" --llm_model "$llm_model" --temperature 0.5 --add_context --add_examples --n_samples 5 2>&1 | tee "${logdir}/${pred}_consistent_context_examples.log"

    # Self-Consistency: Default + Context + Examples with CoT
    python3 -u prompt_llm.py --datapreds "$datapred" --llm_model "$llm_model" --temperature 0.5 --add_context --add_expls --n_samples 5 2>&1 | tee "${logdir}/${pred}_consistent_context_expls.log"

    end_time=$(date +%s)
    elapsed_seconds=$((end_time - start_time))
    hours=$((elapsed_seconds / 3600))
    minutes=$(( (elapsed_seconds % 3600) / 60 ))
    seconds=$((elapsed_seconds % 60))

    printf "\nTotal elapsed time for ${datapred}: ${hours} hours, ${minutes} minutes, ${seconds} seconds.\n\n"
done
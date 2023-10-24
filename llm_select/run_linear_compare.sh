#!/bin/bash

if [ "$#" -eq 0 ]; then
    echo "Please provide at least one datapred as argument."
    exit 1
fi

start_time=$(date +%s)

# Iterate through all of the dataset + prediction task pairs
for datapred in "$@"; do
    IFS='/' read -r dataset pred <<< "$datapred"

    # Baseline experiment
    echo "Running baseline experiments for $datapred..."
    python3 -u linear_compare.py --datapred "$datapred" --seeds 1 2 3 4 5 --exp_suffix baseline --buffer_chat 2>&1 | tee "./logs/linear_compare_${dataset}_baseline.log"
done

end_time=$(date +%s)
elapsed_seconds=$((end_time - start_time))
hours=$((elapsed_seconds / 3600))
minutes=$(( (elapsed_seconds % 3600) / 60 ))
seconds=$((elapsed_seconds % 60))

printf "\nTotal elapsed time: ${hours} hours, ${minutes} minutes, ${seconds} seconds.\n\n"
#!/bin/bash

start_time=$(date +%s)
echo "Start time: $(date)"

mkdir -p ../logs/glasso

python3 -u glasso_select.py --datapreds mimic-icd/hf > ../logs/glasso/mimic-icd_hf.log 2>&1 & \
python3 -u glasso_select.py --datapreds mimic-icd/ckd > ../logs/glasso/mimic-icd_ckd.log 2>&1 & \
python3 -u glasso_select.py --datapreds mimic-icd/copd > ../logs/glasso/mimic-icd_copd.log 2>&1 & \
python3 -u glasso_select.py --datapreds acs/income > ../logs/glasso/acs_income.log 2>&1 & \
python3 -u glasso_select.py --datapreds acs/employment > ../logs/glasso/acs_employment.log 2>&1 & \
python3 -u glasso_select.py --datapreds acs/public_coverage > ../logs/glasso/acs_public_coverage.log 2>&1 & \
python3 -u glasso_select.py --datapreds acs/mobility > ../logs/glasso/acs_mobility.log 2>&1

end_time=$(date +%s)
elapsed_seconds=$((end_time - start_time))
hours=$((elapsed_seconds / 3600))
minutes=$(( (elapsed_seconds % 3600) / 60 ))
seconds=$((elapsed_seconds % 60))

printf "\nTotal elapsed time: ${hours} hours, ${minutes} minutes, ${seconds} seconds.\n\n"
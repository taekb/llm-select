#!/bin/bash

start_time=$(date +%s)
echo "Start time: $(date)"

mkdir -p ../logs/mrmr

python3 -u mrmr_select.py --datapreds mimic-icd/hf > ../logs/mrmr/mimic-icd_hf.log 2>&1 & \
python3 -u mrmr_select.py --datapreds mimic-icd/ckd > ../logs/mrmr/mimic-icd_ckd.log 2>&1 & \
python3 -u mrmr_select.py --datapreds mimic-icd/copd > ../logs/mrmr/mimic-icd_copd.log 2>&1 & \
python3 -u mrmr_select.py --datapreds acs/income > ../logs/mrmr/acs_income.log 2>&1 & \
python3 -u mrmr_select.py --datapreds acs/employment > ../logs/mrmr/acs_employment.log 2>&1 & \
python3 -u mrmr_select.py --datapreds acs/public_coverage > ../logs/mrmr/acs_public_coverage.log 2>&1 & \
python3 -u mrmr_select.py --datapreds acs/mobility > ../logs/mrmr/acs_mobility.log 2>&1

end_time=$(date +%s)
elapsed_seconds=$((end_time - start_time))
hours=$((elapsed_seconds / 3600))
minutes=$(( (elapsed_seconds % 3600) / 60 ))
seconds=$((elapsed_seconds % 60))

printf "\nTotal elapsed time: ${hours} hours, ${minutes} minutes, ${seconds} seconds.\n\n"
#!/bin/bash

start_time=$(date +%s)
echo "Start time: $(date)"

mkdir -p ../logs/lassonet

python3 -u lassonet_select.py --datapreds mimic-icd/hf --device_idx 0 > ../logs/lassonet/mimic-icd_hf.log 2>&1 & \
python3 -u lassonet_select.py --datapreds mimic-icd/ckd --device_idx 1 > ../logs/lassonet/mimic-icd_ckd.log 2>&1 & \
python3 -u lassonet_select.py --datapreds mimic-icd/copd --device_idx 2 > ../logs/lassonet/mimic-icd_copd.log 2>&1 & \
python3 -u lassonet_select.py --datapreds acs/income --device_idx 3 > ../logs/lassonet/acs_income.log 2>&1 & \
python3 -u lassonet_select.py --datapreds acs/employment --device_idx 4 > ../logs/lassonet/acs_employment.log 2>&1 & \
python3 -u lassonet_select.py --datapreds acs/public_coverage --device_idx 5 > ../logs/lassonet/acs_public_coverage.log 2>&1 & \
python3 -u lassonet_select.py --datapreds acs/mobility --device_idx 6 > ../logs/lassonet/acs_mobility.log 2>&1

end_time=$(date +%s)
elapsed_seconds=$((end_time - start_time))
hours=$((elapsed_seconds / 3600))
minutes=$(( (elapsed_seconds % 3600) / 60 ))
seconds=$((elapsed_seconds % 60))

printf "\nTotal elapsed time: ${hours} hours, ${minutes} minutes, ${seconds} seconds.\n\n"
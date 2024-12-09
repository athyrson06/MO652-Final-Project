#!/bin/bash

# Define parameter grids
SEEDS=({1..20})
FRACTIONS=(1 2 4 8 16 32 64)
NOISE_LEVELS=(0)
USECLASS=(True)
MODELS=(1)
DROPS=(0)
N_JOBS=(1 2 4 8 16 32 64) # Add the n_jobs parameter to the grid

# Path to dataset
DATA_PATH="data/pha-asteroids.csv"

# Number of parallel combinations to run at the same time
MAX_PARALLEL=1

# Log file for successful runs
LOG_FILE="successful_runs_RF.log"
echo "Run Log - $(date)" > "$LOG_FILE"

# Initialize the process counter
counter=0

# Loop through parameter combinations
for seed in "${SEEDS[@]}"; do
  for fraction in "${FRACTIONS[@]}"; do
    for noise in "${NOISE_LEVELS[@]}"; do
      for useclass in "${USECLASS[@]}"; do
        for model in "${MODELS[@]}"; do
          for drop in "${DROPS[@]}"; do
            for n_jobs in "${N_JOBS[@]}"; do
              # Construct the command
              CMD="python modelRunRF.py -s $seed -f $fraction -d $DATA_PATH -n $noise -c $useclass -m $model -k $drop -j $n_jobs"
              
              # Run the command
              eval $CMD
              
              # Check exit status of the last command
              if [ $? -eq 0 ]; then
                # Log the successful run
                echo "SUCCESS: $CMD" | tee -a "$LOG_FILE"
              else
                # Log the failed run (optional, for debugging)
                echo "FAILURE: $CMD" >&2
              fi
              
              # Increment counter and manage parallelism
              ((counter++))
              if (( counter % MAX_PARALLEL == 0 )); then
                wait
              fi
            done
          done
        done
      done
    done
  done
done

# Wait for any remaining processes to finish
wait

echo "Completed! Successful runs logged in $LOG_FILE."

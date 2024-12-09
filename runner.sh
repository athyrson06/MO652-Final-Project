#!/bin/bash

# Define parameter grids
SEEDS=(42 123 314)
FRACTIONS=(1 2 4)
NOISE_LEVELS=(0 1 2 3)
USECLASS=(True False)
MODELS=(0 1 2 3 4 5 6)
DROPS=(0 1 2 3 4)

# SEEDS=(42 314 123)
# FRACTIONS=(1)
# NOISE_LEVELS=(0)
# USECLASS=(True)
# MODELS=(0)
# DROPS=(0)

# Path to dataset
DATA_PATH="data/pha-asteroids.csv"

# Number of processes per command
NUM_PROCESSES=20

# Number of parallel combinations to run
MAX_PARALLEL=3

# Log file for successful runs
LOG_FILE="successful_runs.log"
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
            # Construct the command
            CMD="mpiexec -n $NUM_PROCESSES python modelRun.py -s $seed -f $fraction -d $DATA_PATH -n $noise -c $useclass -m $model -k $drop"
            
            # Run the command
            eval $CMD

            # # Run the command in the background
            # echo "Running: $CMD"

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

# Wait for any remaining processes to finish
wait

echo "Grid search completed! Successful runs logged in $LOG_FILE."

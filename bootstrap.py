import os
from mpi4py import MPI
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import numpy as np
from sklearn.svm import SVC
import time
from argparse import ArgumentParser

# Parse arguments at the start
parser = ArgumentParser()
parser.add_argument("-s", "--seed", dest="seed", help="Random seed", default=42, type=int)
parser.add_argument("-f", "--fraction", dest="fraction", help="Fraction of dataset each rank uses. Ex: for 1/4 use -f 4", default=125, type=int)
parser.add_argument("-d", "--data", dest="data", help="Path to the dataset", default="data/pha-asteroids.csv")
args = parser.parse_args()

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Broadcast arguments from rank 0
if rank == 0:
    print("Rank 0: Broadcasting arguments...")
args = comm.bcast(args, root=0)

# Rank 0 loads the dataset
if rank == 0:
    print(f"Rank 0: Loading dataset from {args.data}...")
    try:
        df = pd.read_csv(args.data, index_col=0)
        print("Rank 0: Dataset loaded successfully!")
    except Exception as e:
        print(f"Rank 0: Error loading dataset: {e}")
        df = None
else:
    df = None

# Broadcast the dataset to all ranks
df = comm.bcast(df, root=0)

# Check if the dataset was loaded correctly
if df is None:
    print(f"Rank {rank}: Failed to load dataset. Exiting...")
    MPI.Finalize()
    exit()

# Seed consistency across ranks
print(f"Rank {rank}: Using random seed = {args.seed}")

# Separate features and target
X = df.drop(["pha", "class"], axis=1)
y = df["pha"]

# Split the data into training and test sets (all ranks use the same split for consistency)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed)

# Define parameter grid for GridSearchCV
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False]
}

# Measure execution time
start_time = time.time()

# Each rank generates its own bootstrap sample
rank_seed = args.seed + rank
rank_sample_size = int(X_train.shape[0] / args.fraction)
X_resampled, y_resampled = resample(X_train, y_train, 
                                    n_samples=rank_sample_size, random_state=rank_seed)

# Perform GridSearchCV on the resampled data
rf = RandomForestClassifier(random_state=rank_seed)
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring="roc_auc", verbose=0)
grid_search.fit(X_resampled, y_resampled)

# Get the best model and evaluate it on the test set
best_model = grid_search.best_estimator_
y_prob = best_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)

# Execution time for this rank
execution_time = time.time() - start_time

# Gather results (rank, AUC, execution time, best parameters) at rank 0
results = comm.gather((rank, auc, execution_time, grid_search.best_params_), root=0)

# Rank 0 aggregates and processes the results
if rank == 0:
    print("\nBootstrapping and Bagging with GridSearchCV Results:")

    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    with open("results/grid_search_results_rf.txt", "w") as f:
        for result in sorted(results):
            print(f"Rank {result[0]}: AUC = {result[1]:.4f} in {result[2]:.2f} seconds.")
            print(f"  Best Params: {result[3]}")
            
            # Write to the results file
            f.write(f"Rank {result[0]}, AUC = {result[1]:.4f}, Time = {result[2]:.2f}s, Best Params = {result[3]}\n")
    
    print("GridSearchCV and bagging completed.")

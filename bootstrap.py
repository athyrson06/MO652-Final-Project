from mpi4py import MPI
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.svm import SVC
import time

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Rank 0 loads the dataset
if rank == 0:
    print("Rank 0: Loading dataset...")
    try:
        data_path = "data/pha-asteroids.csv"
        df = pd.read_csv(data_path, index_col=0)
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

# Separate features and target
X = df.drop(["pha", "class"], axis=1)
y = df["pha"]

# Split the data into training and test sets (all ranks use the same split for consistency)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Each rank generates its own bootstrap sample
X_resampled, y_resampled = resample(X_train, y_train, 
                                    n_samples=int(y_train.shape[0]/512), random_state=rank)

# Medir o tempo de execução
start_time = time.time()

# Train a base model (Decision Tree) on the bootstrap sample
model = SVC(kernel="linear", probability=True, random_state=42)
model.fit(X_resampled, y_resampled)

# Make predictions on the test set
# y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[::,1]
auc = roc_auc_score(y_test, y_prob)

# Tempo total de execução para este rank
execution_time = time.time() - start_time

# Tempo total de execução para este rank
execution_time = time.time() - start_time

# Gather results (model and accuracy) at rank 0
results = comm.gather((rank, auc, execution_time), root=0)

# Rank 0 aggregates the results
# print(X_train.shape)
# print("shape of resampled data: ", X_resampled.shape)
if rank == 0:
    print("\nBootstrapping and Bagging Results:")
    for result in sorted(results):
        # print(f"Rank {result[0]}: Accuracy = {result[1]:.4f}")
        print(f"Rank {result[0]}: AUC = {result[1]:.4f} in {result[2]:.2f} seconds.")
    print("Bagging ensemble completed.")

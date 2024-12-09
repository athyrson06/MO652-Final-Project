import os
import pandas as pd
import numpy as np
import time
from argparse import ArgumentParser
import sys
from mpi4py import MPI

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import TargetEncoder


start_time = time.time()

# Parse arguments at the start
parser = ArgumentParser()
parser.add_argument("-s", "--seed", dest="seed", help="Random seed", default=42, type=int)
parser.add_argument("-f", "--fraction", dest="fraction", help="Fraction of dataset each rank uses. Ex: for 1/4 use -f 4", default=125, type=int)
parser.add_argument("-d", "--data", dest="data", help="Path to the dataset", default="data/pha-asteroids.csv")
parser.add_argument("-n", "--noise", dest="noise", help="Noise level", default=0, type=int)
parser.add_argument("-c", "--useclass", dest="useclass", help="Use class column", default=False, type=bool)
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
    print(args)
    try:
        df = pd.read_csv(args.data, index_col=0)
        classCol = df['class']

        # Separate features and target
        X = df.drop(['pha', 'class', 'neo'], axis=1)
        y = df['pha']

    # Split the data into training and test sets (all ranks use the same split for consistency)
        X_train, X_test, y_train, y_test, C_train, C_test = train_test_split(X, y, classCol, test_size=0.2, random_state=args.seed)
        data = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test, 'C_train': C_train, 'C_test': C_test}


        model_dict = {
            "LogisticRegression": LogisticRegression(random_state=args.seed, max_iter=1000),
            "RandomForest": RandomForestClassifier(random_state=args.seed),
            "GradientBoosting": GradientBoostingClassifier(random_state=args.seed),
            "AdaBoost": AdaBoostClassifier(algorithm='SAMME', random_state=args.seed),
            "Decision Tree": DecisionTreeClassifier(random_state=args.seed),
            "NeuralNetwork": MLPClassifier(random_state=args.seed),
            "SVM": SVC(random_state=args.seed, probability=True),
        }

        keys = list(model_dict.keys())
        keys = keys * (size//len(keys))
        models = [keys[i::size] for i in range(size)]
        # print(models)
    except Exception as e:
        print(f"Rank 0: Error loading dataset: {e}")
else:
    df = None
    data = None
    model_dict = None
    models = None

# Broadcast the dataset to all ranks
df = comm.bcast(df, root=0)
data = comm.bcast(data, root=0)
models_dict = comm.bcast(model_dict, root=0)

# Scatter the keys to all ranks
model_name = comm.scatter(models, root=0)[0]
model = models_dict[model_name]

# print(model_name)
# exit()

# Check if the dataset was loaded correctly
if df is None:
    print(f"Rank {rank}: Failed to load dataset. Exiting...")
    MPI.Finalize()
    exit()


# Get the data from the broadcast
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']

if args.noise > 0:
    mu, sigma = 0, args.noise
    np.random.seed(args.seed)
    # creating a noise with the same dimension as the dataset
    noise = np.random.normal(mu, sigma, X_train.shape)

    X_train = data['X_train'] + noise

if args.useclass:
    C_train = data['C_train']
    C_test = data['C_test']

    X_train = pd.concat([X_train, C_train], axis=1)
    X_test = pd.concat([X_test, C_test], axis=1)

    encoder = TargetEncoder().set_output(transform="pandas")
    X_train = encoder.fit_transform(X_train, y_train)
    X_test = encoder.transform(X_test)

# Measure execution time
start_time = time.time()

# Each rank generates its own bootstrap sample
rank_seed = args.seed + rank
rank_sample_size = int(X_train.shape[0] / args.fraction)
X_resampled, y_resampled = resample(X_train, y_train, 
                                    n_samples=rank_sample_size, random_state=rank_seed, stratify=y_train)

# Perform GridSearchCV on the resampled data
model.fit(X_resampled, y_resampled)


y_prob = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)

# Execution time for this rank
execution_time = time.time() - start_time

# Gather results (rank, AUC, execution time, best parameters) at rank 0
results = comm.gather((model_name, rank, auc, execution_time), root=0)

# Rank 0 aggregates and processes the results
if rank == 0:
    print("\Results:")
    # print(f"size = {size}")
    # Convert results to a pandas DataFrame
    results_df = pd.DataFrame(
        results, 
        columns=["Model", "Rank", "AUC", "ExecutionTime"]
    )

    # Display results
    for _, row in results_df.iterrows():
        print(f"Rank {row['Rank']:.0f}: AUC = {row['AUC']:.4f} in {row['ExecutionTime']:.2f} seconds for Model {row['Model']}.")

    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)

    # Save results to a CSV file
    results_path = f"results/grid_search_results_{args.seed}_{args.fraction}_{args.useclass}_{args.noise}.csv"
    results_df.round(3).to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    print("Completed.")

end_time = time.time()
# print(f"Rank {rank} finished in {end_time - start_time:.2f} seconds.")
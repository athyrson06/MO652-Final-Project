from mpi4py import MPI
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import time

# Inicializar o ambiente MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Caminho para o dataset
data_path = "data/pha-asteroids.csv"

# Apenas o rank 0 lê o arquivo
if rank == 0:
    try:
        print("Rank 0: Carregando os dados...")
        df = pd.read_csv(data_path, index_col=0)
        print("Rank 0: Dados carregados com sucesso!")
    except FileNotFoundError:
        print(f"Rank 0: Arquivo '{data_path}' não encontrado. Certifique-se de que o caminho está correto.")
        df = None
    except Exception as e:
        print(f"Rank 0: Ocorreu um erro ao carregar os dados: {e}")
        df = None
else:
    df = None

# Rank 0 distribui o DataFrame para todos os ranks
df = comm.bcast(df, root=0)

# Verifica se o DataFrame foi carregado corretamente
if df is None:
    print(f"Rank {rank}: Falha ao carregar o DataFrame. Finalizando.")
    MPI.Finalize()
    exit()

# Divisão de recursos e rótulos
X = df.drop(["pha", "class"], axis=1)
y = df["pha"]

# Máximo de classificadores
MAX_CLASSIFIERS = 4

# Garante que não há mais ranks do que classificadores
if size > MAX_CLASSIFIERS:
    if rank == 0:
        print(f"Este script suporta no máximo {MAX_CLASSIFIERS} ranks.")
    MPI.Finalize()
    exit()

# Definir os classificadores disponíveis
classifiers = [
    ("RandomForest", RandomForestClassifier(n_estimators=100, random_state=42)),
    ("SVM", SVC(kernel="linear", probability=True, random_state=42)),
    ("KNN", KNeighborsClassifier(n_neighbors=5)),
    ("LogisticRegression", LogisticRegression(random_state=42, max_iter=1000))
]

# Garantir que não excedemos o número de classificadores
if rank >= len(classifiers):
    print(f"Rank {rank} não será usado, pois há apenas {len(classifiers)} classificadores.")
    MPI.Finalize()
    exit()

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Medir o tempo de execução
start_time = time.time()

# Cada rank treina seu próprio classificador
clf_name, clf = classifiers[rank]
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Tempo total de execução para este rank
execution_time = time.time() - start_time

# Coletar os resultados em todos os ranks
results = comm.gather((rank, clf_name, accuracy, execution_time), root=0)

# O rank 0 exibe os resultados
if rank == 0:
    print("Resultados dos classificadores:")
    for result in sorted(results):
        print(f"Rank {result[0]} - {result[1]}: Acurácia = {result[2]:.4f}, Tempo = {result[3]:.2f} segundos")

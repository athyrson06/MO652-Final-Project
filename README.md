# MO652 Final Project

This repository contains the implementation, data, and analysis for the final project of the MO652 course, focusing on high-performance computing (HPC) techniques applied to the classification of potentially hazardous asteroids (PHAs).

The original dataset can be found at [Kaggle](https://www.kaggle.com/datasets/sakhawat18/asteroid-dataset). 
---

## Repository Structure

### Root Directory
- **`data`**: Contains the datasets and visualizations derived from data analysis.
- **`first_exps (ignore)`**: Early experimental scripts and notebooks. Useful for reference but not critical for the final project.
- **`results`**: Contains the results of experiments, including metrics and visualizations.
- **`data_analyses.ipynb`**: Jupyter notebook for data analysis and exploration.
- **`LICENSE`**: Project licensing information.
- **`local.sh`**: Helper script for local environment setup.
- **`modelRun.py`**: Script for running classification models.
- **`modelRunRF.py`**: Script for running experiments with the Random Forest model.
- **`playground-mpi.ipynb`**: Notebook for experimenting with MPI (Message Passing Interface).
- **`README.md`**: Project documentation.
- **`requirements.txt`**: Dependencies required for the project.
- **`runner.sh`**: Script to execute experiments for multiple configurations.
- **`runner_RF.sh`**: Script to run experiments specific to the Random Forest classifier.

### `data` Directory
- **`asteroid-dataset.csv`**: Primary dataset containing asteroid data.
- **`pha-asteroids.csv`**: Filtered dataset focusing on PHAs.
- **`3D PCA of Asteroid Dataset.png`**: PCA visualization of the asteroid dataset.
- **`corr_matrix.png`**: Correlation matrix visualization of features.
- **`corr_matrix_final.png`**: Finalized correlation matrix used for feature selection.
- **`pha_in_each_class.png`**: Distribution of PHAs across classes.
- **`r_regression.png`**: Regression analysis visualization.

### `first_exps (ignore)` Directory
Contains exploratory scripts and notebooks:
- **`bootstrap.py`**, **`boottwo.py`**, **`gridsearch.py`**, **`mpi-test.py`**, **`noise.py`**: Early experimental scripts.
- **`playground-first try.ipynb`**: Initial notebook for exploratory data analysis.
- **`playground-pha.ipynb`**: Notebook focusing on PHAs.

### `results` Directory
- **`grid_search_results`**: Stores results from grid search experiments.
  - **CSV Files**:
    - **`RF.csv`**, **`mean.csv`**: Metrics from Random Forest experiments.
  - **Visualizations**:
    - **`AUC_LOSS médio para 20 Execuções using RandomForest.png`**
    - **`AUC médio para 20 Execuções em paralelo.png`**
    - **`EFFICIENCY médio para 20 Execuções.png`**
    - **`LOSS_AUC médio para 20 Execuções em paralelo.png`**
    - **`SPEEDUP médio para 20 Execuções.png`**
    - **`SPEEDUP médio para 20 Execuções em paralelo.png`**
    - **`TIME médio para 20 Execuções.png`**
    - **`TIME médio para 20 Execuções em paralelo.png`**
    - **`Trade-off between LOSS_AUC and SPEEDUP.png`**
  - **`notebook final.ipynb`**: Jupyter notebook summarizing the results and findings.

---

## Getting Started

### Prerequisites
- Python 3.8+
- Required libraries specified in `requirements.txt`.

### Installation
1. Clone the repository:
   ```bash
    git clone https://github.com/athyrson06/MO652-Final-Project
2. Navigate to the project directory:
    ```bash
    cd MO652-Final-Project
3. Install dependencies:
    ```bash
    pip install -r requirements.txt


# CAST AI Take-Home Task

## Introduction

In this task we are presented with an anonymised dataset containing over 300 columns of data, with hundreds of thousands of data points. We are tasked with creating a classification model towards the label/target presented in the data. Given that the data is anonymised, and that we have restricted access to domain expertise for the data/business processes which created this dataset, we will be taking a more mechanical approach to this assignment, acknowledging that more in-depth data cleaning, and specialised feature engineering can be achieved on real data, with access to domain experts who can assist in decoding and interpreting data, its values, its irregularities, and so on.

## Folder Structure

```bash
CastAITask/ 
├── data/ # Folder to store datasets 
│ └── data.parquet # Download and store the data.parquet file here 
├── docs/ # Documentation and assets 
│ └── example_confusion_matrix.png # Example confusion matrix screenshot 
├── outputs/ # Outputs from models and EDA 
│ └── EDA/ # Additional helper plots for the .ipynb and README.md
│ └── mlruns/ # Folder used for experiment results of the MLflow experiment tracking
├── report/ # Source code for models and utilities
│ └── REPORT.md # The final report with summary, to accompany experiments.ipynb 
├── src/ # Source code for models and utilities 
│ ├── datautils/ # data utilities
│ │ └── ClassRebalancer.py # utility to perform class re-balancing to address imbalance
│ │ └── DataSampler.py # utility to sample down the data
│ ├── features/ # Feature engineering implementation
│ │ └── FeatureEngineer.py 
│ ├── models/ # Our neural network models/classifiers built using PyTorch
│ │ └── NeuralNets.py 
│ ├── utils/ # Utility functions 
│ │ └── EDA.py # EDA helper functions
│ └── experiments.ipynb # Recipe to run the experiments and to reproduce results
├── tests/ # Unit tests for the code
```

## Report and Discussion

To see what has been done in this project, and for a summary of the results, with discussion, please see the `src/experiments.ipynb` Jupyter notebook, along with the `report/REPORT.md` markdown file. The Jupyter notebook is intended to complement the report. See the installation instructions below for how to setup MLflow to view the results of the experiments, along with the artifacts.

## Installation

### Prerequisites

- Python 3.7 or higher (Python 3.12 recommended)
- Virtual environment (recommended)
- Optional: CUDA 12.X for running Torch models with GPU support

### Install Instructions

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/BrutishGuy/CastAITask.git
   cd CastAITask
   ``` 
2. **Set Up a Virtual Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
    Or, if you are on Windows,
    ```bash
    python3 -m venv venv
    .venv\Scripts\activate
    ```
3. **Install Required Packages:**
    ```bash
    pip3 install -r requirements.txt
    ```
4. **Download the Data**
    For this step, download the `data.parquet` file provided for the task, and store it in the `./data` folder, according to the folder structure described above.

### MLflow Setup

Installing via the `requirements.txt` file using pip, as mentioned above, should install the mlflow package for you already. If not, for some reason, then please install mlflow by running a simple `pip install mlflow` in your virtual env.

Then, make sure that your virtual environment is activated. Once done, simply change directory to the `outputs/` directory, where the `mlruns/` sub-directory exists and where the MLflow experiments/runs are stored. Then, start MLflow. The commands, assuming you are in the root project directory are:

```bash
cd outputs/
mlflow ui
```

### How to Run the Analysis

#### Running the Logistic Regression, Random Forest and BERT Experiments with Various Feature Extraction Techniques
To run the experiments, follow along in the Jupyter notebook, which contains the recipes for the exploratory data analysis (EDA), the feature engineering, the modelling experiments (tracked via MLflow), and discussion of results. This is located in the `experiments.ipynb` notebook in the root of the repository.

### Further Code Explanation

#### src/utils/EDA.py

These helper utilities contain the EDA helper functions for the project to produce:

- Descriptive statistics 
- Plots for the correlation matrix and feature correlations to the target
- Exploratory PCA and plots relating to the explained variance and to the loadings vs. raw features
- Misc. utilities for missing values and more

#### src/features/FeatureEngineer.py
This implements feature extraction for time-based features, transformations (log and quantile), imputation, scaling and more

#### src/models/NeuralNets.py
Defines a basic neural network and a more advanced neural network using PyTorch, and scikit-learn style classifier mixins to be used in pipelines for convenience

#### src/optimiser/ModelOptimiser.py
Implements the hyperopt hyperparameter Bayesian optimisation, together with MLflow experiment tracking, as a scikit-learn style model mixin for convenience

#### src/datautils/ClassRebalancer.py
Implements class rebalancing as a scikit-learn style mixin for pipelines.

#### src/datautils/DataSampler.py
This class downsamples the original dataset owing to memory restrictions on the local machine this project was run on
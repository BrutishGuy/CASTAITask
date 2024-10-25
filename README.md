# CAST AI Take-Home Task

## Project Overview

This project aims to analyze the sentiment of Twitter comments. We utilize both classical NLP and ML approaches, such as TF-IDF and Word2Vec for feature extraction, logistic regression and random forests for modelling, and advanced language models, such as BERT, to classify the sentiment of comments as positive, negative, neutral, or irrelevant.

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
3. **Install Required Packages:**
    ```bash
    pip3 install -r requirements.txt
    ```
4. **Download the Data**
    For this step, download the `data.parquet` file provided for the task, and store it in the `./data` folder, according to the folder structure described above.

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
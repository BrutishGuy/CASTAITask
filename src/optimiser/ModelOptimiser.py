import mlflow
import mlflow.sklearn
from hyperopt import fmin, tpe, Trials, STATUS_OK
from sklearn.model_selection import cross_validate
from mlflow.exceptions import MlflowException
import numpy as np 
import pandas as pd 

class ModelOptimiser:
    def __init__(self, 
                 X_train: pd.DataFrame | np.ndarray, 
                 y_train: pd.Series | np.ndarray, 
                 search_space: dict, 
                 model_constructor: callable, 
                 experiment_name: str,
                 experiment_description: str | dict,
                 experiment_tags: dict,
                 X_val: pd.DataFrame | np.ndarray = None,
                 y_val: pd.Series | np.ndarray = None,
                 scoring: dict = {"roc_auc": "roc_auc", "precision": "precision", "recall": "recall", "f1": "f1"}, 
                 cv: int = 3, 
                 max_evals: int = 50,
                 error_score_behaviour: str | int | float = "raise",
                 verbose: bool = False):
        """
        Initialize the ModelOptimiser.

        Parameters:
        -----------
        X_train : pd.DataFrame | np.ndarray
            The training features.
        y_train : pd.Series | np.ndarray
            The training labels.
        search_space : dict
            The hyperparameter search space for the model.
        model_constructor : callable
            A function that constructs the model, given hyperparameters.
        experiment_name : str
            A unique name for the experiment to provide to mlflow when creating the experiment.
        experiment_description : str
            A short description of the experiment to provide to mlflow when creating the experiment.
        experiment_tags : dict
            A dict containing tags which should be associated with the experiment.
        X_val : pd.DataFrame | np.ndarray
            The validation features.
        y_val : pd.Series | np.ndarray
            The validation labels.
        scoring : str
            The scoring metric (default is a dict, {"roc_auc": "roc_auc", "precision": "precision", "recall": "recall", "f1": "f1"}).
        cv : int
            Number of cross-validation folds (default is 3).
        max_evals : int
            Maximum number of Hyperopt evaluations (default is 50).
        error_score_behaviour : str | int | float
            The behaviour the cross_validate method from sklearn (https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.cross_validate.html) 
            should have when there is an error during fitting (defatul is "raise").
        verbose : str
            Whether to print log messages during the course of optimisation (default is False)
        """
        self.X_train = X_train
        self.y_train = y_train
        self.search_space = search_space
        self.model_constructor = model_constructor
        self.scoring = scoring
        self.cv = cv
        self.max_evals = max_evals
        self.trials = Trials()
        self.experiment_description = experiment_description
        self.experiment_tags = experiment_tags
        self.experiment_name = experiment_name 
        self.X_val = X_val 
        self.y_val = y_val
        self.verbose = verbose
        self.error_score_behaviour = error_score_behaviour
        # create the cxperiment, providing a unique name, if it doesn't exist already

        try:
            self.logreg_experiment_def = mlflow.create_experiment(
                name=self.experiment_name, tags=experiment_tags
            )
        except MlflowException:
            print("Experiment already exists... skipping creation and setting as experiment instead.")

        self.logreg_experiment = mlflow.set_experiment(self.experiment_name)


    def objective(self, params: dict) -> dict:
        """
        The objective function for Hyperopt.

        Parameters:
        -----------
        params : dict
            Hyperparameters for the model.

        Returns:
        --------
        dict : loss and status of the optimization.
        """
        with mlflow.start_run(nested=True):           
            # construct the model with hyperparams
            model = self.model_constructor(params)
            # evaluate it using CV with self.cv folds and self.scoring metrics

            if self.X_val is not None and self.y_val is not None:
                scores = cross_validate(model, self.X_train, self.y_train, cv=self.cv, scoring=self.scoring, params={"X_val": self.X_val, "y_val": self.y_val}, error_score=self.error_score_behaviour)
            else:
                scores = cross_validate(model, self.X_train, self.y_train, cv=self.cv, scoring=self.scoring, error_score=self.error_score_behaviour)
            
            if self.verbose:
                print(scores)
            
            metrics = {"ROCAUC": scores["test_roc_auc"].mean(), 
                       "Precision": scores["test_precision"].mean(), 
                       "Recall": scores["test_recall"].mean(),
                       "F1": scores["test_f1"].mean()}

            # log params and metrics in MLflow
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

            return {'loss': -scores["test_roc_auc"].mean(), 'status': STATUS_OK}

    def optimize(self) -> dict:
        """
        Run the optimization process with Hyperopt.

        Returns:
        --------
        dict : Best hyperparameters found during optimization.
        """
        best = fmin(fn=self.objective,
                    space=self.search_space,
                    algo=tpe.suggest,
                    max_evals=self.max_evals,
                    trials=self.trials)
        return best

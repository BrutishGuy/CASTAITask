import mlflow
import mlflow.sklearn
from hyperopt import fmin, tpe, Trials, STATUS_OK
from sklearn.model_selection import cross_validate
from mlflow.exceptions import MlflowException

class ModelOptimiser:
    def __init__(self, 
                 X_train, 
                 y_train, 
                 search_space, 
                 model_constructor, 
                 experiment_name,
                 experiment_description,
                 experiment_tags,
                 X_val = None,
                 y_val = None,
                 scoring={"roc_auc": "roc_auc", "precision": "precision", "recall": "recall", "f1": "f1"}, 
                 cv=3, 
                 max_evals=50):
        """
        Initialize the ModelOptimiser.

        Parameters:
        -----------
        X_train : pd.DataFrame
            The training features.
        y_train : pd.Series
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
        scoring : str
            The scoring metric (default is a dict, {"roc_auc": "roc_auc", "precision": "precision", "recall": "recall", "f1": "f1"}).
        cv : int
            Number of cross-validation folds (default is 3).
        max_evals : int
            Maximum number of Hyperopt evaluations (default is 50).
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
        # create the cxperiment, providing a unique name, if it doesn't exist already

        try:
            self.logreg_experiment_def = mlflow.create_experiment(
                name=self.experiment_name, tags=experiment_tags
            )
        except MlflowException:
            print("Experiment already exists... skipping creation and setting as experiment instead.")

        self.logreg_experiment = mlflow.set_experiment(self.experiment_name)


    def objective(self, params):
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
                scores = cross_validate(model, self.X_train, self.y_train, cv=self.cv, scoring=self.scoring, params={"X_val": self.X_val, "y_val": self.y_val}, error_score="raise")
            else:
                scores = cross_validate(model, self.X_train, self.y_train, cv=self.cv, scoring=self.scoring, error_score="raise")
                    
            metrics = {"ROCAUC": scores["test_roc_auc"].mean(), 
                       "Precision": scores["test_precision"].mean(), 
                       "Recall": scores["test_recall"].mean(),
                       "F1": scores["test_f1"].mean()}

            # log params and metrics in MLflow
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

            return {'loss': -scores["test_roc_auc"].mean(), 'status': STATUS_OK}

    def optimize(self):
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

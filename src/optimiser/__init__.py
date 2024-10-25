import mlflow
import mlflow.sklearn
from hyperopt import fmin, tpe, Trials, STATUS_OK
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

class ModelOptimiser:
    def __init__(self, X_train, y_train, search_space, model_constructor, scoring='roc_auc', cv=3, max_evals=50):
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
        scoring : str
            The scoring metric (default is 'roc_auc').
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
            # Construct the model with the current set of hyperparameters
            model = self.model_constructor(params)

            # Create a pipeline with preprocessing and feature engineering
            pipeline = Pipeline([
                ('preprocessing', preprocessor),
                ('feature_engineering', feature_engineer),
                ('classifier', model)
            ])

            # Apply class rebalancing to the training set
            X_train_resampled, y_train_resampled = class_rebalancer.transform(self.X_train, self.y_train)

            # Evaluate the model using cross-validation
            score = cross_val_score(pipeline, X_train_resampled, y_train_resampled, cv=self.cv, scoring=self.scoring).mean()

            # Log the parameters and score in MLflow
            mlflow.log_params(params)
            mlflow.log_metric(self.scoring, score)

            return {'loss': -score, 'status': STATUS_OK}

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

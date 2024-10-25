from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
import pandas as pd
from typing import Tuple

class ClassRebalancer(TransformerMixin, BaseEstimator):
    """
    A transformer mixin for handling class imbalance using oversampling or undersampling.

    """
    
    def __init__(self, method: str = 'undersample'):
        self.method = method
    
    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series) -> 'ClassRebalancer':
        # no actual fitting required, just storing the method
        return self
    
    def transform(self, X: pd.DataFrame | np.ndarray, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply class balancing (oversampling or undersampling) to the data provided


        """
        if self.method == 'oversample':
            sampler = RandomOverSampler(random_state=42)
        elif self.method == 'undersample':
            sampler = RandomUnderSampler(random_state=42)
        
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        return X_resampled, y_resampled

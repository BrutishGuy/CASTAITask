from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
import pandas as pd
from typing import Optional

class FeatureEngineer(TransformerMixin, BaseEstimator):
    """
    A scikit-learn compatible feature engineering class for handling missing values, 
    multicollinearity, scaling, and optional dimensionality reduction using PCA or TSNE.

    Attributes:
    -----------
    use_pca : bool
        Whether to apply PCA for dimensionality reduction and multicollinearity handling.
    use_vif : bool
        Whether to apply VIF for multicollinearity detection (alternative to PCA).
    n_components : int
        Number of components to retain for PCA.
    use_tsne : bool
        Whether to apply TSNE for visualization.
    """
    
    def __init__(self, imputation_strategy: str = 'median', vif_threshold: int = 10, 
                 use_pca: bool = False, n_components: int = 100, use_tsne: bool = False):
        self.imputation_strategy = imputation_strategy
        self.vif_threshold = vif_threshold
        self.use_pca = use_pca
        self.n_components = n_components
        self.use_tsne = use_tsne
        self.scaler = None
        self.drop_features = []
        self.pca = None
        self.tsne = None

    def fit(self, X: pd.DataFrame | np.ndarray, y: Optional[pd.Series] = None) -> 'FeatureEngineer':
        """
        Fit the transformer to the data (handle missing values and multicollinearity).
        """
        # convert to pandas for convenience
        if type(X) == np.ndarray:
            X = pd.DataFrame(X)

        # remember to handle missing values
        numerical_cols = X.select_dtypes(include=['float64', 'float32', 'int64']).columns
        self.num_imputer = SimpleImputer(strategy=self.imputation_strategy)
        X[numerical_cols] = self.num_imputer.fit_transform(X[numerical_cols])

        # scale the data
        self.scaler = StandardScaler()
        self.scaler.fit(X[numerical_cols]) # .drop(columns=self.drop_features, errors='ignore')
        
        # apply PCA for multicollinearity removal and dim reduction, if selected
        if self.use_pca:
            self.pca = PCA(n_components=self.n_components)
            self.pca.fit(X[numerical_cols])
        else:
            # detect multicollinearity using VIF if PCA is not selected
            self.drop_features = self._detect_multicollinearity(X[numerical_cols])
        
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """
        Transform the data (remove multicollinear features, scale the data).
        """
        # convert to pandas for convenience
        if type(X) == np.ndarray:
            X = pd.DataFrame(X)

        # remember to handle missing values
        numerical_cols = X.select_dtypes(include=['float64', 'float32', 'int64']).columns
        X[numerical_cols] = self.num_imputer.transform(X[numerical_cols])

        # scale the data
        X[numerical_cols] = self.scaler.transform(X[numerical_cols])

        # if PCA is used, apply dimensionality reduction
        if self.use_pca:
            X_pca = self.pca.transform(X[numerical_cols])
            X = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
        else:
            # remove features with high multicollinearity (based on VIF)
            X = X.drop(columns=self.drop_features, errors='ignore')

        # optionally apply TSNE for dimensionality reduction visualization or for debug
        if self.use_tsne:
            self.tsne = TSNE(n_components=2)
            X_tsne = self.tsne.fit_transform(X)
            X = pd.DataFrame(X_tsne, columns=['TSNE1', 'TSNE2'])

        return X

    def _detect_multicollinearity(self, X: pd.DataFrame | np.ndarray) -> list:
        """
        Detect multicollinearity using VIF and return features to drop.
        """
        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        
        drop_features = vif_data[vif_data["VIF"] > self.vif_threshold]["feature"].tolist()
        return drop_features

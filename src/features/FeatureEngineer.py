from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
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
    multicollinearity, scaling, optional dimensionality reduction (PCA or TSNE), 
    datetime feature extraction, and numerical transformations (Log, Quantile, BoxCox).
    
    Attributes:
    -----------
    imputation_strategy : str
        The imputation strategy for missing values ('median', 'mean', etc.).
    vif_threshold : int
        Variance Inflation Factor threshold to detect multicollinearity.
    use_pca : bool
        Whether to apply PCA for dimensionality reduction and multicollinearity handling.
    n_components : int
        Number of components to retain for PCA.
    use_tsne : bool
        Whether to apply TSNE for visualization.
    datetime_col : str
        The name of the datetime column to extract features from.
    default_transform : str
        The default transformation to apply to numerical columns ('log', 'quantile', 'boxcox').
    transform_overrides : dict
        A dictionary mapping specific columns to transformations, overriding the default.
    """
    
    def __init__(self, imputation_strategy: str = 'median', vif_threshold: int = 10, 
                 use_pca: bool = False, n_components: int = 100, use_tsne: bool = False,
                 datetime_col: str = None, use_transform: bool = True, default_transform: str = 'log', 
                 transform_overrides: dict = None, preprocessor: ColumnTransformer = None):
        self.imputation_strategy = imputation_strategy
        self.vif_threshold = vif_threshold
        self.use_pca = use_pca
        self.n_components = n_components
        self.use_tsne = use_tsne
        self.datetime_col = datetime_col
        self.use_transform = use_transform
        self.default_transform = default_transform
        self.transform_overrides = transform_overrides or {}
        self.transformers = {}
        self.scaler = None
        self.drop_features = []
        self.pca = None
        self.tsne = None
        self.preprocessor = None or preprocessor
        
    
    def _extract_time_features(self, df: pd.DataFrame):
        """
        Extracts time-based features from the datetime column.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe.
        
        Returns:
        -----------
        df : pd.DataFrame
            Dataframe with new time features.
        """
        if self.datetime_col and self.datetime_col in df.columns:
            df['month'] = df[self.datetime_col].dt.month
            df['day_of_month'] = df[self.datetime_col].dt.day
            df['day_of_week'] = df[self.datetime_col].dt.weekday
            df['hour'] = df[self.datetime_col].dt.hour
            df['minute'] = df[self.datetime_col].dt.minute
            df['second'] = df[self.datetime_col].dt.second
        return df

    
    def _fit_transformation(self, X: pd.DataFrame, col: str, transform: str):
        """
        Apply the selected transformation to a specific column.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input dataframe.
        col : str
            Column name to apply transformation to.
        transform : str
            Type of transformation to apply ('log', 'quantile', 'boxcox').
        
        Returns:
        -----------
        X : pd.DataFrame
            Dataframe with the transformed column.
        """
        print(f"Fitting transform {transform} to {col}")
        if transform == 'log':
            X[col] = np.log1p(X[col]) # np.log1p gives us log(1 + x), to avoid log(0) :O
        elif transform == 'quantile':
            qt = QuantileTransformer(output_distribution='normal', random_state=42)
            qt = qt.fit(X[[col]])
            self.transformers[col] = qt
        elif transform == 'boxcox':
            pt = PowerTransformer(method='box-cox', standardize=True)
            X[col] = pt.fit(X[[col]] + np.finfo(float).eps)
            self.transformers[col] = pt
        return X

    def _apply_transformation(self, X: pd.DataFrame, col: str, transform: str):
        """
        Apply the selected transformation to a specific column.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input dataframe.
        col : str
            Column name to apply transformation to.
        transform : str
            Type of transformation to apply ('log', 'quantile', 'boxcox').
        
        Returns:
        -----------
        X : pd.DataFrame
            Dataframe with the transformed column.
        """
        if transform == 'log':
            X[col] = np.log1p(X[col])  # np.log1p gives us log(1 + x), to avoid log(0) :O
        elif transform == 'quantile':
            qt = self.transformers[col]
            X[col] = qt.fit_transform(X[[col]])
        elif transform == 'boxcox':
            pt = self.transformers[col]
            X[col] = pt.transform(X[[col]] + np.finfo(float).eps)
        return X

    def fit(self, X: pd.DataFrame | np.ndarray, y: Optional[pd.Series] = None) -> 'FeatureEngineer':
        """
        Fit the transformer to the data (handle missing values, scaling, multicollinearity, and PCA).
        
        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Input dataframe or numpy array.
        y : Optional[pd.Series]
            Target values (optional, for compatibility with sklearn).
        
        Returns:
        -----------
        self : FeatureEngineer
            The fitted transformer.
        """
        X = X.copy()

        # convert to pandas if input is a numpy array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        # extract time-based features
        X = self._extract_time_features(X)
        
        # handle missing values
        numerical_cols = X.select_dtypes(include=['float64', 'float32', 'int64']).columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns

        self.num_imputer = SimpleImputer(strategy=self.imputation_strategy)
        X[numerical_cols] = self.num_imputer.fit_transform(X[numerical_cols])

        # scale the data using standard scaling
        self.scaler = StandardScaler()
        self.scaler.fit(X[numerical_cols]) 

        # fit transformations to numerical columns
        if self.use_transform:
            for col in numerical_cols:
                transform = self.transform_overrides.get(col, self.default_transform)
                X = self._fit_transformation(X, col, transform)

        if self.preprocessor is None:
            # for numerical features: leave in-place
            # for categorical features: one-hot enc
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', FunctionTransformer(lambda x: x), numerical_cols),  
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols) 
                ]
            )

        X = self.preprocessor.fit_transform(X)
        X = pd.DataFrame(X)

        # numerical column indices are different after one-hot encoding
        numerical_cols = X.select_dtypes(include=['float64', 'float32', 'int64']).columns

        self.post_onehot_imputer = SimpleImputer(strategy="constant")
        X[numerical_cols] = self.post_onehot_imputer.fit_transform(X[numerical_cols])

        # apply PCA or VIF for multicollinearity removal if selected
        if self.use_pca:
            self.pca = PCA(n_components=self.n_components)
            self.pca.fit(X[numerical_cols])
        else:
            # detect multicollinearity using VIF if PCA is not selected
            self.drop_features = self._detect_multicollinearity(X[numerical_cols])
        
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """
        Transform the data (remove multicollinear features, scale, extract time features, apply transformations).
        
        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Input dataframe or numpy array.
        
        Returns:
        -----------
        X : pd.DataFrame
            Transformed dataframe.
        """
        X = X.copy()

        # convert to pandas if input is a numpy array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        # extract time-based features
        X = self._extract_time_features(X)

        # handle missing values
        numerical_cols = X.select_dtypes(include=['float64', 'float32', 'int64']).columns
        X[numerical_cols] = self.num_imputer.transform(X[numerical_cols])

        # scale the data using standard scaling
        X[numerical_cols] = self.scaler.transform(X[numerical_cols])

        # apply transformations to numerical columns
        if self.use_transform:
            for col in numerical_cols:
                transform = self.transform_overrides.get(col, self.default_transform)
                X = self._apply_transformation(X, col, transform)

        X = self.preprocessor.transform(X)
        X = pd.DataFrame(X)

        # numerical column indices are different after one-hot encoding
        numerical_cols = X.select_dtypes(include=['float64', 'float32', 'int64']).columns

        X[numerical_cols] = self.post_onehot_imputer.transform(X[numerical_cols])

        # apply PCA if selected
        if self.use_pca:
            X_pca = self.pca.transform(X[numerical_cols])
            X = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
        else:
            # remove features with high multicollinearity
            X = X.drop(columns=self.drop_features, errors='ignore')

        # optionally apply TSNE for dimensionality reduction visualization
        if self.use_tsne:
            self.tsne = TSNE(n_components=2)
            X_tsne = self.tsne.fit_transform(X)
            X = pd.DataFrame(X_tsne, columns=['TSNE1', 'TSNE2'])

        return X

    def _detect_multicollinearity(self, X: pd.DataFrame | np.ndarray) -> list:
        """
        Detect multicollinearity using VIF and return features to drop.
        
        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Input dataframe or numpy array.
        
        Returns:
        -----------
        drop_features : list
            List of features to drop due to multicollinearity.
        """
        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        drop_features = vif_data[vif_data["VIF"] > self.vif_threshold]["feature"].tolist()
        return drop_features

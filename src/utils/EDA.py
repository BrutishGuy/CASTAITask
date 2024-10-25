import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

class EDA:
    """
    A class to perform Exploratory Data Analysis on a dataset.

    Attributes:
    -----------
    df : pd.DataFrame
        The dataset to be analyzed.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def descriptive_stats(self):
        """
        Display basic descriptive statistics for numerical and categorical features.
        """
        print("Numerical Features Summary:")
        print(self.df.describe())
        print("\nCategorical Features Summary:")
        print(self.df.describe(include='object'))
    
    def plot_features(self, grid_size=(3, 3), plot_type='hist', n_bins=20):
        """
        Plot the distributions of numerical and categorical features.

        Parameters:
        -----------
        grid_size : tuple
            Grid size for plotting, e.g., (3, 3) will plot 9 features per page.
        plot_type : str
            Type of plot to use for numerical features. Can be 'hist' or 'box'.
        """
        numerical_cols = self.df.select_dtypes(include=['float64', 'float32', 'int64']).columns
        categorical_cols = self.df.select_dtypes(include=['object', 'string', 'category']).columns

        # plot numerical features in batches
        num_plots = len(numerical_cols)
        for i in range(0, num_plots, grid_size[0] * grid_size[1]):
            batch_cols = numerical_cols[i:i + grid_size[0] * grid_size[1]]
            self._plot_numerical_features(batch_cols, grid_size, plot_type, n_bins=n_bins)

        # plot categorical features in batches
        cat_plots = len(categorical_cols)
        for i in range(0, cat_plots, grid_size[0] * grid_size[1]):
            batch_cols = categorical_cols[i:i + grid_size[0] * grid_size[1]]
            self._plot_categorical_features(batch_cols, grid_size)

    def _plot_numerical_features(self, cols, grid_size, plot_type, n_bins=20):
        """
        Helper function to plot numerical features in a grid layout.
        """
        rows, cols_in_grid = grid_size
        fig, axes = plt.subplots(rows, cols_in_grid, figsize=(5 * cols_in_grid, 4 * rows))
        axes = axes.flatten()
        
        for idx, col in enumerate(cols):
            ax = axes[idx]
            if plot_type == 'hist':
                ax.hist(self.df[col].dropna(), bins=n_bins, color='blue', alpha=0.7)
                ax.set_title(f'Distribution of {col}')
            elif plot_type == 'box':
                ax.boxplot(self.df[col].dropna(), vert=False)
                ax.set_title(f'Boxplot of {col}')
            ax.set_xlabel(col)
        
        # remove any unused subplots
        for j in range(len(cols), len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.show()

    def _plot_categorical_features(self, cols, grid_size):
        """
        Helper function to plot categorical features in a grid layout.
        """
        rows, cols_in_grid = grid_size
        fig, axes = plt.subplots(rows, cols_in_grid, figsize=(5 * cols_in_grid, 4 * rows))
        axes = axes.flatten()

        for idx, col in enumerate(cols):
            ax = axes[idx]
            sns.countplot(y=self.df[col], ax=ax, palette='viridis', order=self.df[col].value_counts().index[:10])
            ax.set_title(f'Top Categories in {col}')
            ax.set_xlabel('Count')
        
        # remove any unused subplots
        for j in range(len(cols), len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()
    
    def missing_values(self):
        """
        Display missing values in the dataset.
        """
        missing_values = self.df.isnull().sum()
        print("Missing Values:\n", missing_values)
        
    def plot_missing_values_heatmap(self):
        """
        Visualise missing values in the dataset.
        """
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.df.isnull(), cbar=False, cmap='viridis')
        plt.title("Missing Values Heatmap")
        plt.show()
    
    def target_distribution(self):
        """
        Plot the distribution of the target variable.
        """
        plt.figure(figsize=(6, 4))
        sns.countplot(x='label', data=self.df)
        plt.title("Target Variable Distribution")
        plt.show()
    
    def correlation_matrix(self, corr_figsize=(24, 16), annot=False):
        """
        Plot the correlation matrix for numerical features.
        """
        numerical_data = self.df.select_dtypes(include=['float64', 'float32', 'int64'])
        fig, ax = plt.subplots(figsize=corr_figsize)
        corr_matrix = numerical_data.corr()
        sns.heatmap(corr_matrix, annot=annot, cmap='coolwarm', fmt='.2f', ax=ax)
        plt.title("Correlation Matrix")
        plt.show()
    
    def target_correlations(self, corr_figsize=(24, 16), annot=False):
        """
        Plot the correlations of dependent variables against target variable, for numerical features.
        """
        # select only numerical features, including the newly created binary label
        numerical_data = self.df.select_dtypes(include=['float64', 'float32', 'int64']).copy()
        # binarize the target variable ('label') in the original dataframe
        numerical_data['label'] = self.df['label'].map({'Interrupted': 1, 'Continue': 0})

        # extract the target variable
        target = numerical_data['label']
        features = numerical_data.drop(columns=['label'])

        # calculate correlations using Pearson, Spearman individually
        corr_pearson = features.apply(lambda x: x.corr(target, method='pearson')).sort_values(ascending=False)
        corr_spearman = features.apply(lambda x: x.corr(target, method='spearman')).sort_values(ascending=False)

        # plotting the results
        fig, ax = plt.subplots(1, 2, figsize=corr_figsize)
        
        ax[0].set_title('Pearson method')
        sns.heatmap(corr_pearson.to_frame(), ax=ax[0], annot=annot)
        
        ax[1].set_title('Spearman method')
        sns.heatmap(corr_spearman.to_frame(), ax=ax[1], annot=annot)
        
        #ax[2].set_title('Kendall method')
        #sns.heatmap(corr_kendall.to_frame(), ax=ax[2], annot=annot)

        plt.show()

    def handle_missing_values(self, X, strategy):
        """
        Handle missing values in the dataset based on the specified strategy.

        Parameters:
        -----------
        X : pd.DataFrame
            The input data.
        strategy : str
            Strategy to handle missing values: "drop", "median", "mean", "zerofill".

        Returns:
        --------
        np.ndarray
            Data with missing values handled.
        """
        if strategy == 'drop':
            return X.dropna()
        elif strategy == 'median':
            imputer = SimpleImputer(strategy='median')
        elif strategy == 'mean':
            imputer = SimpleImputer(strategy='mean')
        elif strategy == 'zerofill':
            imputer = SimpleImputer(strategy='constant', fill_value=0)
        else:
            raise ValueError("Invalid strategy. Choose from 'drop', 'median', 'mean', or 'zerofill'.")

        return imputer.fit_transform(X)

    def pca_analysis(self, n_components=5, missing_value_strategy='drop', plot_loadings=True, n_loadings_components=5, loads_figsize=(15, 10)):
        """
        Perform PCA and visualize the first two components, with missing value handling.
        Also plot explained variance and display loadings.

        Parameters:
        -----------
        n_components : int
            Number of PCA components to retain.
        missing_value_strategy : str
            Strategy to handle missing values: "drop", "median", "mean", "zerofill".
        plot_loadings : bool
            Whether to plot loadings of the first few components.
        """
        numerical_data = self.df.select_dtypes(include=['float64', 'float32', 'int64'])
        scaled_data = StandardScaler().fit_transform(numerical_data)

        # remember to handle missing values
        cleaned_data = self.handle_missing_values(scaled_data, missing_value_strategy)

        # PCA for dimensionality reduction and multicollinearity removal
        pca = PCA(n_components=n_components)
        pca_components = pca.fit_transform(cleaned_data)
        
        # explained variance extraction
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)

        # plot explained variance
        plt.figure(figsize=(8, 5))
        plt.plot(np.arange(1, n_components + 1), explained_variance, marker='o', label='Individual Explained Variance')
        plt.plot(np.arange(1, n_components + 1), cumulative_variance, marker='o', label='Cumulative Explained Variance')
        plt.axhline(y=0.90, color='r', linestyle='--', label='90% Variance Threshold')
        plt.title("Explained Variance per PCA Component")
        plt.xlabel("Principal Component")
        plt.ylabel("Variance Explained")
        plt.legend()
        plt.show()

        # plot the first two PCA components
        plt.figure(figsize=(8, 6))
        plt.scatter(pca_components[:, 0], pca_components[:, 1], c=self.df['label'].apply(lambda x: 0 if x == 'Interrupted' else 1), cmap='viridis')
        plt.title(f"PCA: First 2 Components")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.show()

        if plot_loadings:
            # plot the loadings for the PCA components

            if n_loadings_components > n_components:
                raise ValueError(f"Number of loadings components to plot must be less than or equal to the number of components, n_components = {n_components}.")
            loadings = pd.DataFrame(pca.components_[0:n_loadings_components, :], columns=numerical_data.columns)
            maxPC = 1.01 * np.max(np.abs(loadings.values))
            
            f, axes = plt.subplots(n_loadings_components, 1, figsize=loads_figsize, sharex=True)
            for i, ax in enumerate(axes):
                pc_loadings = loadings.loc[i, :]
                colors = ['C0' if l > 0 else 'C1' for l in pc_loadings]
                ax.axhline(color='#888888')
                pc_loadings.plot.bar(ax=ax, color=colors)
                ax.set_ylabel(f'PC{i+1}')
                ax.set_ylim(-maxPC, maxPC)
            plt.tight_layout()
            plt.show()

        return loadings

    def tsne_analysis(self, missing_value_strategy='drop'):
        """
        Perform TSNE for dimensionality reduction and visualization, with missing value handling.

        Parameters:
        -----------
        missing_value_strategy : str
            Strategy to handle missing values: "drop", "median", "mean", "zerofill".
        """
        numerical_data = self.df.select_dtypes(include=['float64', 'float32', 'int64'])
        scaled_data = StandardScaler().fit_transform(numerical_data)

        # remember to handle missing values
        cleaned_data = self.handle_missing_values(scaled_data, missing_value_strategy)
        
        tsne = TSNE(n_components=2, random_state=42)
        tsne_components = tsne.fit_transform(cleaned_data)
        
        plt.figure(figsize=(8, 6))
        plt.scatter(tsne_components[:, 0], tsne_components[:, 1], c=self.df['label'].apply(lambda x: 0 if x == 'Interrupted' else 1), cmap='viridis')
        plt.title("TSNE: 2D Components")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.show()

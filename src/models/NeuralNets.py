from sklearn.base import BaseEstimator, ClassifierMixin
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import math 

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()  # standard for binary classification
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out
    
class AdvancedNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob=0.3, 
                 batch_norm=True, activation_function='relu'):
        super(AdvancedNeuralNetwork, self).__init__()

        # map activation function name to activation functions
        activation_functions = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'tanh': nn.Tanh()
        }
        self.activation = activation_functions[activation_function]

        self.batch_norm = batch_norm 
        
        self.fc1 = nn.Linear(input_size, hidden_size * 3)
        self.bn1 = nn.BatchNorm1d(hidden_size * 3) if batch_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout_prob)
        
        self.fc2 = nn.Linear(hidden_size * 3, hidden_size * 2)
        self.bn2 = nn.BatchNorm1d(hidden_size * 2) if batch_norm else nn.Identity()
        self.dropout2 = nn.Dropout(dropout_prob)
        
        self.fc3 = nn.Linear(hidden_size * 2, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size) if batch_norm else nn.Identity()
        self.dropout3 = nn.Dropout(dropout_prob)
        
        self.fc4 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.dropout2(out)
        
        out = self.fc3(out)
        out = self.bn3(out)
        out = self.activation(out)
        out = self.dropout3(out)
        
        out = self.fc4(out)  # no sigmoid function here - BCEWithLogitsLoss handles it
        return out

class TabTransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, n_heads, n_layers, output_size, dropout_prob=0.3):
        super(TabTransformerModel, self).__init__()
        
        # Linear layer to project input features into hidden size
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # Transformer encoder with multi-head self-attention
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=n_heads, dropout=dropout_prob)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Final linear layer for classification
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Project input to hidden size
        x = self.input_proj(x)  # Shape: [batch_size, input_size] -> [batch_size, hidden_size]
        
        # Reshape to [sequence_length, batch_size, hidden_size] expected by the Transformer
        x = x.unsqueeze(0)  # Adds sequence dimension: [1, batch_size, hidden_size]
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x)
        
        # Pool the transformer output to get a single vector per example (use mean pooling)
        x = x.mean(dim=0)  # Shape: [batch_size, hidden_size]
        
        # Classification layer
        x = self.fc(x)  # Shape: [batch_size, output_size]
        
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, dropout_prob, max_len=15000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout_prob)

        # Create positional encoding for the maximum length
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2) * -(math.log(10000.0) / hidden_size))
        pe = torch.zeros(max_len, hidden_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape becomes [1, max_len, hidden_size]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Remove any extra dimensions and apply positional encoding correctly
        # Ensuring x shape is [batch_size, seq_len, hidden_size]
        batch_size, seq_len, hidden_size = x.size()
        
        # Apply positional encoding by trimming to the input sequence length
        x = x + self.pe[:, :seq_len, :].expand(batch_size, seq_len, hidden_size)
        return self.dropout(x)

class NeuralNetworkClassifier(BaseEstimator, ClassifierMixin):
    """
    A PyTorch-based simple neural network classifier using the scikit-learn classifier mixin interface.
    
    Methods:
    ----------
    fit(X: np.ndarray, y: np.ndarray) -> 'NeuralNetworkClassifier':
        Trains the neural network on the input data.
    
    predict(X: np.ndarray) -> np.ndarray:
        Predicts the class for the input data.
    """

    def __init__(self, input_size: int, hidden_size: int = 64, output_size: int = 1,
                 dropout_prob: float = 0.3, learning_rate: float = 0.001, 
                 optimizer_type: str = 'Adam', use_scheduler: bool = False,
                 batch_norm: bool = False, activation_function: str = 'relu',
                 weight_decay: float = 1e-4, patience: int = 10, epochs: int = 500, 
                 device: torch.device = None, verbose: bool = True):
        """
        Initializes the neural network classifier.

        Parameters:
        ----------
        input_size : int
            The size of the input layer (number of features).
        hidden_size : int
            The number of hidden units in the network.
        output_size : int
            The number of output units (1 for binary classification).
        dropout_prob : float
            The dropout probability for regularization.
        learning_rate : float
            The learning rate for the optimizer.
        optimizer_type : str
            Type of optimizer to use ('Adam' or 'AdamW').
        use_scheduler : bool
            Whether to use a learning rate scheduler.
        weight_decay : float
            The weight decay (L2 regularization) factor.
        patience : int
            The number of epochs to wait for validation loss improvement before applying early stopping.
        epochs : int
            The number of training epochs.
        device : torch.device
            The device to use for training (GPU or CPU).
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_prob = dropout_prob
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.use_scheduler = use_scheduler
        self.weight_decay = weight_decay
        self.patience = patience
        self.batch_norm = batch_norm
        self.activation_function = activation_function
        self.epochs = epochs
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose = verbose

        self.classes_ = None  # stores classes after fit
        self.n_features_in_ = None  # features in input data
        
        self._build_model()

    def _build_model(self):
        """
        Builds the PyTorch neural network model.
        """
        self.model = NeuralNetwork(
            input_size=self.input_size, 
            hidden_size=self.hidden_size, 
            output_size=self.output_size
       ).to(self.device)

        self.criterion = nn.BCELoss()

        if self.optimizer_type == 'AdamW':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # LR scheduler
        if self.use_scheduler:
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=self.patience, verbose=True)

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> 'NeuralNetworkClassifier':
        """
        Trains the neural network on the input data.
        
        Parameters:
        ----------
        X : np.ndarray
            The input data (features) for training.
        y : np.ndarray
            The target labels for training.
        
        Returns:
        ----------
        NeuralNetworkClassifier
            The fitted classifier (self).
        """
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(self.device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(self.device)

        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        for epoch in range(self.epochs):
            self.model.train() # set back to training
            self.optimizer.zero_grad()

            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            loss.backward()
            self.optimizer.step()
            
            # validation step for learning rate scheduling
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = self.criterion(val_outputs, y_val_tensor)
            
            # udpate the learning rate scheduler based on validation loss
            if self.use_scheduler:
                self.scheduler.step(val_loss)
            
            if self.verbose:
                print(f'Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the class for the input data.
        
        Parameters:
        ----------
        X : np.ndarray
            The input data (features) to predict.
        
        Returns:
        ----------
        np.ndarray
            The predicted class labels.
        """
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = torch.sigmoid(outputs).cpu().numpy()
            return (predictions > 0.5).astype(int).flatten() 

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the class for the input data.
        
        Parameters:
        ----------
        X : np.ndarray
            The input data (features) to predict.
        
        Returns:
        ----------
        np.ndarray
            The predicted class labels.
        """
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = torch.sigmoid(outputs).cpu().numpy()
            return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the class for the input data.
        
        Parameters:
        ----------
        X : np.ndarray
            The input data (features) to predict.
        
        Returns:
        ----------
        np.ndarray
            The predicted class labels.
        """
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.sigmoid(outputs).cpu().numpy()
            return np.hstack([1 - probabilities, probabilities])  # return in (1-p, p) format for sklearn

class AdvancedNeuralNetworkClassifier(NeuralNetworkClassifier):
    """
    An advanced neural network classifier with additional regularization and complexity.
    
    Inherits from NeuralNetworkClassifier but uses a more complex model architecture with Batch Normalization
    and configurable activation functions.
    """
    
    def _build_model(self):
        """
        Builds the advanced PyTorch neural network model.
        """
        self.model = AdvancedNeuralNetwork(
            input_size=self.input_size, 
            hidden_size=self.hidden_size, 
            output_size=self.output_size, 
            dropout_prob=self.dropout_prob,
            batch_norm=self.batch_norm, 
            activation_function=self.activation_function 
        ).to(self.device)

        self.criterion = nn.BCEWithLogitsLoss()

        # choose optimizer
        if self.optimizer_type == 'AdamW':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # use an LR scheduler if enabled
        if self.use_scheduler:
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=self.patience, verbose=True)

class TransformerClassifier(BaseEstimator, ClassifierMixin):
    """
    A PyTorch-based transformer classifier using the scikit-learn classifier mixin interface.

    Attributes:
    -----------
    classes_ : np.ndarray
        Unique classes in the target variable.
    n_features_in_ : int
        Number of features in the input data.
    """
    def __init__(self, input_size: int, hidden_size: int = 128, n_heads: int = 4, n_layers: int = 2, 
                 output_size: int = 1, dropout_prob: float = 0.3, learning_rate: float = 0.001,
                 optimizer_type: str = 'Adam', use_scheduler: bool = True, weight_decay: float = 1e-4, 
                 patience: int = 5, epochs: int = 500, device: torch.device = None, verbose: bool = False):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.output_size = output_size
        self.dropout_prob = dropout_prob
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.use_scheduler = use_scheduler
        self.weight_decay = weight_decay
        self.patience = patience
        self.epochs = epochs
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose = verbose
        self.classes_ = None
        self.n_features_in_ = None

        self._build_model()

    def _build_model(self):
        self.model = TabTransformerModel(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            output_size=self.output_size,
            dropout_prob=self.dropout_prob
        ).to(self.device)

        self.criterion = nn.BCEWithLogitsLoss()

        if self.optimizer_type == 'AdamW':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        if self.use_scheduler:
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=self.patience, verbose=True)
    
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        """
        Trains the neural network on the input data.
        
        Parameters:
        ----------
        X : np.ndarray
            The input data (features) for training.
        y : np.ndarray
            The target labels for training.
        
        Returns:
        ----------
        NeuralNetworkClassifier
            The fitted classifier (self).
        """
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        # Convert training data into DataLoader
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Adjust batch_size as necessary
        validation_dataset = TensorDataset(X_tensor, y_)
        validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False) 

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.model(X_batch)  # Pass batch through model
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            # validation step for learning rate scheduling
            val_loss = 0            
            self.model.eval()
            for X_val_batch, y_val_batch in validation_loader:
                X_val_batch, y_val_batch = X_val_batch.to(self.device), y_val_batch.to(self.device)
                
                with torch.no_grad():
                    val_outputs = self.model(X_val_batch)  # Pass batch through model
                    val_loss += self.criterion(val_outputs, y_val_batch)

            # udpate the learning rate scheduler based on validation loss
            if self.use_scheduler:
                self.scheduler.step(val_loss)
            
            if self.verbose:
                print(f'Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
                
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print("Early stopping triggered.")
                    break

        return self

    def predict(self, X: np.ndarray):
        """
        Predicts targets for an input data set X using the trained model
        
        Parameters:
        ----------
        X : np.ndarray
            The input data (features) for prediction.

        Returns:
        ----------
        np.ndarray
            The array/list of predictions
        """
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = torch.sigmoid(outputs).cpu().numpy()
            return (predictions > 0.5).astype(int).flatten()

    def predict_proba(self, X: np.ndarray):
        """
        Predicts targets for an input data set X using the trained model
        
        Parameters:
        ----------
        X : np.ndarray
            The input data (features) for prediction.

        Returns:
        ----------
        np.ndarray
            The array/list of predictions
        """
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.sigmoid(outputs).cpu().numpy()
            return np.hstack([1 - probabilities, probabilities])
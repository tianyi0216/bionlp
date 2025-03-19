import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from typing import Dict, List, Tuple, Callable, Union, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


class EarlyStopping:
    """Early stopping utility to stop training when validation metric doesn't improve.
    
    Parameters
    ----------
    patience : int
        Number of epochs to wait after last improvement
    delta : float
        Minimum change to qualify as improvement
    mode : str
        'min' for metrics to minimize (loss), 'max' for metrics to maximize (accuracy)
    verbose : bool
        Whether to print messages
    """
    
    def __init__(self, patience: int = 10, delta: float = 0, mode: str = 'min', verbose: bool = True):
        """Initialize early stopping."""
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_min = np.Inf if mode == 'min' else -np.Inf
        self.val_score_max = -np.Inf if mode == 'max' else np.Inf
    
    def __call__(self, val_score, model):
        """Check if training should be stopped.
        
        Parameters
        ----------
        val_score : float
            Current validation score
        model : torch.nn.Module
            Model to save when validation improves
            
        Returns
        -------
        bool
            Whether to stop training
        """
        score = -val_score if self.mode == 'min' else val_score
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_score, model)
            self.counter = 0
        
        return self.early_stop
    
    def save_checkpoint(self, val_score, model):
        """Save model when validation score improves."""
        if self.verbose:
            improvement = 'decreased' if self.mode == 'min' else 'increased'
            print(f'Validation score {improvement} ({self.val_score_min:.6f} --> {val_score:.6f}). Saving model.')
        
        self.model_state = model.state_dict()
        if self.mode == 'min':
            self.val_score_min = val_score
        else:
            self.val_score_max = val_score
    
    def load_best_model(self, model):
        """Load the best model state."""
        model.load_state_dict(self.model_state)
        return model


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: Callable,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 100,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    early_stopping: Optional[EarlyStopping] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    verbose: bool = True
) -> Dict:
    """Train a PyTorch model.
    
    Parameters
    ----------
    model : nn.Module
        PyTorch model to train
    train_loader : DataLoader
        DataLoader for training data
    val_loader : DataLoader
        DataLoader for validation data
    criterion : Callable
        Loss function
    optimizer : Optimizer
        Optimizer for training
    num_epochs : int
        Maximum number of epochs
    device : str
        Device for training ('cuda' or 'cpu')
    early_stopping : EarlyStopping, optional
        Early stopping utility
    scheduler : lr_scheduler, optional
        Learning rate scheduler
    verbose : bool
        Whether to print progress
        
    Returns
    -------
    Dict
        Training history
    """
    # Move model to device
    model = model.to(device)
    
    # Initialize tracking variables
    best_val_loss = float('inf')
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_metrics': {}
    }
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]") if verbose else train_loader
        
        for batch in train_iterator:
            # Move batch to device
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(features)
            
            # Reshape outputs if needed
            if outputs.shape != labels.shape:
                if outputs.shape[0] == labels.shape[0]:
                    if len(labels.shape) == 1:
                        # Binary classification where labels are not one-hot
                        labels = labels.unsqueeze(1)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update tracking
            train_loss += loss.item()
            train_steps += 1
            
            if verbose and isinstance(train_iterator, tqdm):
                train_iterator.set_postfix({'loss': loss.item()})
        
        # Calculate average training loss
        avg_train_loss = train_loss / train_steps
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_steps = 0
        val_preds = []
        val_labels = []
        
        val_iterator = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]") if verbose else val_loader
        
        with torch.no_grad():
            for batch in val_iterator:
                # Move batch to device
                features = batch['features'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass
                outputs = model(features)
                
                # Reshape outputs if needed
                if outputs.shape != labels.shape:
                    if outputs.shape[0] == labels.shape[0]:
                        if len(labels.shape) == 1:
                            # Binary classification where labels are not one-hot
                            labels = labels.unsqueeze(1)
                
                # Calculate loss
                loss = criterion(outputs, labels)
                
                # Update tracking
                val_loss += loss.item()
                val_steps += 1
                
                # Store predictions and labels for metrics
                val_preds.extend(outputs.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                
                if verbose and isinstance(val_iterator, tqdm):
                    val_iterator.set_postfix({'loss': loss.item()})
        
        # Calculate average validation loss
        avg_val_loss = val_loss / val_steps
        history['val_loss'].append(avg_val_loss)
        
        # Calculate validation metrics
        # Assume binary classification for now
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        
        # For binary classification, threshold predictions
        if val_preds.shape[1] == 1:
            val_preds_class = (val_preds > 0.5).astype(int)
            try:
                val_auc = roc_auc_score(val_labels, val_preds)
                val_accuracy = accuracy_score(val_labels, val_preds_class)
                val_precision = precision_score(val_labels, val_preds_class, zero_division=0)
                val_recall = recall_score(val_labels, val_preds_class, zero_division=0)
                val_f1 = f1_score(val_labels, val_preds_class, zero_division=0)
                
                # Save metrics to history
                history['val_metrics'][epoch] = {
                    'auc': val_auc,
                    'accuracy': val_accuracy,
                    'precision': val_precision,
                    'recall': val_recall,
                    'f1': val_f1
                }
                
                if verbose:
                    print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val AUC: {val_auc:.4f} - Val Acc: {val_accuracy:.4f}')
            except:
                if verbose:
                    print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}')
        else:
            if verbose:
                print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}')
        
        # Learning rate scheduler step
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()
        
        # Early stopping check
        if early_stopping is not None:
            if early_stopping(avg_val_loss, model):
                if verbose:
                    print(f'Early stopping triggered after epoch {epoch+1}')
                model = early_stopping.load_best_model(model)
                break
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
    
    # Load best model
    if early_stopping is None:
        model.load_state_dict(best_model_state)
    
    return {'model': model, 'history': history}


def evaluate_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    criterion: Optional[Callable] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    task_type: str = 'classification'  # 'classification' or 'regression'
) -> Dict:
    """Evaluate a PyTorch model on test data.
    
    Parameters
    ----------
    model : nn.Module
        PyTorch model to evaluate
    test_loader : DataLoader
        DataLoader for test data
    criterion : Callable, optional
        Loss function
    device : str
        Device for evaluation ('cuda' or 'cpu')
    task_type : str
        Type of task ('classification' or 'regression')
        
    Returns
    -------
    Dict
        Evaluation metrics
    """
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Initialize tracking variables
    test_loss = 0.0
    test_steps = 0
    all_preds = []
    all_labels = []
    all_trial_ids = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Move batch to device
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            
            # Get trial IDs if available
            if 'trial_ids' in batch:
                all_trial_ids.extend(batch['trial_ids'])
            
            # Forward pass
            outputs = model(features)
            
            # Reshape outputs if needed
            if labels.dim() == 1:
                labels = labels.unsqueeze(1)
            
            # Calculate loss if criterion provided
            if criterion is not None:
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                test_steps += 1
            
            # Store predictions and labels
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics based on task type
    results = {}
    
    if criterion is not None:
        results['loss'] = test_loss / test_steps
    
    if task_type == 'classification':
        # Binary classification metrics
        if all_preds.shape[1] == 1:
            all_preds_class = (all_preds > 0.5).astype(int)
            
            try:
                # Classification metrics
                results['auc'] = roc_auc_score(all_labels, all_preds)
                results['accuracy'] = accuracy_score(all_labels, all_preds_class)
                results['precision'] = precision_score(all_labels, all_preds_class, zero_division=0)
                results['recall'] = recall_score(all_labels, all_preds_class, zero_division=0)
                results['f1'] = f1_score(all_labels, all_preds_class, zero_division=0)
                
                # Confusion matrix
                cm = confusion_matrix(all_labels, all_preds_class)
                results['confusion_matrix'] = cm
                
                # Classification report
                report = classification_report(all_labels, all_preds_class, output_dict=True)
                results['classification_report'] = report
            except:
                print("Warning: Could not calculate all classification metrics.")
    
    elif task_type == 'regression':
        # Regression metrics
        results['mse'] = mean_squared_error(all_labels, all_preds)
        results['rmse'] = np.sqrt(results['mse'])
        results['mae'] = mean_absolute_error(all_labels, all_preds)
        
        # R-squared only if not constant
        if np.var(all_labels) > 0:
            results['r2'] = r2_score(all_labels, all_preds)
    
    # Store the raw predictions and labels
    results['predictions'] = all_preds
    results['labels'] = all_labels
    
    # Create a results DataFrame if trial_ids are available
    if all_trial_ids:
        results_df = pd.DataFrame({
            'trial_id': all_trial_ids,
            'true_label': all_labels.flatten(),
            'prediction': all_preds.flatten()
        })
        
        if task_type == 'classification':
            results_df['predicted_class'] = (results_df['prediction'] > 0.5).astype(int)
        
        results['results_df'] = results_df
    
    return results


def plot_training_history(history: Dict, figsize: Tuple[int, int] = (12, 8)):
    """Plot training history.
    
    Parameters
    ----------
    history : Dict
        Training history from train_model
    figsize : Tuple[int, int]
        Figure size
    """
    plt.figure(figsize=figsize)
    
    # Plot training and validation loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Get validation metrics if available
    if history['val_metrics']:
        epochs = list(history['val_metrics'].keys())
        
        # Plot accuracy if available
        if 'accuracy' in history['val_metrics'][epochs[0]]:
            plt.subplot(2, 2, 2)
            plt.plot(epochs, [history['val_metrics'][e]['accuracy'] for e in epochs])
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Validation Accuracy')
            plt.grid(True)
        
        # Plot AUC if available
        if 'auc' in history['val_metrics'][epochs[0]]:
            plt.subplot(2, 2, 3)
            plt.plot(epochs, [history['val_metrics'][e]['auc'] for e in epochs])
            plt.xlabel('Epoch')
            plt.ylabel('AUC')
            plt.title('Validation AUC')
            plt.grid(True)
        
        # Plot F1 if available
        if 'f1' in history['val_metrics'][epochs[0]]:
            plt.subplot(2, 2, 4)
            plt.plot(epochs, [history['val_metrics'][e]['f1'] for e in epochs])
            plt.xlabel('Epoch')
            plt.ylabel('F1 Score')
            plt.title('Validation F1 Score')
            plt.grid(True)
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm: np.ndarray, figsize: Tuple[int, int] = (8, 6)):
    """Plot confusion matrix.
    
    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix
    figsize : Tuple[int, int]
        Figure size
    """
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


def plot_roc_curve(labels: np.ndarray, predictions: np.ndarray, figsize: Tuple[int, int] = (8, 6)):
    """Plot ROC curve.
    
    Parameters
    ----------
    labels : np.ndarray
        True labels
    predictions : np.ndarray
        Predicted probabilities
    figsize : Tuple[int, int]
        Figure size
    """
    from sklearn.metrics import roc_curve, auc
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(labels, predictions)
    roc_auc = auc(fpr, tpr)
    
    # Plot
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


def save_model(model: nn.Module, path: str, metadata: Dict = None):
    """Save PyTorch model with metadata.
    
    Parameters
    ----------
    model : nn.Module
        PyTorch model to save
    path : str
        Path to save the model
    metadata : Dict, optional
        Additional metadata to save with the model
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Prepare save dictionary
    save_dict = {
        'model_state_dict': model.state_dict(),
    }
    
    # Add metadata if provided
    if metadata is not None:
        save_dict['metadata'] = metadata
    
    # Save the model
    torch.save(save_dict, path)
    print(f"Model saved to {path}")


def load_model(model: nn.Module, path: str) -> Tuple[nn.Module, Dict]:
    """Load PyTorch model with metadata.
    
    Parameters
    ----------
    model : nn.Module
        PyTorch model structure to load weights into
    path : str
        Path to the saved model
        
    Returns
    -------
    Tuple[nn.Module, Dict]
        Loaded model and metadata
    """
    # Load the save dictionary
    checkpoint = torch.load(path)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get metadata if available
    metadata = checkpoint.get('metadata', {})
    
    return model, metadata


# Prediction specific utilities
def predict(
    model: nn.Module,
    features: torch.Tensor,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """Make predictions with a trained model.
    
    Parameters
    ----------
    model : nn.Module
        Trained PyTorch model
    features : torch.Tensor
        Input features
    device : str
        Device for prediction ('cuda' or 'cpu')
    threshold : float
        Classification threshold for binary classification
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Raw predictions and class predictions (for classification)
    """
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Move features to device
    features = features.to(device)
    
    # Make prediction
    with torch.no_grad():
        predictions = model(features)
    
    # Convert to numpy
    predictions_np = predictions.cpu().numpy()
    
    # For binary classification, also return thresholded values
    class_predictions = (predictions_np > threshold).astype(int)
    
    return predictions_np, class_predictions


# Function to predict outcomes for new trials
def predict_trial_outcomes(
    model: nn.Module,
    trial_data: pd.DataFrame,
    processor: Union['TrialOutcomeProcessor', Dict],
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    threshold: float = 0.5
) -> pd.DataFrame:
    """Predict outcomes for new trials.
    
    Parameters
    ----------
    model : nn.Module
        Trained PyTorch model
    trial_data : pd.DataFrame
        DataFrame with new trial data
    processor : Union[TrialOutcomeProcessor, Dict]
        Trial processor or preprocessing objects dictionary
    device : str
        Device for prediction ('cuda' or 'cpu')
    threshold : float
        Classification threshold for binary classification
        
    Returns
    -------
    pd.DataFrame
        DataFrame with predictions
    """
    # Preprocess the trial data
    if hasattr(processor, 'preprocess_for_model'):
        # If processor is a TrialOutcomeProcessor instance
        processed_data = processor.preprocess_for_model(trial_data)
        features_df = processed_data['features']
    else:
        # If processor is a dictionary of preprocessing objects
        # Handle text features
        if 'text_cols' in processor:
            for col in processor['text_cols']:
                if col in trial_data.columns:
                    trial_data['combined_text'] = trial_data[processor['text_cols']].apply(
                        lambda row: ' '.join([str(text) for text in row if text not in [None, 'none', '']]), 
                        axis=1
                    )
                    trial_data['text_length'] = trial_data['combined_text'].str.len()
                    trial_data['word_count'] = trial_data['combined_text'].str.split().str.len()
        
        # Handle categorical features
        cat_features = []
        for col in processor.get('categorical_processor', {}):
            if col in trial_data.columns:
                encoder = processor['categorical_processor'][col]
                
                # Check if sparse_output is available in this encoder 
                # (This handles the case where the processor was saved with an older scikit-learn version)
                try:
                    # For newer scikit-learn versions
                    encoded = encoder.transform(trial_data[[col]])
                except ValueError as e:
                    if "unexpected keyword argument 'sparse'" in str(e):
                        # Try to adapt to the encoder's API
                        from sklearn import __version__ as sklearn_version
                        if int(sklearn_version.split('.')[0]) >= 1:
                            # For sklearn >= 1.0, use sparse_output if sparse is not available
                            encoded = encoder.transform(trial_data[[col]])
                        else:
                            # For older sklearn versions
                            encoded = encoder.transform(trial_data[[col]]).toarray()
                    else:
                        raise e
                
                feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0]]
                encoded_df = pd.DataFrame(encoded, columns=feature_names, index=trial_data.index)
                cat_features.append(encoded_df)
        
        # Handle numeric features
        num_features = []
        for col in processor.get('numeric_processor', {}):
            if col in trial_data.columns:
                scaler = processor['numeric_processor'][col]
                scaled = scaler.transform(trial_data[[col]])
                scaled_df = pd.DataFrame(scaled, columns=[col], index=trial_data.index)
                num_features.append(scaled_df)
        
        # Combine features
        if cat_features and num_features:
            features_df = pd.concat(cat_features + num_features, axis=1)
        elif cat_features:
            features_df = pd.concat(cat_features, axis=1)
        elif num_features:
            features_df = pd.concat(num_features, axis=1)
        else:
            raise ValueError("No features could be processed for prediction")
    
    # Convert to tensor
    features = torch.FloatTensor(features_df.values)
    
    # Get predictions
    raw_predictions, class_predictions = predict(model, features, device, threshold)
    
    # Create results DataFrame
    results_df = trial_data.copy()
    results_df['prediction_probability'] = raw_predictions.flatten()
    results_df['prediction'] = class_predictions.flatten()
    
    return results_df


# Example usage
if __name__ == "__main__":
    # Example model
    class SimpleModel(nn.Module):
        def __init__(self, input_dim=10):
            super().__init__()
            self.fc = nn.Linear(input_dim, 1)
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x):
            return self.sigmoid(self.fc(x))
    
    # Create dummy data
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100, 1)).float()
    
    # Split data
    train_X, val_X = X[:80], X[80:]
    train_y, val_y = y[:80], y[80:]
    
    # Create dummy dataloaders
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, features, labels):
            self.features = features
            self.labels = labels
        
        def __len__(self):
            return len(self.features)
        
        def __getitem__(self, idx):
            return {
                'features': self.features[idx],
                'labels': self.labels[idx],
                'trial_id': f"NCT{idx:05d}"
            }
    
    train_dataset = DummyDataset(train_X, train_y)
    val_dataset = DummyDataset(val_X, val_y)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16)
    
    # Create model and training components
    model = SimpleModel()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Train the model
    print("Training model...")
    early_stopping = EarlyStopping(patience=5, mode='min')
    results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=10,
        early_stopping=early_stopping,
        verbose=True
    )
    
    # Evaluate the model
    print("\nEvaluating model...")
    eval_results = evaluate_model(
        model=results['model'],
        test_loader=val_loader,
        criterion=criterion,
        task_type='classification'
    )
    
    print("\nEvaluation metrics:")
    for metric, value in eval_results.items():
        if metric not in ['confusion_matrix', 'classification_report', 'predictions', 'labels', 'results_df']:
            print(f"{metric}: {value}")
    
    # Plot results
    print("\nPlotting training history...")
    plot_training_history(results['history'])
    
    # Plot confusion matrix
    print("\nPlotting confusion matrix...")
    if 'confusion_matrix' in eval_results:
        plot_confusion_matrix(eval_results['confusion_matrix'])
    
    # Plot ROC curve
    print("\nPlotting ROC curve...")
    if 'predictions' in eval_results and 'labels' in eval_results:
        plot_roc_curve(eval_results['labels'], eval_results['predictions'])
    
    # Save and load model
    print("\nSaving and loading model...")
    save_model(model, 'model.pt', {'input_dim': 10})
    loaded_model, metadata = load_model(SimpleModel(), 'model.pt')
    print(f"Loaded model metadata: {metadata}")
    
    # Make predictions
    print("\nMaking predictions...")
    new_features = torch.randn(5, 10)
    raw_preds, class_preds = predict(loaded_model, new_features)
    for i in range(len(raw_preds)):
        print(f"Sample {i+1}: Probability {raw_preds[i][0]:.4f}, Class {class_preds[i][0]}")
    
    # Clean up
    if os.path.exists('model.pt'):
        os.remove('model.pt') 
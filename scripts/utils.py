"""
ENZYMES-Hard Challenge: Utility Functions

Common utility functions for the challenge.
"""

import numpy as np
import torch
from typing import List, Dict, Optional, Tuple


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_constraints(model: torch.nn.Module, training_time: float) -> Dict[str, bool]:
    """
    Check if model meets challenge constraints.
    
    Args:
        model: PyTorch model
        training_time: Training time in seconds
    
    Returns:
        Dictionary with constraint status
    """
    num_params = count_parameters(model)
    
    return {
        'parameters_ok': num_params <= 100000,
        'time_ok': training_time <= 300,
        'num_parameters': num_params,
        'training_time': training_time
    }


def impute_features(x: np.ndarray, method: str = 'mean') -> np.ndarray:
    """
    Impute missing features.
    
    Args:
        x: Feature matrix with potential NaN values
        method: Imputation method ('mean', 'median', 'zero')
    
    Returns:
        Imputed feature matrix
    """
    x = x.copy()
    
    for j in range(x.shape[1]):
        mask = np.isnan(x[:, j])
        if mask.any():
            valid = x[~mask, j]
            if len(valid) > 0:
                if method == 'mean':
                    fill_value = valid.mean()
                elif method == 'median':
                    fill_value = np.median(valid)
                elif method == 'zero':
                    fill_value = 0.0
                else:
                    fill_value = valid.mean()
            else:
                fill_value = 0.0
            x[mask, j] = fill_value
    
    return x


def compute_class_weights(labels: List[int], num_classes: int = 6) -> torch.Tensor:
    """
    Compute class weights for imbalanced classification.
    
    Args:
        labels: List of class labels (1-indexed)
        num_classes: Number of classes
    
    Returns:
        Tensor of class weights
    """
    counts = np.zeros(num_classes)
    for label in labels:
        counts[label - 1] += 1  # Convert to 0-indexed
    
    # Inverse frequency weighting
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    
    return torch.tensor(weights, dtype=torch.float32)


def normalize_features(train_x: np.ndarray, val_x: np.ndarray = None, 
                       test_x: np.ndarray = None) -> Tuple:
    """
    Normalize features using training set statistics.
    
    Args:
        train_x: Training features
        val_x: Validation features (optional)
        test_x: Test features (optional)
    
    Returns:
        Tuple of normalized features
    """
    # Compute stats from training set (ignoring NaN)
    mean = np.nanmean(train_x, axis=0)
    std = np.nanstd(train_x, axis=0)
    std[std == 0] = 1.0  # Avoid division by zero
    
    train_norm = (train_x - mean) / std
    
    result = [train_norm]
    
    if val_x is not None:
        val_norm = (val_x - mean) / std
        result.append(val_norm)
    
    if test_x is not None:
        test_norm = (test_x - mean) / std
        result.append(test_norm)
    
    return tuple(result) if len(result) > 1 else result[0]


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'max'):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics to maximize, 'min' for metrics to minimize
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric value
        
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop

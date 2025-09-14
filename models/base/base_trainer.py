"""
Base trainer class for glucose prediction models.

This module provides an abstract base class for model training
that can be extended by specific model implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import numpy as np
from darts import TimeSeries

from .data_handler import DataHandler


class BaseGlucoseTrainer(ABC):
    """
    Abstract base class for glucose prediction model trainers.
    
    This class provides common training functionality that can be
    extended by specific model implementations.
    """
    
    def __init__(self, data_path: str = "data/t1d_glucose_data.csv"):
        """
        Initialize the base trainer.
        
        Args:
            data_path: Path to the glucose data CSV file
        """
        self.data_handler = DataHandler(data_path)
        self.model = None
        
        # Common hyperparameters
        self.RAND = 42
        
        # Set random seed
        import torch
        torch.manual_seed(self.RAND)
    
    @abstractmethod
    def create_model(self, **kwargs):
        """
        Create the model instance.
        
        This method must be implemented by subclasses.
        
        Args:
            **kwargs: Model-specific parameters
        """
        pass
    
    @abstractmethod
    def train_model(self, ts_train_scaled: TimeSeries, ts_test_scaled: TimeSeries, 
                   ts_features_scaled: TimeSeries, **kwargs) -> None:
        """
        Train the model.
        
        This method must be implemented by subclasses.
        
        Args:
            ts_train_scaled: Scaled training target data
            ts_test_scaled: Scaled test target data (for validation)
            ts_features_scaled: Scaled feature data
            **kwargs: Training-specific parameters
        """
        pass
    
    def save_model(self, model_path: str) -> str:
        """
        Save the trained model.
        
        This method can be overridden by subclasses for model-specific saving.
        
        Args:
            model_path: Path to save the model
            
        Returns:
            Path where the model was saved
        """
        print(f"Saving model to {model_path}...")
        # Default implementation - subclasses should override
        raise NotImplementedError("Model saving must be implemented by subclasses")
    
    def run_training(self, epochs: int = 100, model_name: str = "GlucoseModel", **kwargs) -> str:
        """
        Run the complete training pipeline.
        
        Args:
            epochs: Number of training epochs
            model_name: Name for the trained model
            **kwargs: Additional model-specific parameters
            
        Returns:
            Path to the trained model
        """
        print(f"Starting {self.__class__.__name__} Training")
        print("="*50)
        
        # Load and preprocess data
        ts_target, ts_features = self.data_handler.load_data()
        
        # Split data
        (ts_train, ts_test, ts_holdout, 
         ts_features_train, ts_features_test, ts_features_holdout) = self.data_handler.split_data(ts_target, ts_features)
        
        # Scale data
        (ts_train_scaled, ts_test_scaled, ts_holdout_scaled,
         ts_features_train_scaled, ts_features_test_scaled, ts_features_holdout_scaled) = self.data_handler.scale_data(
            ts_train, ts_test, ts_holdout, ts_features_train, ts_features_test, ts_features_holdout
        )
        
        # Create model
        self.create_model(epochs=epochs, model_name=model_name, **kwargs)
        
        # Train model
        # Use full feature dataset for training to avoid index issues
        ts_features_full_scaled = self.data_handler.scaler_features.transform(ts_features)
        ts_features_full_scaled = ts_features_full_scaled.astype(np.float32)
        self.train_model(ts_train_scaled, ts_test_scaled, ts_features_full_scaled, **kwargs)
        
        # Save model
        model_path = f"model_data/{model_name}"
        self.save_model(model_path)
        
        return model_path

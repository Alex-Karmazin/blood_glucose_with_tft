"""
TFT Glucose Model Trainer.

This module provides TFT-specific training functionality
extending the base trainer class.
"""

import os
from typing import Dict, List, Tuple, Optional
from darts import TimeSeries
from darts.models import TFTModel
from darts.utils.likelihood_models import QuantileRegression

from ..base.base_trainer import BaseGlucoseTrainer


class TFTGlucoseTrainer(BaseGlucoseTrainer):
    """
    TFT Glucose Model Trainer.
    
    This class provides TFT-specific training functionality
    extending the base trainer class.
    """
    
    def __init__(self, data_path: str = "data/t1d_glucose_data.csv"):
        """
        Initialize the TFT trainer.
        
        Args:
            data_path: Path to the glucose data CSV file
        """
        super().__init__(data_path)
        
        # TFT-specific hyperparameters (from notebook)
        self.QUANTILES = [0.01, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99]
        self.INLEN = 35
        self.HIDDEN = 6
        self.LSTMLAYERS = 3
        self.ATTH = 2
        self.BATCH = 48
        self.LEARN = 0.0010223
        self.DROPOUT = 0.1
        self.VALWAIT = 1
        self.N_FC = 1
    
    def create_model(self, epochs: int = 100, model_name: str = "TFT_Glucose", **kwargs):
        """
        Create the TFT model.
        
        Args:
            epochs: Number of training epochs
            model_name: Name for the model
            **kwargs: Additional model parameters
        """
        print(f"Creating TFT model with {epochs} epochs...")
        
        # Override defaults with kwargs if provided
        hidden_size = kwargs.get('hidden_size', self.HIDDEN)
        lstm_layers = kwargs.get('lstm_layers', self.LSTMLAYERS)
        num_attention_heads = kwargs.get('num_attention_heads', self.ATTH)
        dropout = kwargs.get('dropout', self.DROPOUT)
        batch_size = kwargs.get('batch_size', self.BATCH)
        learning_rate = kwargs.get('learning_rate', self.LEARN)
        input_chunk_length = kwargs.get('input_chunk_length', self.INLEN)
        output_chunk_length = kwargs.get('output_chunk_length', self.N_FC)
        
        self.model = TFTModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            hidden_size=hidden_size,
            lstm_layers=lstm_layers,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            batch_size=batch_size,
            n_epochs=epochs,
            nr_epochs_val_period=self.VALWAIT,
            likelihood=QuantileRegression(self.QUANTILES),
            optimizer_kwargs={"lr": learning_rate},
            model_name=model_name,
            log_tensorboard=kwargs.get('log_tensorboard', True),
            random_state=self.RAND,
            force_reset=kwargs.get('force_reset', True),
            save_checkpoints=kwargs.get('save_checkpoints', True),
            add_relative_index=True
        )
        
        print("✓ TFT model created successfully")
    
    def train_model(self, ts_train_scaled: TimeSeries, ts_test_scaled: TimeSeries, 
                   ts_features_scaled: TimeSeries, **kwargs) -> None:
        """
        Train the TFT model.
        
        Args:
            ts_train_scaled: Scaled training target data
            ts_test_scaled: Scaled test target data (for validation)
            ts_features_scaled: Scaled feature data
            **kwargs: Training-specific parameters
        """
        print("Training TFT model...")
        try:
            self.model.fit(
                series=ts_train_scaled,
                past_covariates=ts_features_scaled,
                val_series=ts_test_scaled,
                val_past_covariates=ts_features_scaled,
                verbose=kwargs.get('verbose', True)
            )
            print("✓ TFT model training completed")
        except Exception as e:
            print(f"✗ Error during TFT training: {e}")
            raise
    
    def save_model(self, model_path: str) -> str:
        """
        Save the trained TFT model.
        
        Args:
            model_path: Path to save the model
            
        Returns:
            Path where the model was saved
        """
        print(f"Saving TFT model to {model_path}...")
        
        # Create directory if it doesn't exist
        os.makedirs(model_path, exist_ok=True)
        
        # Save the model using Darts' save method
        if self.model is not None:
            # Darts saves as .pt files
            model_file_path = os.path.join(model_path, "model.pt")
            self.model.save(model_file_path)
            print(f"✓ TFT model saved to: {model_file_path}")
            return model_file_path
        else:
            print("✗ No model to save")
            raise ValueError("No trained model available to save")
    
    def get_model_info(self) -> Dict:
        """
        Get information about the TFT model.
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"status": "No model created"}
        
        return {
            "model_type": "TFT (Temporal Fusion Transformer)",
            "input_chunk_length": self.INLEN,
            "output_chunk_length": self.N_FC,
            "hidden_size": self.HIDDEN,
            "lstm_layers": self.LSTMLAYERS,
            "attention_heads": self.ATTH,
            "dropout": self.DROPOUT,
            "batch_size": self.BATCH,
            "learning_rate": self.LEARN,
            "quantiles": self.QUANTILES,
            "status": "Model created"
        }
    
    def print_model_info(self):
        """Print information about the TFT model."""
        info = self.get_model_info()
        print("\nTFT Model Information:")
        print("=" * 40)
        for key, value in info.items():
            print(f"{key.replace('_', ' ').title()}: {value}")

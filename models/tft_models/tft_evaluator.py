"""
TFT Glucose Model Evaluator.

This module provides TFT-specific evaluation functionality
extending the base evaluator class.
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from darts import TimeSeries
from darts.models import TFTModel
from darts.utils.likelihood_models import QuantileRegression

from ..base.base_evaluator import BaseGlucoseEvaluator


class TFTGlucoseEvaluator(BaseGlucoseEvaluator):
    """
    TFT Glucose Model Evaluator.
    
    This class provides TFT-specific evaluation functionality
    extending the base evaluator class.
    """
    
    def __init__(self, data_path: str = "data/t1d_glucose_data.csv"):
        """
        Initialize the TFT evaluator.
        
        Args:
            data_path: Path to the glucose data CSV file
        """
        super().__init__(data_path)

        
        # TFT-specific hyperparameters (from notebook)
        self.INLEN = 35
        self.HIDDEN = 6
        self.LSTMLAYERS = 3
        self.ATTH = 2
        self.BATCH = 48
        self.LEARN = 0.0010223
        self.DROPOUT = 0.1
        self.VALWAIT = 1
        self.N_FC = 1
        self.N_SAMPLES = 100
    
    def create_model(self, epochs: int = 5, model_name: str = "TFT_Glucose_Eval", **kwargs):
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
        
        self.model = TFTModel(
            input_chunk_length=self.INLEN,
            output_chunk_length=self.N_FC,
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
            log_tensorboard=False,
            random_state=self.RAND,
            force_reset=True,
            save_checkpoints=False,
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
                verbose=kwargs.get('verbose', False)
            )
            print("✓ TFT model training completed")
        except Exception as e:
            print(f"✗ Error during TFT training: {e}")
            raise
    
    def generate_predictions(self, ts_input: TimeSeries, ts_features: TimeSeries, 
                           n_steps: int, **kwargs) -> TimeSeries:
        """
        Generate predictions using the TFT model.
        
        Args:
            ts_input: Input time series for prediction
            ts_features: Feature time series
            n_steps: Number of steps to predict
            **kwargs: Prediction-specific parameters
            
        Returns:
            Predicted time series with quantiles
        """
        print(f"Generating TFT predictions for {n_steps} steps...")
        
        try:
            predictions = self.model.predict(
                n=n_steps,
                series=ts_input,
                past_covariates=ts_features,
                num_samples=kwargs.get('num_samples', self.N_SAMPLES),
                n_jobs=kwargs.get('n_jobs', 12),
                verbose=kwargs.get('verbose', False)
            )
            print("✓ TFT predictions generated successfully")
            return predictions
        except Exception as e:
            print(f"✗ Error generating TFT predictions: {e}")
            return None
    
    def load_model(self, model_path: str):
        """
        Load an existing TFT model.
        
        Args:
            model_path: Path to the TFT model directory
        """
        print(f"Loading TFT model from {model_path}...")
        try:
            # Look for the model.pt file in the directory
            model_file_path = os.path.join(model_path, "model.pt")
            if os.path.exists(model_file_path):
                self.model = TFTModel.load(model_file_path)
                print("✓ TFT model loaded successfully")
            else:
                raise FileNotFoundError(f"Model file not found: {model_file_path}")
        except Exception as e:
            print(f"✗ Error loading TFT model: {e}")
            raise
    
    def get_model_info(self) -> Dict:
        """
        Get information about the TFT model.
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"status": "No model loaded"}
        
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
            "status": "Model loaded"
        }
    
    def print_model_info(self):
        """Print information about the TFT model."""
        info = self.get_model_info()
        print("\nTFT Model Information:")
        print("=" * 40)
        for key, value in info.items():
            print(f"{key.replace('_', ' ').title()}: {value}")

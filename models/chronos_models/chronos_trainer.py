"""
Chronos Glucose Model Trainer.

This module provides Chronos-specific training functionality
extending the base trainer class.
"""

import os
from typing import Dict, List, Tuple, Optional
from darts import TimeSeries
from chronos import BaseChronosPipeline

from ..base.base_trainer import BaseGlucoseTrainer


class ChronosGlucoseTrainer(BaseGlucoseTrainer):
    """
    Chronos Glucose Model Trainer.
    
    This class provides Chronos-specific training functionality
    extending the base trainer class.
    
    Note: Chronos models are pretrained foundation models that are typically
    used for zero-shot forecasting. Fine-tuning is possible but requires
    additional setup and is not implemented in this basic version.
    """
    
    def __init__(self, data_path: str = "data/t1d_glucose_data.csv"):
        """
        Initialize the Chronos trainer.
        
        Args:
            data_path: Path to the glucose data CSV file
        """
        super().__init__(data_path)
        
        # Chronos-specific hyperparameters
        self.model_name = "amazon/chronos-t5-small"  # Default model
        self.device = "cuda"
        self.torch_dtype = "bfloat16"
        
        # Chronos doesn't use traditional quantiles, but we can simulate them
        self.QUANTILES = [0.1, 0.2, 0.5, 0.8, 0.9]  # Different from TFT
    
    def create_model(self, model_name: str = "amazon/chronos-t5-small", **kwargs):
        """
        Create the Chronos model.
        
        Args:
            model_name: Name of the Chronos model to use
            **kwargs: Additional model parameters
        """
        print(f"Creating Chronos model: {model_name}...")
        
        # Override defaults with kwargs if provided
        self.model_name = kwargs.get('model_name', model_name)
        self.device = kwargs.get('device', self.device)
        self.torch_dtype = kwargs.get('torch_dtype', self.torch_dtype)
        
        try:
            import torch
            torch_dtype = torch.bfloat16 if self.torch_dtype == "bfloat16" else torch.float32
            
            self.model = BaseChronosPipeline.from_pretrained(
                self.model_name,
                device_map=self.device,
                torch_dtype=torch_dtype,
            )
            print("✓ Chronos model created successfully")
        except Exception as e:
            print(f"✗ Error creating Chronos model: {e}")
            raise
    
    def train_model(self, ts_train_scaled: TimeSeries, ts_test_scaled: TimeSeries, 
                   ts_features_scaled: TimeSeries, **kwargs) -> None:
        """
        Train the Chronos model.
        
        Note: Chronos models are pretrained and typically used for zero-shot forecasting.
        Fine-tuning is possible but requires additional setup. For now, we'll use the
        pretrained model directly.
        
        Args:
            ts_train_scaled: Scaled training target data (not used for pretrained models)
            ts_test_scaled: Scaled test target data (not used for pretrained models)
            ts_features_scaled: Scaled feature data (not used for pretrained models)
            **kwargs: Training-specific parameters
        """
        print("Using pretrained Chronos model (no training required)")
        print("Note: Chronos models are foundation models designed for zero-shot forecasting.")
        print("Fine-tuning is possible but requires additional setup.")
        print("✓ Chronos model ready for inference")
    
    def save_model(self, model_path: str) -> str:
        """
        Save the Chronos model.
        
        Note: Since Chronos models are pretrained, we don't actually save anything.
        The model can be loaded directly from the pretrained weights.
        
        Args:
            model_path: Path to save the model (not used for pretrained models)
            
        Returns:
            Path where the model was saved (model name for pretrained models)
        """
        print(f"Chronos model is pretrained and doesn't need saving.")
        print(f"Model can be loaded directly using: {self.model_name}")
        
        # Create directory and save model info
        os.makedirs(model_path, exist_ok=True)
        info_file_path = os.path.join(model_path, "model_info.json")
        
        import json
        from datetime import datetime
        model_info = self.get_model_info()
        model_info['save_timestamp'] = datetime.now().isoformat()
        model_info['model_path'] = self.model_name
        
        with open(info_file_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"✓ Model info saved to: {info_file_path}")
        
        return self.model_name
    
    def get_model_info(self) -> Dict:
        """
        Get information about the Chronos model.
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"status": "No model created"}
        
        return {
            "model_type": "Chronos (Pretrained Foundation Model)",
            "model_name": self.model_name,
            "device": self.device,
            "torch_dtype": self.torch_dtype,
            "quantiles": self.QUANTILES,
            "status": "Model created"
        }
    
    def print_model_info(self):
        """Print information about the Chronos model."""
        info = self.get_model_info()
        print("\nChronos Model Information:")
        print("=" * 40)
        for key, value in info.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available Chronos models.
        
        Returns:
            List of available model names
        """
        return [
            "amazon/chronos-t5-tiny",
            "amazon/chronos-t5-small", 
            "amazon/chronos-t5-base",
            "amazon/chronos-t5-large",
            "amazon/chronos-bolt-small",
            "amazon/chronos-bolt-base",
        ]
    
    def print_available_models(self):
        """Print available Chronos models."""
        models = self.get_available_models()
        print("\nAvailable Chronos Models:")
        print("=" * 40)
        for model in models:
            print(f"  {model}")
        print("\nNote: Chronos models are pretrained and ready for zero-shot forecasting.")
        print("No training is required, but fine-tuning is possible for domain-specific data.")
    
    def run_training(self, epochs: int = 100, model_name: str = "Chronos_Glucose", **kwargs) -> str:
        """
        Run the complete training pipeline.
        
        Note: For Chronos models, this is more of a "setup" pipeline since
        the models are pretrained and don't require traditional training.
        
        Args:
            epochs: Number of training epochs (not used for pretrained models)
            model_name: Name for the model
            **kwargs: Additional model-specific parameters
            
        Returns:
            Model name (since it's pretrained)
        """
        print(f"Starting {self.__class__.__name__} Setup")
        print("="*50)
        
        # Load and preprocess data (for validation)
        ts_target, ts_features = self.data_handler.load_data()
        
        # Split data (for validation)
        (ts_train, ts_test, ts_holdout, 
         ts_features_train, ts_features_test, ts_features_holdout) = self.data_handler.split_data(ts_target, ts_features)
        
        # Scale data (for validation)
        (ts_train_scaled, ts_test_scaled, ts_holdout_scaled,
         ts_features_train_scaled, ts_features_test_scaled, ts_features_holdout_scaled) = self.data_handler.scale_data(
            ts_train, ts_test, ts_holdout, ts_features_train, ts_features_test, ts_features_holdout
        )
        
        # Create model
        self.create_model(model_name=self.model_name, **kwargs)
        
        # "Train" model (actually just validate it's working)
        self.train_model(ts_train_scaled, ts_test_scaled, ts_features_train_scaled, **kwargs)
        
        # Return model name (since it's pretrained)
        return self.model_name

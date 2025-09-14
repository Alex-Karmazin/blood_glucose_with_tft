#!/usr/bin/env python3
"""
TFT Glucose Model Training Script

This script trains a TFT model for glucose prediction based on the Model and Forecast Notebook.
It can be used to train a model before running inference and evaluation.

Usage:
    python train_tft_model.py [--epochs EPOCHS] [--model_name MODEL_NAME]
"""

import argparse
import pandas as pd
import numpy as np
import torch
import warnings
import logging
from typing import Tuple

# Darts imports
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel
from darts.utils.likelihood_models import QuantileRegression

# Suppress warnings
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

class TFTGlucoseTrainer:
    """
    TFT Glucose Model Trainer
    
    This class handles training a TFT model for glucose prediction.
    """
    
    def __init__(self, data_path: str = "t1d_glucose_data.csv"):
        """
        Initialize the trainer.
        
        Args:
            data_path: Path to the glucose data CSV file
        """
        self.data_path = data_path
        self.model = None
        self.scaler_target = None
        self.scaler_features = None
        
        # Model hyperparameters (from notebook)
        self.QUANTILES = [0.01, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99]
        self.SPLIT = 0.90
        self.INLEN = 35
        self.HIDDEN = 6
        self.LSTMLAYERS = 3
        self.ATTH = 2
        self.BATCH = 48
        self.LEARN = 0.0010223
        self.DROPOUT = 0.1
        self.VALWAIT = 1
        self.N_FC = 1
        self.RAND = 42
        self.N_SAMPLES = 100
        self.N_JOBS = 12
        
        # Set random seed
        torch.manual_seed(self.RAND)
    
    def load_and_preprocess_data(self) -> Tuple[TimeSeries, TimeSeries]:
        """
        Load and preprocess the glucose data.
        
        Returns:
            Tuple of (target_time_series, features_time_series)
        """
        print("Loading and preprocessing data...")
        
        # Load data
        data = pd.read_csv(self.data_path)
        data.rename(columns={"date_time": "datetime_col"}, inplace=True)
        data['datetime_col'] = pd.to_datetime(data['datetime_col'])
        
        # Convert data types
        data['bolus'] = data['bolus'].astype('float32')
        data['carbs'] = data['carbs'].astype('float32')
        data['insulin_on_board'] = data['insulin_on_board'].astype('float32')
        
        # Convert float64 columns to float32
        int_columns = data.select_dtypes(include=['float64']).columns
        data[int_columns] = data[int_columns].astype('float32')
        
        # Create time series objects
        ts_target = TimeSeries.from_dataframe(
            data, 'datetime_col', 'glucose_value', freq='5T'
        )
        
        # Extract features
        feature_cols = [col for col in data.columns 
                       if col not in ['glucose_value', 'datetime_col']]
        df_features = data[['datetime_col'] + feature_cols]
        ts_features = TimeSeries.from_dataframe(df_features, 'datetime_col', freq='5T')
        
        print(f"✓ Data loaded: {len(data)} rows, {len(feature_cols)} features")
        print(f"✓ Time range: {data['datetime_col'].min()} to {data['datetime_col'].max()}")
        
        return ts_target, ts_features
    
    def split_data(self, ts_target: TimeSeries, ts_features: TimeSeries) -> Tuple:
        """
        Split data into train, test, and holdout sets.
        
        Returns:
            Tuple of split time series
        """
        print("Splitting data into train/test/holdout sets...")
        
        # Calculate split points
        train_size = int(len(ts_target) * self.SPLIT)
        split_timestamp = ts_target.to_dataframe().index[train_size]
        
        # Split target data
        ts_train, ts_temp = ts_target.split_after(split_timestamp)
        test_size = int(len(ts_temp) * 0.5)
        split_timestamp_test = ts_temp.to_dataframe().index[test_size]
        ts_test, ts_holdout = ts_temp.split_after(split_timestamp_test)
        
        # Split feature data
        ts_features_train, ts_features_temp = ts_features.split_after(split_timestamp)
        ts_features_test, ts_features_holdout = ts_features_temp.split_after(split_timestamp_test)
        
        print(f"✓ Training set: {len(ts_train)} points ({ts_train.duration})")
        print(f"✓ Test set: {len(ts_test)} points ({ts_test.duration})")
        print(f"✓ Holdout set: {len(ts_holdout)} points ({ts_holdout.duration})")
        
        return (ts_train, ts_test, ts_holdout, 
                ts_features_train, ts_features_test, ts_features_holdout)
    
    def scale_data(self, ts_train: TimeSeries, ts_test: TimeSeries, ts_holdout: TimeSeries,
                   ts_features_train: TimeSeries, ts_features_test: TimeSeries, 
                   ts_features_holdout: TimeSeries) -> Tuple:
        """
        Scale the data.
        
        Returns:
            Tuple of scaled time series
        """
        print("Scaling data...")
        
        # Scale target data
        self.scaler_target = Scaler()
        self.scaler_target.fit(ts_train)
        ts_train_scaled = self.scaler_target.transform(ts_train)
        ts_test_scaled = self.scaler_target.transform(ts_test)
        ts_holdout_scaled = self.scaler_target.transform(ts_holdout)
        
        # Scale feature data
        self.scaler_features = Scaler()
        self.scaler_features.fit(ts_features_train)
        ts_features_train_scaled = self.scaler_features.transform(ts_features_train)
        ts_features_test_scaled = self.scaler_features.transform(ts_features_test)
        ts_features_holdout_scaled = self.scaler_features.transform(ts_features_holdout)
        
        # Convert to float32
        ts_train_scaled = ts_train_scaled.astype(np.float32)
        ts_test_scaled = ts_test_scaled.astype(np.float32)
        ts_holdout_scaled = ts_holdout_scaled.astype(np.float32)
        ts_features_train_scaled = ts_features_train_scaled.astype(np.float32)
        ts_features_test_scaled = ts_features_test_scaled.astype(np.float32)
        ts_features_holdout_scaled = ts_features_holdout_scaled.astype(np.float32)
        
        print("✓ Data scaling completed")
        
        return (ts_train_scaled, ts_test_scaled, ts_holdout_scaled,
                ts_features_train_scaled, ts_features_test_scaled, ts_features_holdout_scaled)
    
    def create_model(self, epochs: int = 100, model_name: str = "TFT_Glucose") -> TFTModel:
        """
        Create the TFT model.
        
        Args:
            epochs: Number of training epochs
            model_name: Name for the model
            
        Returns:
            TFT model instance
        """
        print(f"Creating TFT model with {epochs} epochs...")
        
        model = TFTModel(
            input_chunk_length=self.INLEN,
            output_chunk_length=self.N_FC,
            hidden_size=self.HIDDEN,
            lstm_layers=self.LSTMLAYERS,
            num_attention_heads=self.ATTH,
            dropout=self.DROPOUT,
            batch_size=self.BATCH,
            n_epochs=epochs,
            nr_epochs_val_period=self.VALWAIT,
            likelihood=QuantileRegression(self.QUANTILES),
            optimizer_kwargs={"lr": self.LEARN},
            model_name=model_name,
            log_tensorboard=True,
            random_state=self.RAND,
            force_reset=True,
            save_checkpoints=True,
            add_relative_index=True
        )
        
        print("✓ Model created successfully")
        return model
    
    def train_model(self, model: TFTModel, ts_train_scaled: TimeSeries, 
                   ts_test_scaled: TimeSeries, ts_features_scaled: TimeSeries) -> None:
        """
        Train the TFT model.
        
        Args:
            model: TFT model to train
            ts_train_scaled: Scaled training target data
            ts_test_scaled: Scaled test target data (for validation)
            ts_features_scaled: Scaled feature data
        """
        print("Starting model training...")
        print("This may take a while depending on the number of epochs...")
        
        try:
            model.fit(
                series=ts_train_scaled,
                past_covariates=ts_features_scaled,
                val_series=ts_test_scaled,
                val_past_covariates=ts_features_scaled,
                verbose=True
            )
            print("✓ Model training completed successfully")
        except Exception as e:
            print(f"✗ Error during training: {e}")
            raise
    
    def run_training(self, epochs: int = 100, model_name: str = "TFT_Glucose") -> str:
        """
        Run the complete training pipeline.
        
        Args:
            epochs: Number of training epochs
            model_name: Name for the model
            
        Returns:
            Path to the trained model
        """
        print("Starting TFT Glucose Model Training")
        print("="*50)
        
        # Load and preprocess data
        ts_target, ts_features = self.load_and_preprocess_data()
        
        # Split data
        (ts_train, ts_test, ts_holdout, 
         ts_features_train, ts_features_test, ts_features_holdout) = self.split_data(ts_target, ts_features)
        
        # Scale data
        (ts_train_scaled, ts_test_scaled, ts_holdout_scaled,
         ts_features_train_scaled, ts_features_test_scaled, ts_features_holdout_scaled) = self.scale_data(
            ts_train, ts_test, ts_holdout, ts_features_train, ts_features_test, ts_features_holdout
        )
        
        # Create model
        model = self.create_model(epochs, model_name)
        
        # Train model
        self.train_model(model, ts_train_scaled, ts_test_scaled, ts_features_train_scaled)
        
        # Save model
        model_path = f"{model_name}"
        print(f"✓ Model saved to: {model_path}")
        
        return model_path

def main():
    """Main function to run the training."""
    parser = argparse.ArgumentParser(description='TFT Glucose Model Training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--model_name', type=str, default='TFT_Glucose',
                       help='Name for the trained model (default: TFT_Glucose)')
    parser.add_argument('--data_path', type=str, default='t1d_glucose_data.csv',
                       help='Path to the glucose data CSV file')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = TFTGlucoseTrainer(data_path=args.data_path)
    
    # Run training
    try:
        model_path = trainer.run_training(epochs=args.epochs, model_name=args.model_name)
        
        print(f"\n{'='*80}")
        print("TRAINING COMPLETED SUCCESSFULLY")
        print(f"{'='*80}")
        print(f"Model saved to: {model_path}")
        print(f"You can now run inference using:")
        print(f"  python tft_inference_evaluation.py --model_path {model_path}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

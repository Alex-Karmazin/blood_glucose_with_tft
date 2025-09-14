#!/usr/bin/env python3
"""
TFT Glucose Model Performance Evaluation Script

This script evaluates TFT model performance on glucose prediction using multiple metrics.
It can work with a pre-trained model or train a quick model for evaluation.

Usage:
    python evaluate_tft_performance.py [--model_path MODEL_PATH] [--quick_train] [--epochs EPOCHS]
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import warnings
import logging
from typing import Dict, List, Tuple, Optional
import os
from datetime import datetime

# Darts imports
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel
from darts.metrics import rmse, mape, mae, smape
from darts.utils.likelihood_models import QuantileRegression

# Suppress warnings
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

class TFTPerformanceEvaluator:
    """
    TFT Glucose Model Performance Evaluator
    
    This class evaluates TFT model performance using multiple metrics and quantiles.
    """
    
    def __init__(self, data_path: str = "t1d_glucose_data.csv"):
        """
        Initialize the evaluator.
        
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
        
        # Metrics to calculate
        self.metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'SMAPE': smape
        }
        
        # Set random seed
        import torch
        torch.manual_seed(self.RAND)
    
    def load_data(self) -> Tuple[TimeSeries, TimeSeries]:
        """Load and preprocess the glucose data."""
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
        """Split data into train, test, and holdout sets."""
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
        """Scale the data."""
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
    
    def load_or_create_model(self, model_path: Optional[str] = None, 
                           quick_train: bool = False, epochs: int = 5) -> TFTModel:
        """
        Load existing model or create and train a new one.
        
        Args:
            model_path: Path to existing model
            quick_train: Whether to do quick training
            epochs: Number of epochs for quick training
            
        Returns:
            TFT model
        """
        if model_path and os.path.exists(model_path):
            print(f"Loading existing model from {model_path}...")
            try:
                self.model = TFTModel.load_from_checkpoint(model_path)
                print("✓ Model loaded successfully")
                return self.model
            except Exception as e:
                print(f"✗ Error loading model: {e}")
                print("Creating new model...")
        
        print("Creating new TFT model...")
        self.model = TFTModel(
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
            model_name="TFT_Glucose_Eval",
            log_tensorboard=False,
            random_state=self.RAND,
            force_reset=True,
            save_checkpoints=False,
            add_relative_index=True
        )
        
        if quick_train:
            print(f"Quick training model for {epochs} epochs...")
            return self.model
        else:
            print("✓ Model created (not trained)")
            return self.model
    
    def train_model(self, ts_train_scaled: TimeSeries, ts_test_scaled: TimeSeries, 
                   ts_features_scaled: TimeSeries) -> None:
        """Train the model."""
        print("Training model...")
        try:
            self.model.fit(
                series=ts_train_scaled,
                past_covariates=ts_features_scaled,
                val_series=ts_test_scaled,
                val_past_covariates=ts_features_scaled,
                verbose=False
            )
            print("✓ Model training completed")
        except Exception as e:
            print(f"✗ Error during training: {e}")
            raise
    
    def generate_predictions(self, ts_input: TimeSeries, ts_features: TimeSeries, 
                           n_steps: int) -> TimeSeries:
        """Generate predictions."""
        print(f"Generating predictions for {n_steps} steps...")
        
        try:
            predictions = self.model.predict(
                n=n_steps,
                series=ts_input,
                past_covariates=ts_features,
                num_samples=self.N_SAMPLES,
                n_jobs=self.N_JOBS,
                verbose=False
            )
            print("✓ Predictions generated successfully")
            return predictions
        except Exception as e:
            print(f"✗ Error generating predictions: {e}")
            return None
    
    def evaluate_quantiles(self, predictions: TimeSeries, actual: TimeSeries, 
                          dataset_name: str) -> Dict:
        """Evaluate model performance across different quantiles."""
        print(f"\nEvaluating {dataset_name} dataset...")
        
        results = {
            'dataset': dataset_name,
            'quantiles': {},
            'summary': {}
        }
        
        # Evaluate each quantile
        for quantile in self.QUANTILES:
            try:
                # Extract quantile predictions
                q_pred = predictions.quantile(quantile)
                
                # Inverse transform predictions
                q_pred_inverse = self.scaler_target.inverse_transform(q_pred)
                
                # Calculate metrics
                quantile_results = {}
                for metric_name, metric_func in self.metrics.items():
                    try:
                        value = metric_func(q_pred_inverse, actual)
                        quantile_results[metric_name] = float(value)
                    except Exception as e:
                        print(f"Warning: Could not calculate {metric_name} for quantile {quantile}: {e}")
                        quantile_results[metric_name] = np.nan
                
                results['quantiles'][f'Q{int(quantile*100):02d}'] = quantile_results
                
            except Exception as e:
                print(f"Error evaluating quantile {quantile}: {e}")
                results['quantiles'][f'Q{int(quantile*100):02d}'] = {
                    metric: np.nan for metric in self.metrics.keys()
                }
        
        # Find best quantile for each metric
        for metric_name in self.metrics.keys():
            best_quantile = None
            best_value = np.inf
            
            for q_name, q_results in results['quantiles'].items():
                if metric_name in q_results and not np.isnan(q_results[metric_name]):
                    if q_results[metric_name] < best_value:
                        best_value = q_results[metric_name]
                        best_quantile = q_name
            
            if best_quantile:
                results['summary'][f'best_{metric_name.lower()}'] = {
                    'quantile': best_quantile,
                    'value': best_value
                }
        
        return results
    
    def print_results(self, results: Dict) -> None:
        """Print evaluation results in a formatted table."""
        print(f"\n{'='*80}")
        print(f"EVALUATION RESULTS - {results['dataset'].upper()}")
        print(f"{'='*80}")
        
        # Create results table
        table_data = []
        for q_name, q_results in results['quantiles'].items():
            row = [q_name]
            for metric_name in self.metrics.keys():
                value = q_results.get(metric_name, np.nan)
                if not np.isnan(value):
                    row.append(f"{value:.2f}")
                else:
                    row.append("N/A")
            table_data.append(row)
        
        headers = ['Quantile'] + list(self.metrics.keys())
        print(tabulate(table_data, headers=headers, tablefmt='psql', floatfmt='.2f'))
        
        # Print best quantiles
        print(f"\n{'Best Quantiles by Metric:'}")
        print(f"{'='*40}")
        for metric_name in self.metrics.keys():
            best_info = results['summary'].get(f'best_{metric_name.lower()}', {})
            if best_info:
                print(f"{metric_name:>8}: {best_info['quantile']} ({best_info['value']:.2f})")
    
    def plot_predictions(self, predictions: TimeSeries, actual: TimeSeries, 
                        dataset_name: str, quantiles_to_plot: List[float] = [0.2, 0.5, 0.8]) -> None:
        """Plot predictions vs actual values for selected quantiles."""
        print(f"\nGenerating plots for {dataset_name}...")
        
        # Create figure
        fig, axes = plt.subplots(len(quantiles_to_plot), 1, figsize=(15, 4*len(quantiles_to_plot)))
        if len(quantiles_to_plot) == 1:
            axes = [axes]
        
        # Convert to pandas for easier plotting
        actual_pd = actual.to_dataframe().iloc[:, 0]  # Get first column as series
        
        for i, quantile in enumerate(quantiles_to_plot):
            try:
                # Get quantile predictions
                q_pred = predictions.quantile(quantile)
                q_pred_inverse = self.scaler_target.inverse_transform(q_pred)
                q_pred_pd = q_pred_inverse.to_dataframe().iloc[:, 0]  # Get first column as series
                
                # Plot
                axes[i].plot(actual_pd.index, actual_pd.values, label='Actual', linewidth=2, alpha=0.8)
                axes[i].plot(q_pred_pd.index, q_pred_pd.values, label=f'Q{int(quantile*100):02d} Prediction', 
                           linewidth=2, alpha=0.8)
                
                axes[i].set_title(f'{dataset_name} - Q{int(quantile*100):02d} Quantile')
                axes[i].set_ylabel('Glucose (mg/dL)')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
                
                # Calculate and display RMSE
                rmse_value = rmse(q_pred_inverse, actual)
                axes[i].text(0.02, 0.98, f'RMSE: {rmse_value:.2f}', 
                           transform=axes[i].transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
            except Exception as e:
                print(f"Error plotting quantile {quantile}: {e}")
        
        plt.tight_layout()
        plt.savefig(f'{dataset_name.lower().replace(" ", "_")}_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_evaluation(self, model_path: Optional[str] = None, quick_train: bool = False, 
                      epochs: int = 5, use_holdout: bool = True) -> Dict:
        """
        Run complete evaluation pipeline.
        
        Args:
            model_path: Path to existing model
            quick_train: Whether to do quick training
            epochs: Number of epochs for quick training
            use_holdout: Whether to evaluate on holdout data
            
        Returns:
            Dictionary with all evaluation results
        """
        print("Starting TFT Glucose Model Performance Evaluation")
        print("="*60)
        
        # Load data
        ts_target, ts_features = self.load_data()
        
        # Split data
        (ts_train, ts_test, ts_holdout, 
         ts_features_train, ts_features_test, ts_features_holdout) = self.split_data(ts_target, ts_features)
        
        # Scale data
        (ts_train_scaled, ts_test_scaled, ts_holdout_scaled,
         ts_features_train_scaled, ts_features_test_scaled, ts_features_holdout_scaled) = self.scale_data(
            ts_train, ts_test, ts_holdout, ts_features_train, ts_features_test, ts_features_holdout
        )
        
        # Load or create model
        self.load_or_create_model(model_path, quick_train, epochs)
        
        # Train model if needed
        if quick_train or not model_path:
            # Use full feature dataset for training to avoid index issues
            ts_features_full_scaled = self.scaler_features.transform(ts_features)
            ts_features_full_scaled = ts_features_full_scaled.astype(np.float32)
            self.train_model(ts_train_scaled, ts_test_scaled, ts_features_full_scaled)
        
        # Choose evaluation dataset
        if use_holdout:
            eval_ts_scaled = ts_holdout_scaled
            eval_ts_features_scaled = ts_features_holdout_scaled
            eval_ts_original = ts_holdout
            eval_name = "Holdout"
        else:
            eval_ts_scaled = ts_test_scaled
            eval_ts_features_scaled = ts_features_test_scaled
            eval_ts_original = ts_test
            eval_name = "Test"
        
        # Generate predictions
        # Use full feature dataset for prediction to avoid index issues
        ts_features_full_scaled = self.scaler_features.transform(ts_features)
        ts_features_full_scaled = ts_features_full_scaled.astype(np.float32)
        
        predictions = self.generate_predictions(
            ts_test_scaled,  # Use test data as input for prediction
            ts_features_full_scaled,
            len(eval_ts_scaled)
        )
        
        if predictions is None:
            print("✗ Could not generate predictions. Exiting.")
            return {}
        
        # Evaluate performance
        results = self.evaluate_quantiles(predictions, eval_ts_original, eval_name)
        
        # Print results
        self.print_results(results)
        
        # Generate plots
        self.plot_predictions(predictions, eval_ts_original, eval_name)
        
        return results

def main():
    """Main function to run the evaluation."""
    parser = argparse.ArgumentParser(description='TFT Glucose Model Performance Evaluation')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to the trained model directory')
    parser.add_argument('--quick_train', action='store_true',
                       help='Do quick training with few epochs')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of epochs for quick training (default: 5)')
    parser.add_argument('--data_path', type=str, default='t1d_glucose_data.csv',
                       help='Path to the glucose data CSV file')
    parser.add_argument('--use_test', action='store_true',
                       help='Use test data instead of holdout data for evaluation')
    
    args = parser.parse_args()
    
    # Determine which dataset to use
    use_holdout = not args.use_test
    
    # Create evaluator
    evaluator = TFTPerformanceEvaluator(data_path=args.data_path)
    
    # Run evaluation
    try:
        results = evaluator.run_evaluation(
            model_path=args.model_path,
            quick_train=args.quick_train,
            epochs=args.epochs,
            use_holdout=use_holdout
        )
        
        if results:
            print(f"\n{'='*80}")
            print("EVALUATION COMPLETED SUCCESSFULLY")
            print(f"{'='*80}")
            print(f"Results saved and plots generated.")
            print(f"Best performing quantiles:")
            for metric_name in evaluator.metrics.keys():
                best_info = results['summary'].get(f'best_{metric_name.lower()}', {})
                if best_info:
                    print(f"  {metric_name}: {best_info['quantile']} (value: {best_info['value']:.2f})")
        else:
            print("Evaluation failed.")
            
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

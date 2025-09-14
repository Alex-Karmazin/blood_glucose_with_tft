#!/usr/bin/env python3
"""
Simple TFT Glucose Model Performance Evaluation Script

This script provides a simplified evaluation of TFT model performance on glucose prediction.
It focuses on core functionality and provides clear error rate estimates.

Usage:
    python simple_tft_evaluation.py [--model_path MODEL_PATH] [--quick_train] [--epochs EPOCHS]
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

# Darts imports
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel
from darts.metrics import rmse, mape, mae, smape
from darts.utils.likelihood_models import QuantileRegression

# Suppress warnings
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

class SimpleTFTEvaluator:
    """
    Simple TFT Glucose Model Evaluator
    
    This class provides a streamlined evaluation of TFT model performance.
    """
    
    def __init__(self, data_path: str = "t1d_glucose_data.csv"):
        """Initialize the evaluator."""
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
    
    def create_and_train_model(self, ts_train_scaled: TimeSeries, ts_test_scaled: TimeSeries, 
                              ts_features_scaled: TimeSeries, epochs: int = 5) -> TFTModel:
        """Create and train a TFT model."""
        print(f"Creating and training TFT model for {epochs} epochs...")
        
        # Create model
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
            model_name="TFT_Glucose_Simple",
            log_tensorboard=False,
            random_state=self.RAND,
            force_reset=True,
            save_checkpoints=False,
            add_relative_index=True
        )
        
        # Train model
        try:
            model.fit(
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
        
        return model
    
    def load_model(self, model_path: str) -> TFTModel:
        """Load an existing model."""
        print(f"Loading model from {model_path}...")
        try:
            model = TFTModel.load_from_checkpoint(model_path)
            print("✓ Model loaded successfully")
            return model
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise
    
    def generate_predictions(self, model: TFTModel, ts_input: TimeSeries, 
                           ts_features: TimeSeries, n_steps: int) -> TimeSeries:
        """Generate predictions using the model."""
        print(f"Generating predictions for {n_steps} steps...")
        
        try:
            predictions = model.predict(
                n=n_steps,
                series=ts_input,
                past_covariates=ts_features,
                num_samples=self.N_SAMPLES,
                verbose=False
            )
            print("✓ Predictions generated successfully")
            return predictions
        except Exception as e:
            print(f"✗ Error generating predictions: {e}")
            return None
    
    def evaluate_performance(self, predictions: TimeSeries, actual: TimeSeries, 
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
                rmse_val = rmse(q_pred_inverse, actual)
                mae_val = mae(q_pred_inverse, actual)
                
                # Calculate MAPE (handle division by zero)
                actual_values = actual.values().flatten()
                pred_values = q_pred_inverse.values().flatten()
                
                # Avoid division by zero
                mape_val = np.mean(np.abs((actual_values - pred_values) / np.maximum(actual_values, 1e-8))) * 100
                
                # Calculate SMAPE
                smape_val = np.mean(2 * np.abs(actual_values - pred_values) / 
                                  (np.abs(actual_values) + np.abs(pred_values) + 1e-8)) * 100
                
                quantile_results = {
                    'RMSE': float(rmse_val),
                    'MAE': float(mae_val),
                    'MAPE': float(mape_val),
                    'SMAPE': float(smape_val)
                }
                
                results['quantiles'][f'Q{int(quantile*100):02d}'] = quantile_results
                
            except Exception as e:
                print(f"Error evaluating quantile {quantile}: {e}")
                results['quantiles'][f'Q{int(quantile*100):02d}'] = {
                    'RMSE': np.nan, 'MAE': np.nan, 'MAPE': np.nan, 'SMAPE': np.nan
                }
        
        # Find best quantile for each metric
        metrics = ['RMSE', 'MAE', 'MAPE', 'SMAPE']
        for metric in metrics:
            best_quantile = None
            best_value = np.inf
            
            for q_name, q_results in results['quantiles'].items():
                if metric in q_results and not np.isnan(q_results[metric]):
                    if q_results[metric] < best_value:
                        best_value = q_results[metric]
                        best_quantile = q_name
            
            if best_quantile:
                results['summary'][f'best_{metric.lower()}'] = {
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
            for metric in ['RMSE', 'MAE', 'MAPE', 'SMAPE']:
                value = q_results.get(metric, np.nan)
                if not np.isnan(value):
                    row.append(f"{value:.2f}")
                else:
                    row.append("N/A")
            table_data.append(row)
        
        headers = ['Quantile', 'RMSE', 'MAE', 'MAPE (%)', 'SMAPE (%)']
        print(tabulate(table_data, headers=headers, tablefmt='psql', floatfmt='.2f'))
        
        # Print best quantiles
        print(f"\n{'Best Quantiles by Metric:'}")
        print(f"{'='*40}")
        for metric in ['RMSE', 'MAE', 'MAPE', 'SMAPE']:
            best_info = results['summary'].get(f'best_{metric.lower()}', {})
            if best_info:
                print(f"{metric:>8}: {best_info['quantile']} ({best_info['value']:.2f})")
    
    def plot_predictions(self, predictions: TimeSeries, actual: TimeSeries, 
                        dataset_name: str) -> None:
        """Plot predictions vs actual values for key quantiles."""
        print(f"\nGenerating plots for {dataset_name}...")
        
        # Key quantiles to plot
        quantiles_to_plot = [0.2, 0.5, 0.8]
        
        # Create figure
        fig, axes = plt.subplots(len(quantiles_to_plot), 1, figsize=(15, 4*len(quantiles_to_plot)))
        if len(quantiles_to_plot) == 1:
            axes = [axes]
        
        # Convert to pandas for easier plotting
        actual_pd = actual.to_dataframe().iloc[:, 0]
        
        for i, quantile in enumerate(quantiles_to_plot):
            try:
                # Get quantile predictions
                q_pred = predictions.quantile(quantile)
                q_pred_inverse = self.scaler_target.inverse_transform(q_pred)
                q_pred_pd = q_pred_inverse.to_dataframe().iloc[:, 0]
                
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
        """Run complete evaluation pipeline."""
        print("Starting Simple TFT Glucose Model Evaluation")
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
        if model_path and os.path.exists(model_path):
            model = self.load_model(model_path)
        else:
            # Use full feature dataset for training to avoid index issues
            ts_features_full_scaled = self.scaler_features.transform(ts_features)
            ts_features_full_scaled = ts_features_full_scaled.astype(np.float32)
            model = self.create_and_train_model(ts_train_scaled, ts_test_scaled, 
                                              ts_features_full_scaled, epochs)
        
        # Choose evaluation dataset
        if use_holdout:
            eval_ts_scaled = ts_holdout_scaled
            eval_ts_original = ts_holdout
            eval_name = "Holdout"
        else:
            eval_ts_scaled = ts_test_scaled
            eval_ts_original = ts_test
            eval_name = "Test"
        
        # Generate predictions
        # Use full feature dataset for prediction
        ts_features_full_scaled = self.scaler_features.transform(ts_features)
        ts_features_full_scaled = ts_features_full_scaled.astype(np.float32)
        
        predictions = self.generate_predictions(
            model, ts_test_scaled, ts_features_full_scaled, len(eval_ts_scaled)
        )
        
        if predictions is None:
            print("✗ Could not generate predictions. Exiting.")
            return {}
        
        # Evaluate performance
        results = self.evaluate_performance(predictions, eval_ts_original, eval_name)
        
        # Print results
        self.print_results(results)
        
        # Generate plots
        self.plot_predictions(predictions, eval_ts_original, eval_name)
        
        return results

def main():
    """Main function to run the evaluation."""
    parser = argparse.ArgumentParser(description='Simple TFT Glucose Model Performance Evaluation')
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
    evaluator = SimpleTFTEvaluator(data_path=args.data_path)
    
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
            for metric in ['RMSE', 'MAE', 'MAPE', 'SMAPE']:
                best_info = results['summary'].get(f'best_{metric.lower()}', {})
                if best_info:
                    print(f"  {metric}: {best_info['quantile']} (value: {best_info['value']:.2f})")
        else:
            print("Evaluation failed.")
            
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


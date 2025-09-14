#!/usr/bin/env python3
"""
TFT Glucose Model Inference and Performance Evaluation Script

This script loads a trained TFT model and evaluates its performance on glucose prediction
using multiple metrics across different quantiles. Based on the Model and Forecast Notebook.

Usage:
    python tft_inference_evaluation.py [--model_path MODEL_PATH] [--data_path DATA_PATH] [--use_holdout]
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

class TFTGlucoseEvaluator:
    """
    TFT Glucose Model Evaluator
    
    This class handles loading a trained TFT model and evaluating its performance
    on glucose prediction tasks using multiple metrics and quantiles.
    """
    
    def __init__(self, model_path: str = "TFT_Glucose", data_path: str = "t1d_glucose_data.csv"):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to the trained model directory
            data_path: Path to the glucose data CSV file
        """
        self.model_path = model_path
        self.data_path = data_path
        self.model = None
        self.scaler_target = None
        self.scaler_features = None
        self.data = None
        self.ts_target = None
        self.ts_features = None
        
        # Model hyperparameters (from notebook)
        self.QUANTILES = [0.01, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99]
        self.SPLIT = 0.90
        self.INLEN = 35
        self.N_SAMPLES = 100
        self.N_JOBS = 12
        
        # Metrics to calculate
        self.metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'SMAPE': smape
        }
        
    def load_data(self) -> None:
        """Load and preprocess the glucose data."""
        print("Loading and preprocessing data...")
        
        # Load data
        self.data = pd.read_csv(self.data_path)
        self.data.rename(columns={"date_time": "datetime_col"}, inplace=True)
        self.data['datetime_col'] = pd.to_datetime(self.data['datetime_col'])
        
        # Convert data types
        self.data['bolus'] = self.data['bolus'].astype('float32')
        self.data['carbs'] = self.data['carbs'].astype('float32')
        self.data['insulin_on_board'] = self.data['insulin_on_board'].astype('float32')
        
        # Convert float64 columns to float32
        int_columns = self.data.select_dtypes(include=['float64']).columns
        self.data[int_columns] = self.data[int_columns].astype('float32')
        
        # Create time series objects
        self.ts_target = TimeSeries.from_dataframe(
            self.data, 'datetime_col', 'glucose_value', freq='5T'
        )
        
        # Extract features
        feature_cols = [col for col in self.data.columns 
                       if col not in ['glucose_value', 'datetime_col']]
        df_features = self.data[['datetime_col'] + feature_cols]
        self.ts_features = TimeSeries.from_dataframe(df_features, 'datetime_col', freq='5T')
        
        print(f"✓ Data loaded: {len(self.data)} rows, {len(feature_cols)} features")
        print(f"✓ Time range: {self.data['datetime_col'].min()} to {self.data['datetime_col'].max()}")
        
    def split_data(self) -> Tuple[TimeSeries, TimeSeries, TimeSeries, TimeSeries, TimeSeries, TimeSeries]:
        """
        Split data into train, test, and holdout sets.
        
        Returns:
            Tuple of (ts_train, ts_test, ts_holdout, ts_features_train, ts_features_test, ts_features_holdout)
        """
        print("Splitting data into train/test/holdout sets...")
        
        # Calculate split points
        train_size = int(len(self.ts_target) * self.SPLIT)
        split_timestamp = self.data.iloc[train_size]['datetime_col']
        
        # Split target data
        ts_train, ts_temp = self.ts_target.split_after(pd.Timestamp(split_timestamp))
        test_size = int(len(ts_temp) * 0.5)
        split_timestamp_test = self.data.iloc[train_size + test_size]['datetime_col']
        ts_test, ts_holdout = ts_temp.split_after(pd.Timestamp(split_timestamp_test))
        
        # Split feature data
        ts_features_train, ts_features_temp = self.ts_features.split_after(pd.Timestamp(split_timestamp))
        ts_features_test, ts_features_holdout = ts_features_temp.split_after(pd.Timestamp(split_timestamp_test))
        
        print(f"✓ Training set: {len(ts_train)} points ({ts_train.duration})")
        print(f"✓ Test set: {len(ts_test)} points ({ts_test.duration})")
        print(f"✓ Holdout set: {len(ts_holdout)} points ({ts_holdout.duration})")
        
        return ts_train, ts_test, ts_holdout, ts_features_train, ts_features_test, ts_features_holdout
    
    def scale_data(self, ts_train: TimeSeries, ts_test: TimeSeries, ts_holdout: TimeSeries,
                   ts_features_train: TimeSeries, ts_features_test: TimeSeries, 
                   ts_features_holdout: TimeSeries) -> Tuple:
        """
        Scale the data using the same scalers as training.
        
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
    
    def load_model(self) -> None:
        """Load the trained TFT model."""
        print(f"Loading trained model from {self.model_path}...")
        
        try:
            self.model = TFTModel.load_from_checkpoint(self.model_path)
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            print("Creating a new model with default parameters...")
            self.model = TFTModel(
                input_chunk_length=self.INLEN,
                output_chunk_length=1,
                hidden_size=6,
                lstm_layers=3,
                num_attention_heads=2,
                dropout=0.1,
                batch_size=48,
                n_epochs=1,  # Just for inference
                likelihood=QuantileRegression(self.QUANTILES),
                random_state=42,
                add_relative_index=True
            )
            print("✓ New model created (will need training)")
    
    def generate_predictions(self, ts_input: TimeSeries, ts_features: TimeSeries, 
                           n_steps: int) -> TimeSeries:
        """
        Generate predictions using the trained model.
        
        Args:
            ts_input: Input time series for prediction
            ts_features: Feature time series
            n_steps: Number of steps to predict
            
        Returns:
            Predicted time series with quantiles
        """
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
        """
        Evaluate model performance across different quantiles.
        
        Args:
            predictions: Predicted time series with quantiles
            actual: Actual time series
            dataset_name: Name of the dataset (for reporting)
            
        Returns:
            Dictionary with evaluation results
        """
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
        """
        Plot predictions vs actual values for selected quantiles.
        
        Args:
            predictions: Predicted time series
            actual: Actual time series
            dataset_name: Name of the dataset
            quantiles_to_plot: List of quantiles to plot
        """
        print(f"\nGenerating plots for {dataset_name}...")
        
        # Create figure
        fig, axes = plt.subplots(len(quantiles_to_plot), 1, figsize=(15, 4*len(quantiles_to_plot)))
        if len(quantiles_to_plot) == 1:
            axes = [axes]
        
        # Convert to pandas for easier plotting
        actual_pd = actual.pd_series()
        
        for i, quantile in enumerate(quantiles_to_plot):
            try:
                # Get quantile predictions
                q_pred = predictions.quantile(quantile)
                q_pred_inverse = self.scaler_target.inverse_transform(q_pred)
                q_pred_pd = q_pred_inverse.pd_series()
                
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
    
    def run_evaluation(self, use_holdout: bool = True) -> Dict:
        """
        Run complete evaluation pipeline.
        
        Args:
            use_holdout: Whether to evaluate on holdout data (True) or test data (False)
            
        Returns:
            Dictionary with all evaluation results
        """
        print("Starting TFT Glucose Model Evaluation")
        print("="*50)
        
        # Load data
        self.load_data()
        
        # Split data
        ts_train, ts_test, ts_holdout, ts_features_train, ts_features_test, ts_features_holdout = self.split_data()
        
        # Scale data
        (ts_train_scaled, ts_test_scaled, ts_holdout_scaled,
         ts_features_train_scaled, ts_features_test_scaled, ts_features_holdout_scaled) = self.scale_data(
            ts_train, ts_test, ts_holdout, ts_features_train, ts_features_test, ts_features_holdout
        )
        
        # Load model
        self.load_model()
        
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
        predictions = self.generate_predictions(
            ts_test_scaled,  # Use test data as input for prediction
            ts_features_test_scaled,
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
    parser = argparse.ArgumentParser(description='TFT Glucose Model Evaluation')
    parser.add_argument('--model_path', type=str, default='TFT_Glucose',
                       help='Path to the trained model directory')
    parser.add_argument('--data_path', type=str, default='t1d_glucose_data.csv',
                       help='Path to the glucose data CSV file')
    parser.add_argument('--use_holdout', action='store_true', default=True,
                       help='Use holdout data for evaluation (default: True)')
    parser.add_argument('--use_test', action='store_true',
                       help='Use test data for evaluation (overrides --use_holdout)')
    
    args = parser.parse_args()
    
    # Determine which dataset to use
    use_holdout = args.use_holdout and not args.use_test
    
    # Create evaluator
    evaluator = TFTGlucoseEvaluator(
        model_path=args.model_path,
        data_path=args.data_path
    )
    
    # Run evaluation
    try:
        results = evaluator.run_evaluation(use_holdout=use_holdout)
        
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


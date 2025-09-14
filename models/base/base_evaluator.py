"""
Base evaluator class for glucose prediction models.

This module provides an abstract base class for model evaluation
that can be extended by specific model implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import numpy as np
from darts import TimeSeries

from .data_handler import DataHandler
from .metrics_calculator import MetricsCalculator


class BaseGlucoseEvaluator(ABC):
    """
    Abstract base class for glucose prediction model evaluators.
    
    This class provides common evaluation functionality that can be
    extended by specific model implementations.
    """
    
    def __init__(self, data_path: str = "data/t1d_glucose_data.csv"):
        """
        Initialize the base evaluator.
        
        Args:
            data_path: Path to the glucose data CSV file
        """
        self.data_handler = DataHandler(data_path)
        self.metrics_calculator = MetricsCalculator()
        self.model = None
        
        # Common hyperparameters
        self.QUANTILES = [0.01, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99]
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
    
    @abstractmethod
    def generate_predictions(self, ts_input: TimeSeries, ts_features: TimeSeries, 
                           n_steps: int, **kwargs) -> TimeSeries:
        """
        Generate predictions using the model.
        
        This method must be implemented by subclasses.
        
        Args:
            ts_input: Input time series for prediction
            ts_features: Feature time series
            n_steps: Number of steps to predict
            **kwargs: Prediction-specific parameters
            
        Returns:
            Predicted time series
        """
        pass
    
    def load_model(self, model_path: str):
        """
        Load an existing model.
        
        This method can be overridden by subclasses for model-specific loading.
        
        Args:
            model_path: Path to the model
        """
        print(f"Loading model from {model_path}...")
        # Default implementation - subclasses should override
        raise NotImplementedError("Model loading must be implemented by subclasses")
    
    def evaluate_performance(self, predictions: TimeSeries, actual: TimeSeries, 
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
        
        results = self.metrics_calculator.evaluate_quantiles(
            predictions, actual, self.QUANTILES, self.data_handler.scaler_target
        )
        results['dataset'] = dataset_name
        
        return results
    
    def print_results(self, results: Dict) -> None:
        """
        Print evaluation results in a formatted table.
        
        Args:
            results: Dictionary with evaluation results
        """
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
                interpretation = self.metrics_calculator.interpret_metric(metric, best_info['value'])
                print(f"{metric:>8}: {best_info['quantile']} ({best_info['value']:.2f}) - {interpretation}")
    
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
        actual_pd = actual.to_dataframe().iloc[:, 0]
        
        # Both actual and predicted data should now have the same time range
        
        for i, quantile in enumerate(quantiles_to_plot):
            try:
                # Get quantile predictions
                q_pred = predictions.quantile(quantile)
                q_pred_inverse = self.data_handler.scaler_target.inverse_transform(q_pred)
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
                actual_values = actual_pd.values.flatten()
                pred_values = q_pred_inverse.values().flatten()
                rmse_value = np.sqrt(np.mean((actual_values - pred_values) ** 2))
                interpretation = self.metrics_calculator.interpret_metric('RMSE', rmse_value)
                axes[i].text(0.02, 0.98, f'RMSE: {rmse_value:.2f} ({interpretation})', 
                           transform=axes[i].transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
            except Exception as e:
                print(f"Error plotting quantile {quantile}: {e}")
        
        plt.tight_layout()
        plt.savefig(f'results/{dataset_name.lower().replace(" ", "_")}_predictions.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_evaluation(self, model_path: Optional[str] = None, quick_train: bool = False, 
                      epochs: int = 5, use_holdout: bool = True, **kwargs) -> Dict:
        """
        Run complete evaluation pipeline.
        
        Args:
            model_path: Path to existing model
            quick_train: Whether to do quick training
            epochs: Number of epochs for quick training
            use_holdout: Whether to evaluate on holdout data
            **kwargs: Additional model-specific parameters
            
        Returns:
            Dictionary with all evaluation results
        """
        print(f"Starting {self.__class__.__name__} Evaluation")
        print("="*60)
        
        # Load data
        ts_target, ts_features = self.data_handler.load_data()
        
        # Split data
        (ts_train, ts_test, ts_holdout, 
         ts_features_train, ts_features_test, ts_features_holdout) = self.data_handler.split_data(ts_target, ts_features)
        
        # Scale data
        (ts_train_scaled, ts_test_scaled, ts_holdout_scaled,
         ts_features_train_scaled, ts_features_test_scaled, ts_features_holdout_scaled) = self.data_handler.scale_data(
            ts_train, ts_test, ts_holdout, ts_features_train, ts_features_test, ts_features_holdout
        )
        
        # Load or create model
        if model_path and hasattr(self, 'load_model'):
            try:
                self.load_model(model_path)
            except NotImplementedError:
                print("Model loading not implemented, creating new model...")
                self.create_model(epochs=epochs, **kwargs)
        else:
            self.create_model(epochs=epochs, **kwargs)
        
        # Train model if needed
        if quick_train or not model_path:
            # Use full feature dataset for training to avoid index issues
            ts_features_full_scaled = self.data_handler.scaler_features.transform(ts_features)
            ts_features_full_scaled = ts_features_full_scaled.astype(np.float32)
            self.train_model(ts_train_scaled, ts_test_scaled, ts_features_full_scaled, **kwargs)
        
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
        ts_features_full_scaled = self.data_handler.scaler_features.transform(ts_features)
        ts_features_full_scaled = ts_features_full_scaled.astype(np.float32)
        
        # For test data, we need to predict the test period, not extend beyond it
        if not use_holdout:
            # Use data up to the start of test period to predict the test period
            # We need to find the point just before the test period starts
            test_start_time = ts_test_scaled.start_time()
            
            # Get data up to (but not including) the test period
            ts_input_for_prediction = ts_train_scaled
            
            # Generate predictions for the test period length
            predictions = self.generate_predictions(
                ts_input_for_prediction, ts_features_full_scaled, len(eval_ts_scaled), **kwargs
            )
        else:
            # For holdout data, use the test data to predict holdout period
            predictions = self.generate_predictions(
                ts_test_scaled, ts_features_full_scaled, len(eval_ts_scaled), **kwargs
            )
        
        if predictions is None:
            print("✗ Could not generate predictions. Exiting.")
            return {}
        
        # Both actual and predicted data should now have the same time range
        actual_for_plotting = eval_ts_original
        
        # Evaluate performance
        results = self.evaluate_performance(predictions, eval_ts_original, eval_name)
        
        # Print results
        self.print_results(results)
        
        # Generate plots with the correct actual data
        self.plot_predictions(predictions, actual_for_plotting, eval_name)
        
        return results

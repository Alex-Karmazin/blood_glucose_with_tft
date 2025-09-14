"""
Chronos Glucose Model Evaluator.

This module provides Chronos-specific evaluation functionality
extending the base evaluator class.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from darts import TimeSeries
from chronos import BaseChronosPipeline

from ..base.base_evaluator import BaseGlucoseEvaluator


class ChronosGlucoseEvaluator(BaseGlucoseEvaluator):
    """
    Chronos Glucose Model Evaluator.
    
    This class provides Chronos-specific evaluation functionality
    extending the base evaluator class.
    """
    
    def __init__(self, data_path: str = "data/t1d_glucose_data.csv"):
        """
        Initialize the Chronos evaluator.
        
        Args:
            data_path: Path to the glucose data CSV file
        """
        super().__init__(data_path)
        
        # Chronos-specific hyperparameters
        self.model_name = "amazon/chronos-t5-small"  # Default model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        # Chronos doesn't use traditional quantiles, but we can simulate them
        # by using different prediction samples
        self.n_samples = 100  # Number of samples for probabilistic forecasting
    
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
        self.n_samples = kwargs.get('n_samples', self.n_samples)
        
        try:
            self.model = BaseChronosPipeline.from_pretrained(
                self.model_name,
                device_map=self.device,
                torch_dtype=self.torch_dtype,
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
        print("✓ Chronos model ready for inference")
    
    def generate_predictions(self, ts_input: TimeSeries, ts_features: TimeSeries, 
                           n_steps: int, **kwargs) -> TimeSeries:
        """
        Generate predictions using the Chronos model.
        
        Args:
            ts_input: Input time series for prediction
            ts_features: Feature time series (not used by Chronos)
            n_steps: Number of steps to predict
            **kwargs: Prediction-specific parameters
            
        Returns:
            Predicted time series with quantiles
        """
        print(f"Generating Chronos predictions for {n_steps} steps...")
        
        try:
            # Convert Darts TimeSeries to numpy array
            context = ts_input.values().flatten()
            
            # Get quantile levels for prediction
            quantile_levels = kwargs.get('quantile_levels', [0.1, 0.2, 0.5, 0.8, 0.9])
            
            # Generate quantile predictions
            # Use the last part of context as input (Chronos works better with recent data)
            context_length = min(len(context), 64)  # Use last 64 points or all if shorter
            recent_context = context[-context_length:]
            
            quantiles, mean = self.model.predict_quantiles(
                context=torch.tensor(recent_context, dtype=torch.float32),
                prediction_length=n_steps,
                quantile_levels=quantile_levels,
            )
            
            # Store quantiles for later use in evaluation
            self.chronos_quantiles = quantiles[0].cpu().numpy()  # Shape: [n_steps, n_quantiles]
            self.chronos_quantile_levels = quantile_levels
            
            # Create a simple TimeSeries with the median (Q50) for basic functionality
            # The actual quantile evaluation will be handled in the evaluate_performance method
            median_idx = quantile_levels.index(0.5) if 0.5 in quantile_levels else len(quantile_levels) // 2
            median_values = quantiles[0, :, median_idx].cpu().numpy()
            
            # Debug: Print some prediction statistics
            print(f"  - Context length used: {context_length}")
            print(f"  - Prediction range: {median_values.min():.2f} to {median_values.max():.2f}")
            print(f"  - Prediction std: {median_values.std():.2f}")
            print(f"  - Input context range: {recent_context.min():.2f} to {recent_context.max():.2f}")
            
            # Create time index for predictions - align with the actual data time range
            # This fixes the chart alignment issue
            if hasattr(self, '_target_time_index') and self._target_time_index is not None:
                # Use the target time index to align predictions
                # The key insight: we need to find where the input data ends in the full dataset
                # and then use the next n_steps for predictions
                
                # Find the end time of the input data in the full dataset
                input_end_time = ts_input.end_time()
                
                # Find the index of this end time in the full time index
                try:
                    # Find the index where the input data ends
                    input_end_idx = None
                    for i, time in enumerate(self._target_time_index):
                        if time >= input_end_time:
                            input_end_idx = i
                            break
                    
                    if input_end_idx is not None:
                        # Use the next n_steps after the input data ends
                        start_idx = input_end_idx + 1
                        end_idx = start_idx + n_steps
                        
                        # Ensure we don't go beyond the available time index
                        if end_idx <= len(self._target_time_index):
                            time_index = self._target_time_index[start_idx:end_idx]
                        else:
                            # Fallback if we're beyond the available data
                            start_time = ts_input.end_time() + ts_input.freq
                            time_index = [start_time + i * ts_input.freq for i in range(n_steps)]
                    else:
                        # Fallback if we can't find the end time
                        start_time = ts_input.end_time() + ts_input.freq
                        time_index = [start_time + i * ts_input.freq for i in range(n_steps)]
                        
                except Exception as e:
                    print(f"Warning: Error in time index calculation: {e}")
                    # Fallback to original method
                    start_time = ts_input.end_time() + ts_input.freq
                    time_index = [start_time + i * ts_input.freq for i in range(n_steps)]
            else:
                # Fallback to original method
                start_time = ts_input.end_time() + ts_input.freq
                time_index = [start_time + i * ts_input.freq for i in range(n_steps)]
            
            # Create TimeSeries with median values
            import pandas as pd
            df_pred = pd.DataFrame(index=time_index, data={'median': median_values})
            predictions = TimeSeries.from_dataframe(df_pred, freq=ts_input.freq)
            
            print("✓ Chronos predictions generated successfully")
            return predictions
            
        except Exception as e:
            print(f"✗ Error generating Chronos predictions: {e}")
            return None
    
    def load_model(self, model_path: str):
        """
        Load an existing Chronos model.
        
        Args:
            model_path: Path to the Chronos model configuration or model name for pretrained models
        """
        print(f"Loading Chronos model: {model_path}...")
        try:
            # Check if it's a saved model configuration directory
            if model_path and os.path.exists(model_path) and os.path.isdir(model_path):
                config_file_path = os.path.join(model_path, "model_config.json")
                if os.path.exists(config_file_path):
                    # Load from saved configuration
                    import json
                    with open(config_file_path, 'r') as f:
                        config = json.load(f)
                    
                    self.model_name = config['model_name']
                    self.device = config.get('device', self.device)
                    self.torch_dtype = config.get('torch_dtype', self.torch_dtype)
                    
                    print(f"Loading pretrained model: {self.model_name}")
                    import torch
                    torch_dtype = torch.bfloat16 if self.torch_dtype == "bfloat16" else torch.float32
                    self.model = BaseChronosPipeline.from_pretrained(
                        self.model_name,
                        device_map=self.device,
                        dtype=torch_dtype,
                    )
                    print("✓ Chronos model loaded from configuration successfully")
                    return
                else:
                    raise FileNotFoundError(f"Model configuration file not found: {config_file_path}")
            
            # If not a valid directory, check if it's a valid Hugging Face model name
            if model_path and not os.path.exists(model_path):
                # Check if it looks like a valid Hugging Face model name
                # Valid HF model names for Chronos should start with 'amazon/chronos'
                if model_path.startswith('amazon/chronos'):
                    print(f"Loading pretrained model: {model_path}")
                    import torch
                    torch_dtype = torch.bfloat16 if self.torch_dtype == "bfloat16" else torch.float32
                    self.model = BaseChronosPipeline.from_pretrained(
                        model_path,
                        device_map=self.device,
                        dtype=torch_dtype,
                    )
                    self.model_name = model_path
                    print("✓ Chronos pretrained model loaded successfully")
                    return
                else:
                    raise ValueError(f"Invalid model path: {model_path}. Must be either a valid directory with model_config.json or a valid Chronos model name (e.g., 'amazon/chronos-t5-tiny').")
            else:
                raise ValueError(f"Model path does not exist: {model_path}")
            
        except Exception as e:
            print(f"✗ Error loading Chronos model: {e}")
            raise
    
    def run_evaluation(self, model_path: Optional[str] = None, quick_train: bool = False, 
                      epochs: int = 5, use_holdout: bool = True, **kwargs) -> Dict:
        """
        Run complete evaluation pipeline for Chronos models.
        
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
        
        # Store the original time index for proper alignment
        self._target_time_index = ts_target.time_index
        
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
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Creating new model...")
                self.create_model(**kwargs)
        else:
            self.create_model(**kwargs)
        
        # Train model if needed (for Chronos, this is just validation)
        if quick_train or not model_path:
            self.train_model(ts_train_scaled, ts_test_scaled, ts_features_train_scaled, **kwargs)
        
        # Choose evaluation dataset
        if use_holdout:
            eval_ts_scaled = ts_holdout_scaled
            eval_ts_original = ts_holdout
            eval_name = "Holdout"
        else:
            eval_ts_scaled = ts_test_scaled
            eval_ts_original = ts_test
            eval_name = "Test"
        
        # Generate predictions with proper alignment
        # For Chronos, we need to use the right input data for prediction
        if not use_holdout:
            # Use training data to predict test period
            ts_input_for_prediction = ts_train_scaled
        else:
            # Use test data to predict holdout period
            ts_input_for_prediction = ts_test_scaled
        
        # Generate predictions
        predictions = self.generate_predictions(
            ts_input_for_prediction, None, len(eval_ts_scaled), **kwargs
        )
        
        if predictions is None:
            print("✗ Could not generate predictions. Exiting.")
            return {}
        
        # Evaluate performance
        results = self.evaluate_performance(predictions, eval_ts_original, eval_name)
        
        # Print results
        self.print_results(results)
        
        # Generate plots with the correct actual data
        self.plot_predictions(predictions, eval_ts_original, eval_name)
        
        return results
    
    def get_model_info(self) -> Dict:
        """
        Get information about the Chronos model.
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"status": "No model loaded"}
        
        return {
            "model_type": "Chronos (Pretrained Foundation Model)",
            "model_name": self.model_name,
            "device": self.device,
            "torch_dtype": str(self.torch_dtype),
            "n_samples": self.n_samples,
            "quantiles": self.QUANTILES,
            "status": "Model loaded"
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
    
    def evaluate_performance(self, predictions: TimeSeries, actual: TimeSeries, 
                           dataset_name: str) -> Dict:
        """
        Evaluate Chronos model performance across different quantiles.
        
        This method overrides the base class to handle Chronos-specific quantile evaluation.
        
        Args:
            predictions: Predicted time series (not used directly for Chronos)
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
        
        # Get actual values
        actual_values = actual.values().flatten()
        
        # Ensure actual values match the length of Chronos quantiles
        min_length = min(len(actual_values), len(self.chronos_quantiles))
        if len(actual_values) != len(self.chronos_quantiles):
            actual_values = actual_values[:min_length]
            # Also trim the chronos_quantiles to match
            self.chronos_quantiles = self.chronos_quantiles[:min_length]
        
        # Evaluate each quantile using stored Chronos quantiles
        for i, quantile in enumerate(self.QUANTILES):
            try:
                # Find closest quantile level in Chronos output
                closest_idx = min(range(len(self.chronos_quantile_levels)), 
                                key=lambda i: abs(self.chronos_quantile_levels[i] - quantile))
                
                # Get quantile predictions
                pred_values = self.chronos_quantiles[:, closest_idx]
                
                # Calculate metrics
                quantile_results = self.metrics_calculator.calculate_metrics(pred_values, actual_values)
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
    
    def plot_predictions(self, predictions: TimeSeries, actual: TimeSeries, 
                        dataset_name: str, quantiles_to_plot: List[float] = [0.2, 0.5, 0.8]) -> None:
        """
        Plot Chronos predictions vs actual values for selected quantiles.
        
        This method overrides the base class to handle Chronos-specific plotting.
        
        Args:
            predictions: Predicted time series (contains median values, used for time alignment)
            actual: Actual time series
            dataset_name: Name of the dataset
            quantiles_to_plot: List of quantiles to plot
        """
        print(f"\nGenerating plots for {dataset_name}...")
        
        # Create figure
        import matplotlib.pyplot as plt
        import pandas as pd
        fig, axes = plt.subplots(len(quantiles_to_plot), 1, figsize=(15, 4*len(quantiles_to_plot)))
        if len(quantiles_to_plot) == 1:
            axes = [axes]
        
        # Convert to pandas for easier plotting
        actual_pd = actual.to_dataframe().iloc[:, 0]
        
        # Check if we have Chronos quantiles
        if not hasattr(self, 'chronos_quantiles') or self.chronos_quantiles is None:
            print("Warning: No Chronos quantiles found, plotting median only")
            # Fallback to median plotting using the predictions TimeSeries
            pred_inverse = self.data_handler.scaler_target.inverse_transform(predictions)
            pred_pd = pred_inverse.to_dataframe().iloc[:, 0]
            
            axes[0].plot(actual_pd.index, actual_pd.values, label='Actual', linewidth=2, alpha=0.8)
            axes[0].plot(pred_pd.index, pred_pd.values, label='Median Prediction', linewidth=2, alpha=0.8)
            axes[0].set_title(f'{dataset_name} - Median Prediction')
            axes[0].set_ylabel('Glucose (mg/dL)')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Calculate RMSE
            actual_values = actual_pd.values.flatten()
            pred_values = pred_inverse.values().flatten()
            rmse_value = np.sqrt(np.mean((actual_values - pred_values) ** 2))
            axes[0].text(0.02, 0.98, f'RMSE: {rmse_value:.2f}', 
                       transform=axes[0].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            # Use Chronos quantiles with proper time alignment
            for i, quantile in enumerate(quantiles_to_plot):
                try:
                    # Find closest quantile level in Chronos output
                    closest_idx = min(range(len(self.chronos_quantile_levels)), 
                                    key=lambda i: abs(self.chronos_quantile_levels[i] - quantile))
                    closest_quantile = self.chronos_quantile_levels[closest_idx]
                    
                    # Get quantile predictions
                    pred_values = self.chronos_quantiles[:, closest_idx]
                    
                    # Use the predictions TimeSeries time index for proper alignment
                    # This ensures the predictions are aligned with the actual data
                    time_index = predictions.time_index
                    
                    # Create TimeSeries for this quantile and inverse transform
                    df_quantile = pd.DataFrame(index=time_index, data={f'q{int(closest_quantile*100):02d}': pred_values})
                    ts_quantile = TimeSeries.from_dataframe(df_quantile, freq=predictions.freq)
                    q_pred_inverse = self.data_handler.scaler_target.inverse_transform(ts_quantile)
                    q_pred_pd = q_pred_inverse.to_dataframe().iloc[:, 0]
                    
                    # Plot
                    axes[i].plot(actual_pd.index, actual_pd.values, label='Actual', linewidth=2, alpha=0.8)
                    axes[i].plot(q_pred_pd.index, q_pred_pd.values, label=f'Q{int(closest_quantile*100):02d} Prediction', 
                               linewidth=2, alpha=0.8)
                    
                    axes[i].set_title(f'{dataset_name} - Q{int(closest_quantile*100):02d} Quantile (Chronos)')
                    axes[i].set_ylabel('Glucose (mg/dL)')
                    axes[i].legend()
                    axes[i].grid(True, alpha=0.3)
                    
                    # Calculate and display RMSE
                    actual_values = actual_pd.values.flatten()
                    pred_values = q_pred_inverse.values().flatten()
                    rmse_value = np.sqrt(np.mean((actual_values - pred_values) ** 2))
                    axes[i].text(0.02, 0.98, f'RMSE: {rmse_value:.2f}', 
                               transform=axes[i].transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                except Exception as e:
                    print(f"Error plotting quantile {quantile}: {e}")
                    # Plot median as fallback
                    pred_inverse = self.data_handler.scaler_target.inverse_transform(predictions)
                    pred_pd = pred_inverse.to_dataframe().iloc[:, 0]
                    axes[i].plot(actual_pd.index, actual_pd.values, label='Actual', linewidth=2, alpha=0.8)
                    axes[i].plot(pred_pd.index, pred_pd.values, label='Median Prediction', linewidth=2, alpha=0.8)
                    axes[i].set_title(f'{dataset_name} - Median Prediction (Fallback)')
                    axes[i].set_ylabel('Glucose (mg/dL)')
                    axes[i].legend()
                    axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_filename = f"results/{dataset_name.lower()}_predictions_chronos.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"Plot saved as: {plot_filename}")
        plt.show()  # Display the plot on screen
        plt.close()  # Close after displaying

"""
Metrics calculation utilities for glucose prediction models.

This module provides common metrics calculation functionality
that can be used across different model architectures.
"""

import numpy as np
from typing import Dict, List, Tuple
from darts import TimeSeries
from darts.metrics import rmse, mae, mape, smape


class MetricsCalculator:
    """
    Calculates various performance metrics for glucose prediction models.
    """
    
    def __init__(self):
        """Initialize the metrics calculator."""
        self.metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'SMAPE': smape
        }
    
    def calculate_metrics(self, pred_values: np.ndarray, actual_values: np.ndarray) -> Dict[str, float]:
        """
        Calculate all metrics manually to ensure they work.
        
        Args:
            pred_values: Predicted values
            actual_values: Actual values
            
        Returns:
            Dictionary with calculated metrics
        """
        # RMSE
        rmse_val = np.sqrt(np.mean((actual_values - pred_values) ** 2))
        
        # MAE
        mae_val = np.mean(np.abs(actual_values - pred_values))
        
        # MAPE (handle division by zero)
        mape_val = np.mean(np.abs((actual_values - pred_values) / np.maximum(actual_values, 1e-8))) * 100
        
        # SMAPE
        smape_val = np.mean(2 * np.abs(actual_values - pred_values) / 
                          (np.abs(actual_values) + np.abs(pred_values) + 1e-8)) * 100
        
        return {
            'RMSE': float(rmse_val),
            'MAE': float(mae_val),
            'MAPE': float(mape_val),
            'SMAPE': float(smape_val)
        }
    
    def calculate_metrics_from_timeseries(self, predictions: TimeSeries, actual: TimeSeries) -> Dict[str, float]:
        """
        Calculate metrics from TimeSeries objects.
        
        Args:
            predictions: Predicted TimeSeries
            actual: Actual TimeSeries
            
        Returns:
            Dictionary with calculated metrics
        """
        pred_values = predictions.values().flatten()
        actual_values = actual.values().flatten()
        
        return self.calculate_metrics(pred_values, actual_values)
    
    def evaluate_quantiles(self, predictions: TimeSeries, actual: TimeSeries, 
                          quantiles: List[float], scaler_target=None) -> Dict:
        """
        Evaluate model performance across different quantiles.
        
        Args:
            predictions: Predicted TimeSeries with quantiles
            actual: Actual TimeSeries
            quantiles: List of quantiles to evaluate
            scaler_target: Optional scaler to inverse transform predictions
            
        Returns:
            Dictionary with evaluation results
        """
        results = {
            'quantiles': {},
            'summary': {}
        }
        
        # Get actual values
        actual_values = actual.values().flatten()
        
        # Evaluate each quantile
        for quantile in quantiles:
            try:
                # Extract quantile predictions
                q_pred = predictions.quantile(quantile)
                
                # Inverse transform predictions if scaler provided
                if scaler_target is not None:
                    q_pred_inverse = scaler_target.inverse_transform(q_pred)
                else:
                    q_pred_inverse = q_pred
                
                pred_values = q_pred_inverse.values().flatten()
                
                # Calculate metrics
                quantile_results = self.calculate_metrics(pred_values, actual_values)
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
    
    def get_metric_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions of the metrics.
        
        Returns:
            Dictionary with metric descriptions
        """
        return {
            'RMSE': 'Root Mean Square Error - Average magnitude of prediction errors (mg/dL)',
            'MAE': 'Mean Absolute Error - Average absolute difference between predicted and actual values (mg/dL)',
            'MAPE': 'Mean Absolute Percentage Error - Average percentage error relative to actual values (%)',
            'SMAPE': 'Symmetric Mean Absolute Percentage Error - Symmetric version of MAPE, less biased toward low values (%)'
        }
    
    def print_metric_descriptions(self):
        """Print descriptions of all metrics."""
        descriptions = self.get_metric_descriptions()
        print("\nMetric Descriptions:")
        print("=" * 50)
        for metric, description in descriptions.items():
            print(f"{metric}: {description}")
    
    def get_interpretation_guidelines(self) -> Dict[str, Dict[str, str]]:
        """
        Get interpretation guidelines for the metrics.
        
        Returns:
            Dictionary with interpretation guidelines
        """
        return {
            'RMSE': {
                'excellent': '< 20 mg/dL',
                'good': '20-40 mg/dL', 
                'fair': '40-60 mg/dL',
                'poor': '> 60 mg/dL'
            },
            'MAE': {
                'excellent': '< 15 mg/dL',
                'good': '15-30 mg/dL',
                'fair': '30-45 mg/dL', 
                'poor': '> 45 mg/dL'
            },
            'MAPE': {
                'excellent': '< 20%',
                'good': '20-35%',
                'fair': '35-50%',
                'poor': '> 50%'
            },
            'SMAPE': {
                'excellent': '< 20%',
                'good': '20-35%',
                'fair': '35-50%',
                'poor': '> 50%'
            }
        }
    
    def interpret_metric(self, metric_name: str, value: float) -> str:
        """
        Interpret a metric value.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            
        Returns:
            Interpretation string
        """
        guidelines = self.get_interpretation_guidelines()
        
        if metric_name not in guidelines:
            return "Unknown metric"
        
        metric_guidelines = guidelines[metric_name]
        
        try:
            # Parse excellent threshold
            excellent_threshold = float(metric_guidelines['excellent'].split()[0].replace('<', ''))
            if value < excellent_threshold:
                return f"Excellent ({value:.2f})"
            
            # Parse good threshold
            good_threshold = float(metric_guidelines['good'].split('-')[1].split()[0])
            if value < good_threshold:
                return f"Good ({value:.2f})"
            
            # Parse fair threshold
            fair_threshold = float(metric_guidelines['fair'].split('-')[1].split()[0])
            if value < fair_threshold:
                return f"Fair ({value:.2f})"
            else:
                return f"Poor ({value:.2f})"
        except (ValueError, IndexError):
            # Fallback if parsing fails
            return f"Value: {value:.2f}"

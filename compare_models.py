#!/usr/bin/env python3
"""
Model Comparison Script

This script provides a framework for comparing different model architectures
on glucose prediction tasks. It can be extended to include new models.

Usage:
    python compare_models.py [--models MODEL_LIST] [--quick_train] [--epochs EPOCHS]
"""

import argparse
import sys
import os
import json
from datetime import datetime
from typing import Dict, List


class ModelComparison:
    """
    Framework for comparing different model architectures.
    """
    
    def __init__(self, data_path: str = "data/t1d_glucose_data.csv"):
        """
        Initialize the model comparison framework.
        
        Args:
            data_path: Path to the glucose data CSV file
        """
        self.data_path = data_path
        self.results = {}
        # Import models lazily to avoid slow startup
        self._available_models = None
    
    @property
    def available_models(self):
        """Lazy load available models to avoid slow imports."""
        if self._available_models is None:
            # Add the models directory to the path
            sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
            
            # Import here to avoid slow imports when just showing help
            from models.tft_models.tft_evaluator import TFTGlucoseEvaluator
            from models.chronos_models.chronos_evaluator import ChronosGlucoseEvaluator
            
            self._available_models = {
                'tft': TFTGlucoseEvaluator,
                'chronos': ChronosGlucoseEvaluator
                # Add more models here as they are implemented
                # 'lstm': LSTMGlucoseEvaluator,
                # 'transformer': TransformerGlucoseEvaluator,
                # 'gru': GRUGlucoseEvaluator,
            }
        return self._available_models
    
    def run_model_evaluation(self, model_name: str, model_class, 
                           quick_train: bool = False, epochs: int = 5, 
                           use_holdout: bool = True, **kwargs) -> Dict:
        """
        Run evaluation for a specific model.
        
        Args:
            model_name: Name of the model
            model_class: Model class to instantiate
            quick_train: Whether to do quick training
            epochs: Number of epochs for training
            use_holdout: Whether to use holdout data
            **kwargs: Model-specific parameters
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"\n{'='*80}")
        print(f"EVALUATING {model_name.upper()} MODEL")
        print(f"{'='*80}")
        
        try:
            # Create model evaluator
            evaluator = model_class(self.data_path)
            
            # Run evaluation
            results = evaluator.run_evaluation(
                model_path=None,  # Always train new model for comparison
                quick_train=quick_train,
                epochs=epochs,
                use_holdout=use_holdout,
                **kwargs
            )
            
            if results:
                # Add model info to results
                results['model_name'] = model_name
                results['model_type'] = evaluator.__class__.__name__
                results['timestamp'] = datetime.now().isoformat()
                
                # Get model info if available
                if hasattr(evaluator, 'get_model_info'):
                    results['model_info'] = evaluator.get_model_info()
                
                return results
            else:
                print(f"✗ {model_name} evaluation failed")
                return None
                
        except Exception as e:
            print(f"✗ Error evaluating {model_name}: {e}")
            return None
    
    def compare_models(self, model_names: List[str], quick_train: bool = False, 
                      epochs: int = 5, use_holdout: bool = True, **kwargs) -> Dict:
        """
        Compare multiple models.
        
        Args:
            model_names: List of model names to compare
            quick_train: Whether to do quick training
            epochs: Number of epochs for training
            use_holdout: Whether to use holdout data
            **kwargs: Model-specific parameters
            
        Returns:
            Dictionary with comparison results
        """
        print("Starting Model Comparison")
        print("="*50)
        
        comparison_results = {
            'timestamp': datetime.now().isoformat(),
            'models': {},
            'summary': {}
        }
        
        # Run evaluation for each model
        for model_name in model_names:
            if model_name not in self.available_models:
                print(f"✗ Model '{model_name}' not available. Available models: {list(self.available_models.keys())}")
                continue
            
            model_class = self.available_models[model_name]
            results = self.run_model_evaluation(
                model_name, model_class, quick_train, epochs, use_holdout, **kwargs
            )
            
            if results:
                comparison_results['models'][model_name] = results
        
        # Generate summary comparison
        self._generate_summary(comparison_results)
        
        # Save results
        self._save_results(comparison_results)
        
        return comparison_results
    
    def _generate_summary(self, results: Dict):
        """Generate summary comparison of models."""
        print(f"\n{'='*80}")
        print("MODEL COMPARISON SUMMARY")
        print(f"{'='*80}")
        
        if not results['models']:
            print("No models to compare.")
            return
        
        # Create comparison table
        from tabulate import tabulate
        
        table_data = []
        headers = ['Model', 'Best RMSE', 'Best MAE', 'Best MAPE', 'Best SMAPE']
        
        for model_name, model_results in results['models'].items():
            row = [model_name.upper()]
            
            for metric in ['RMSE', 'MAE', 'MAPE', 'SMAPE']:
                best_info = model_results['summary'].get(f'best_{metric.lower()}', {})
                if best_info:
                    row.append(f"{best_info['value']:.2f}")
                else:
                    row.append("N/A")
            
            table_data.append(row)
        
        print(tabulate(table_data, headers=headers, tablefmt='psql', floatfmt='.2f'))
        
        # Find best model for each metric
        print(f"\n{'Best Models by Metric:'}")
        print(f"{'='*40}")
        
        for metric in ['RMSE', 'MAE', 'MAPE', 'SMAPE']:
            best_model = None
            best_value = float('inf')
            
            for model_name, model_results in results['models'].items():
                best_info = model_results['summary'].get(f'best_{metric.lower()}', {})
                if best_info and best_info['value'] < best_value:
                    best_value = best_info['value']
                    best_model = model_name
            
            if best_model:
                print(f"{metric:>8}: {best_model.upper()} ({best_value:.2f})")
        
        # Store summary in results
        results['summary'] = {
            'best_models': {
                metric: {
                    'model': best_model,
                    'value': best_value
                } for metric, best_model, best_value in [
                    (metric, 
                     min(results['models'].items(), 
                         key=lambda x: x[1]['summary'].get(f'best_{metric.lower()}', {}).get('value', float('inf')))[0],
                     min(results['models'].items(), 
                         key=lambda x: x[1]['summary'].get(f'best_{metric.lower()}', {}).get('value', float('inf')))[1]['summary'].get(f'best_{metric.lower()}', {}).get('value', float('inf')))
                    for metric in ['RMSE', 'MAE', 'MAPE', 'SMAPE']
                ]
            }
        }
    
    def _save_results(self, results: Dict):
        """Save comparison results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/model_comparison_{timestamp}.json"
        
        os.makedirs("results", exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✓ Results saved to: {filename}")
    
    def list_available_models(self):
        """List all available models."""
        print("Available Models:")
        print("="*30)
        for model_name, model_class in self.available_models.items():
            print(f"  {model_name}: {model_class.__name__}")


def main():
    """Main function to run model comparison."""
    parser = argparse.ArgumentParser(description='Model Comparison for Glucose Prediction')
    parser.add_argument('--models', nargs='+', default=['tft'],
                       help='List of models to compare (default: tft)')
    parser.add_argument('--quick_train', action='store_true',
                       help='Do quick training with few epochs')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of epochs for quick training (default: 5)')
    parser.add_argument('--data_path', type=str, default='data/t1d_glucose_data.csv',
                       help='Path to the glucose data CSV file')
    parser.add_argument('--use_test', action='store_true',
                       help='Use test data instead of holdout data for evaluation')
    parser.add_argument('--list_models', action='store_true',
                       help='List available models and exit')
    
    # TFT-specific parameters
    parser.add_argument('--tft_hidden_size', type=int, default=6,
                       help='Hidden size for TFT model (default: 6)')
    parser.add_argument('--tft_lstm_layers', type=int, default=3,
                       help='Number of LSTM layers for TFT (default: 3)')
    parser.add_argument('--tft_attention_heads', type=int, default=2,
                       help='Number of attention heads for TFT (default: 2)')
    parser.add_argument('--tft_batch_size', type=int, default=48,
                       help='Batch size for TFT training (default: 48)')
    parser.add_argument('--tft_learning_rate', type=float, default=0.0010223,
                       help='Learning rate for TFT (default: 0.0010223)')
    
    # Chronos-specific parameters
    parser.add_argument('--chronos_model_name', type=str, default='amazon/chronos-t5-small',
                       help='Chronos model name (default: amazon/chronos-t5-small)')
    parser.add_argument('--chronos_device', type=str, default='auto',
                       help='Device for Chronos (cuda, cpu, or auto)')
    parser.add_argument('--chronos_torch_dtype', type=str, default='auto',
                       help='Torch dtype for Chronos (bfloat16, float32, or auto)')
    parser.add_argument('--chronos_n_samples', type=int, default=100,
                       help='Number of samples for Chronos (default: 100)')
    
    args = parser.parse_args()
    
    # Create comparison framework
    comparison = ModelComparison(data_path=args.data_path)
    
    # List models if requested
    if args.list_models:
        comparison.list_available_models()
        return
    
    # Determine which dataset to use
    use_holdout = not args.use_test
    
    # Model-specific parameters
    model_params = {
        'tft': {
            'hidden_size': args.tft_hidden_size,
            'lstm_layers': args.tft_lstm_layers,
            'num_attention_heads': args.tft_attention_heads,
            'batch_size': args.tft_batch_size,
            'learning_rate': args.tft_learning_rate
        },
        'chronos': {
            'model_name': args.chronos_model_name,
            'device': args.chronos_device,
            'torch_dtype': args.chronos_torch_dtype,
            'n_samples': args.chronos_n_samples
        }
        # Add parameters for other models here
    }
    
    # Run comparison
    try:
        results = comparison.compare_models(
            model_names=args.models,
            quick_train=args.quick_train,
            epochs=args.epochs,
            use_holdout=use_holdout,
            **model_params.get(args.models[0] if args.models else 'tft', {})  # Pass params for first model
        )
        
        if results and results['models']:
            print(f"\n{'='*80}")
            print("MODEL COMPARISON COMPLETED SUCCESSFULLY")
            print(f"{'='*80}")
            print(f"Compared {len(results['models'])} models")
            print(f"Results saved to results/ directory")
        else:
            print("Model comparison failed.")
            
    except Exception as e:
        print(f"Error during model comparison: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

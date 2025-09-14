#!/usr/bin/env python3
"""
Chronos Glucose Model Evaluation Script

This script evaluates Chronos model performance using the refactored architecture.
It provides a clean interface for evaluating Chronos models on glucose prediction.

Usage:
    python evaluate_chronos.py [--model_name MODEL_NAME] [--device DEVICE]
"""

import argparse
import sys
import os


def main():
    """Main function to run Chronos evaluation."""
    # Add the models directory to the path
    sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
    
    # Import here to avoid slow imports when just showing help
    from models.chronos_models.chronos_evaluator import ChronosGlucoseEvaluator
    
    parser = argparse.ArgumentParser(description='Chronos Glucose Model Evaluation')
    parser.add_argument('--model_name', type=str, default='amazon/chronos-t5-small',
                       help='Chronos model name (default: amazon/chronos-t5-small)')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to saved model configuration (optional)')
    parser.add_argument('--data_path', type=str, default='data/t1d_glucose_data.csv',
                       help='Path to the glucose data CSV file')
    parser.add_argument('--use_test', action='store_true',
                       help='Use test data instead of holdout data for evaluation')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda, cpu, or auto)')
    parser.add_argument('--torch_dtype', type=str, default='auto',
                       help='Torch dtype (bfloat16, float32, or auto)')
    parser.add_argument('--n_samples', type=int, default=100,
                       help='Number of samples for probabilistic forecasting (default: 100)')
    parser.add_argument('--list_models', action='store_true',
                       help='List available Chronos models and exit')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ChronosGlucoseEvaluator(data_path=args.data_path)
    
    # List models if requested
    if args.list_models:
        evaluator.print_available_models()
        return
    
    # Determine device and dtype
    import torch
    if args.device == 'auto':
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    if args.torch_dtype == 'auto':
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    else:
        torch_dtype = getattr(torch, args.torch_dtype)
    
    # Model parameters
    model_params = {
        'model_name': args.model_name,
        'device': device,
        'torch_dtype': torch_dtype,
        'n_samples': args.n_samples
    }
    
    # Determine which dataset to use
    use_holdout = not args.use_test
    
    # Run evaluation
    try:
        results = evaluator.run_evaluation(
            model_path=args.model_path,  # Use saved model config if provided
            quick_train=False,  # No training needed for pretrained models
            epochs=1,  # Not used for pretrained models
            use_holdout=use_holdout,
            **model_params
        )
        
        if results:
            print(f"\n{'='*80}")
            print("CHRONOS EVALUATION COMPLETED SUCCESSFULLY")
            print(f"{'='*80}")
            print(f"Results saved and plots generated.")
            print(f"Best performing quantiles:")
            for metric in ['RMSE', 'MAE', 'MAPE', 'SMAPE']:
                best_info = results['summary'].get(f'best_{metric.lower()}', {})
                if best_info:
                    print(f"  {metric}: {best_info['quantile']} (value: {best_info['value']:.2f})")
            
            # Print model info
            evaluator.print_model_info()
        else:
            print("Evaluation failed.")
            
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

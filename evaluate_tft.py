#!/usr/bin/env python3
"""
TFT Glucose Model Evaluation Script

This script evaluates TFT model performance using the refactored architecture.
It provides a clean interface for evaluating TFT models on glucose prediction.

Usage:
    python evaluate_tft.py [--model_path MODEL_PATH] [--quick_train] [--epochs EPOCHS]
"""

import argparse
import sys
import os


def main():
    """Main function to run TFT evaluation."""
    # Add the models directory to the path
    sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
    
    # Import here to avoid slow imports when just showing help
    from models.tft_models.tft_evaluator import TFTGlucoseEvaluator
    
    parser = argparse.ArgumentParser(description='TFT Glucose Model Evaluation')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to the trained model directory')
    parser.add_argument('--quick_train', action='store_true',
                       help='Do quick training with few epochs')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of epochs for quick training (default: 5)')
    parser.add_argument('--data_path', type=str, default='data/t1d_glucose_data.csv',
                       help='Path to the glucose data CSV file')
    parser.add_argument('--use_test', action='store_true',
                       help='Use test data instead of holdout data for evaluation')
    parser.add_argument('--hidden_size', type=int, default=6,
                       help='Hidden size for TFT model (default: 6)')
    parser.add_argument('--lstm_layers', type=int, default=3,
                       help='Number of LSTM layers (default: 3)')
    parser.add_argument('--attention_heads', type=int, default=2,
                       help='Number of attention heads (default: 2)')
    parser.add_argument('--batch_size', type=int, default=48,
                       help='Batch size for training (default: 48)')
    parser.add_argument('--learning_rate', type=float, default=0.0010223,
                       help='Learning rate (default: 0.0010223)')
    
    args = parser.parse_args()
    
    # Determine which dataset to use
    use_holdout = not args.use_test
    
    # Create evaluator
    evaluator = TFTGlucoseEvaluator(data_path=args.data_path)
    
    # Model parameters
    model_params = {
        'hidden_size': args.hidden_size,
        'lstm_layers': args.lstm_layers,
        'num_attention_heads': args.attention_heads,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate
    }
    
    # Run evaluation
    try:
        results = evaluator.run_evaluation(
            model_path=args.model_path,
            quick_train=args.quick_train,
            epochs=args.epochs,
            use_holdout=use_holdout,
            **model_params
        )
        
        if results:
            print(f"\n{'='*80}")
            print("TFT EVALUATION COMPLETED SUCCESSFULLY")
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

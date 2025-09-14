#!/usr/bin/env python3
"""
TFT Glucose Model Training Script

This script trains a TFT model using the refactored architecture.
It provides a clean interface for training TFT models on glucose prediction.

Usage:
    python train_tft.py [--epochs EPOCHS] [--model_name MODEL_NAME]
"""

import argparse
import sys
import os

# Add the models directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from models.tft_models.tft_trainer import TFTGlucoseTrainer


def main():
    """Main function to run TFT training."""
    parser = argparse.ArgumentParser(description='TFT Glucose Model Training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--model_name', type=str, default='TFT_Glucose',
                       help='Name for the trained model (default: TFT_Glucose)')
    parser.add_argument('--data_path', type=str, default='data/t1d_glucose_data.csv',
                       help='Path to the glucose data CSV file')
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
    parser.add_argument('--input_chunk_length', type=int, default=35,
                       help='Input chunk length (default: 35)')
    parser.add_argument('--output_chunk_length', type=int, default=1,
                       help='Output chunk length (default: 1)')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate (default: 0.1)')
    parser.add_argument('--log_tensorboard', action='store_true', default=True,
                       help='Enable TensorBoard logging (default: True)')
    parser.add_argument('--save_checkpoints', action='store_true', default=True,
                       help='Save model checkpoints (default: True)')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = TFTGlucoseTrainer(data_path=args.data_path)
    
    # Model parameters
    model_params = {
        'hidden_size': args.hidden_size,
        'lstm_layers': args.lstm_layers,
        'num_attention_heads': args.attention_heads,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'input_chunk_length': args.input_chunk_length,
        'output_chunk_length': args.output_chunk_length,
        'dropout': args.dropout,
        'log_tensorboard': args.log_tensorboard,
        'save_checkpoints': args.save_checkpoints
    }
    
    # Run training
    try:
        model_path = trainer.run_training(
            epochs=args.epochs,
            model_name=args.model_name,
            **model_params
        )
        
        print(f"\n{'='*80}")
        print("TFT TRAINING COMPLETED SUCCESSFULLY")
        print(f"{'='*80}")
        print(f"Model saved to: {model_path}")
        print(f"You can now run evaluation using:")
        print(f"  python evaluate_tft.py --model_path {model_path}")
        
        # Print model info
        trainer.print_model_info()
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

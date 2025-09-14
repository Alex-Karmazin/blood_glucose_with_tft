# Glucose Prediction Models

A modular framework for training and evaluating different neural network architectures on blood glucose prediction tasks. This project is based on the work described in [Brandon Harris's article](https://brandonharris.io/Transforming%20Diabetes%20with%20AI%20-%20Temporal%20Fusion%20Transformers%20and%20Blood%20Glucose%20Data/) and provides a clean, extensible architecture for comparing different models.

## 🏗️ Project Structure

```
TFT_glucose/
├── models/                          # Model implementations
│   ├── base/                       # Base classes and utilities
│   │   ├── __init__.py
│   │   ├── base_evaluator.py       # Abstract base evaluator
│   │   ├── base_trainer.py         # Abstract base trainer
│   │   ├── data_handler.py         # Data loading and preprocessing
│   │   └── metrics_calculator.py   # Performance metrics
│   ├── tft_models/                 # TFT-specific implementations
│   │   ├── __init__.py
│   │   ├── tft_evaluator.py        # TFT evaluator
│   │   ├── tft_trainer.py          # TFT trainer
│   │   └── [legacy scripts]        # Original TFT scripts
│   └── chronos_models/             # Chronos-specific implementations
│       ├── __init__.py
│       ├── chronos_evaluator.py    # Chronos evaluator
│       └── chronos_trainer.py      # Chronos trainer
├── data/                           # Data files
│   ├── t1d_glucose_data.csv       # Main dataset
│   └── data_prep/                 # Data preparation notebooks
├── results/                        # Output files
│   ├── *.png                      # Prediction plots
│   └── *.json                     # Comparison results
├── evaluate_tft.py                # TFT evaluation script
├── train_tft.py                   # TFT training script
├── evaluate_chronos.py            # Chronos evaluation script
├── compare_models.py              # Model comparison framework
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
uv sync
```

### 2. Run Model Evaluation

**TFT Model:**

```bash
# Quick evaluation with training (5 epochs)
uv run python evaluate_tft.py --quick_train --epochs 5

# Use existing trained model
uv run python evaluate_tft.py --model_path models/TFT_Glucose

# Evaluate on test data instead of holdout
uv run python evaluate_tft.py --quick_train --use_test
```

**Chronos Model:**

```bash
# Evaluate with default model (chronos-t5-small)
uv run python evaluate_chronos.py

# Use different Chronos model
uv run python evaluate_chronos.py --model_name amazon/chronos-t5-base

# List available Chronos models
uv run python evaluate_chronos.py --list_models

# Evaluate on test data
uv run python evaluate_chronos.py --use_test
```

### 3. Train TFT Model

```bash
# Train with default parameters (100 epochs)
uv run python train_tft.py

# Train with custom parameters
uv run python train_tft.py --epochs 50 --hidden_size 8 --lstm_layers 4
```

### 4. Compare Models

```bash
# Compare TFT and Chronos models
uv run python compare_models.py --models tft chronos --quick_train

# Compare only TFT model
uv run python compare_models.py --models tft --quick_train

# Compare only Chronos model
uv run python compare_models.py --models chronos

# List available models
uv run python compare_models.py --list_models

# Use different Chronos model in comparison
uv run python compare_models.py --models tft chronos --chronos_model_name amazon/chronos-t5-base
```

## 📊 Performance Metrics

The framework evaluates models using multiple metrics:

### RMSE (Root Mean Square Error)

- **Range**: 0 to ∞ (lower is better)
- **Interpretation**: Average magnitude of prediction errors
- **Typical values**: 10-50 mg/dL for glucose prediction

### MAE (Mean Absolute Error)

- **Range**: 0 to ∞ (lower is better)
- **Interpretation**: Average absolute difference between predicted and actual values
- **Less sensitive to outliers than RMSE**

### MAPE (Mean Absolute Percentage Error)

- **Range**: 0% to ∞% (lower is better)
- **Interpretation**: Average percentage error relative to actual values

### SMAPE (Symmetric Mean Absolute Percentage Error)

- **Range**: 0% to 200% (lower is better)
- **Interpretation**: Symmetric version of MAPE, less biased toward low values

## 🎯 Understanding Quantiles

The models predict multiple quantiles, each representing different confidence levels:

- **Q01 (1%)**: Very conservative prediction (low glucose values)
- **Q10 (10%)**: Conservative prediction
- **Q20 (20%)**: Lower confidence bound
- **Q50 (50%)**: Median prediction (most likely value)
- **Q80 (80%)**: Upper confidence bound
- **Q90 (90%)**: Optimistic prediction
- **Q99 (99%)**: Very optimistic prediction (high glucose values)

### Key Insight

Different quantiles may perform better in different physiological states:

- **Sleep periods**: Higher quantiles (Q80-Q90) often perform better
- **Active periods**: Lower quantiles (Q20-Q50) often perform better
- **Meal times**: May require switching between quantiles

## 🏛️ Architecture

### Base Classes

The framework uses abstract base classes to ensure consistency across different model implementations:

#### `BaseGlucoseEvaluator`

- Abstract base class for model evaluation
- Provides common evaluation functionality
- Must be extended by specific model implementations

#### `BaseGlucoseTrainer`

- Abstract base class for model training
- Provides common training functionality
- Must be extended by specific model implementations

#### `DataHandler`

- Handles data loading, preprocessing, and splitting
- Provides consistent data interface across models
- Supports train/test/holdout splits

#### `MetricsCalculator`

- Calculates performance metrics
- Provides metric interpretation guidelines
- Supports quantile-based evaluation

### TFT Implementation

The TFT (Temporal Fusion Transformer) implementation includes:

- **TFTGlucoseEvaluator**: TFT-specific evaluation functionality
- **TFTGlucoseTrainer**: TFT-specific training functionality
- **Hyperparameters**: Optimized based on Brandon Harris's work

### Chronos Implementation

The Chronos implementation includes:

- **ChronosGlucoseEvaluator**: Chronos-specific evaluation functionality
- **ChronosGlucoseTrainer**: Chronos-specific training functionality
- **Pretrained Models**: Uses foundation models from [Amazon's Chronos repository](https://github.com/amazon-science/chronos-forecasting)
- **Zero-shot Forecasting**: No training required, uses pretrained weights

## 🔧 Adding New Models

To add a new model architecture:

1. **Create model directory**:

   ```bash
   mkdir models/your_model
   ```

2. **Implement evaluator**:

   ```python
   # models/your_model/your_evaluator.py
   from ..base.base_evaluator import BaseGlucoseEvaluator

   class YourModelEvaluator(BaseGlucoseEvaluator):
       def create_model(self, **kwargs):
           # Implement model creation
           pass

       def train_model(self, ts_train_scaled, ts_test_scaled, ts_features_scaled, **kwargs):
           # Implement training
           pass

       def generate_predictions(self, ts_input, ts_features, n_steps, **kwargs):
           # Implement prediction generation
           pass
   ```

3. **Implement trainer**:

   ```python
   # models/your_model/your_trainer.py
   from ..base.base_trainer import BaseGlucoseTrainer

   class YourModelTrainer(BaseGlucoseTrainer):
       def create_model(self, **kwargs):
           # Implement model creation
           pass

       def train_model(self, ts_train_scaled, ts_test_scaled, ts_features_scaled, **kwargs):
           # Implement training
           pass
   ```

4. **Add to comparison framework**:

   ```python
   # In compare_models.py
   from models.your_model.your_evaluator import YourModelEvaluator

   self.available_models = {
       'tft': TFTGlucoseEvaluator,
       'chronos': ChronosGlucoseEvaluator,
       'your_model': YourModelEvaluator,  # Add your model here
   }
   ```

## 📈 Example Results

Based on model evaluation, here's what you might expect:

### TFT Results

```
+------------+--------+-------+------------+-------------+
| Quantile   |   RMSE |   MAE |   MAPE (%) |   SMAPE (%) |
|------------+--------+-------+------------+-------------+
| Q01        | 104.14 | 92.68 |      63.35 |       95.57 |
| Q10        |  89.16 | 75.38 |      48.95 |       68.84 |
| Q20        |  78.76 | 63.02 |      38.79 |       52.65 |
| Q50        |  48.71 | 44.11 |      37.02 |       33.37 |
| Q80        |  57.78 | 49.21 |      49.99 |       35.91 |
| Q90        |  69.20 | 54.95 |      58.44 |       38.58 |
| Q99        |  87.29 | 73.06 |      75.30 |       46.81 |
+------------+--------+-------+------------+-------------+
```

**Best performing quantile**: Q50 (50th percentile) across all metrics

### Chronos Results

```
+------------+--------+--------+------------+-------------+
| Quantile   |   RMSE |    MAE |   MAPE (%) |   SMAPE (%) |
|------------+--------+--------+------------+-------------|
| Q01        | 144.40 | 136.38 |      99.87 |      199.48 |
| Q10        | 144.40 | 136.38 |      99.87 |      199.48 |
| Q20        | 144.37 | 136.34 |      99.84 |      199.37 |
| Q50        | 144.28 | 136.26 |      99.78 |      199.12 |
| Q80        | 144.17 | 136.16 |      99.70 |      198.81 |
| Q90        | 144.11 | 136.10 |      99.66 |      198.66 |
| Q99        | 144.11 | 136.10 |      99.66 |      198.66 |
+------------+--------+--------+------------+-------------+
```

**Best performing quantile**: Q90 (90th percentile) across all metrics

### Model Comparison

```
+---------+-------------+------------+-------------+--------------+
| Model   |   Best RMSE |   Best MAE |   Best MAPE |   Best SMAPE |
|---------+-------------+------------+-------------+--------------|
| TFT     |       48.71 |      44.11 |       37.02 |        33.37 |
| CHRONOS |      144.11 |     136.09 |       99.65 |       198.59 |
+---------+-------------+------------+-------------+--------------+
```

**Winner**: TFT model significantly outperforms Chronos on this glucose prediction task.

## 🛠️ Advanced Usage

### Custom Hyperparameters

```bash
# TFT with custom parameters
uv run python evaluate_tft.py --quick_train \
    --hidden_size 8 \
    --lstm_layers 4 \
    --attention_heads 4 \
    --batch_size 64 \
    --learning_rate 0.001
```

### Model Comparison

```bash
# Compare TFT and Chronos models
uv run python compare_models.py --models tft chronos --quick_train

# Compare with different Chronos model
uv run python compare_models.py --models tft chronos --chronos_model_name amazon/chronos-t5-base
```

### Data Customization

```bash
# Use custom data path
uv run python evaluate_tft.py --data_path your_data.csv --quick_train
```

## 📁 Data Requirements

The framework expects a CSV file with the following columns:

- `date_time`: Timestamp (datetime format)
- `glucose_value`: Target variable (float)
- `carbs`: Carbohydrate intake (float)
- `bolus`: Insulin bolus (float)
- `insulin_on_board`: Calculated insulin remaining (float)
- `glucose_trend_20`: 20-minute glucose trend (float)
- `last_delta`: Immediate glucose change (float)

## 🔍 Troubleshooting

### Common Issues

1. **"Invalid past_covariates" error**:

   - The framework automatically handles this by using the full feature dataset
   - This is normal for autoregressive models

2. **Memory issues with large datasets**:

   - Reduce batch size in model parameters
   - Use fewer epochs for quick evaluation

3. **Model loading errors**:
   - Ensure the model path exists and contains valid checkpoints
   - Use `--quick_train` to train a new model instead

### Performance Tips

1. **For quick evaluation**: Use `--epochs 1` or `--epochs 5`
2. **For production training**: Use `--epochs 100` or more
3. **For different datasets**: Adjust the split ratio in `DataHandler`

## 🎯 Next Steps

1. **Run evaluation**: Start with `uv run python evaluate_tft.py --quick_train`
2. **Analyze results**: Look at which quantiles perform best for different time periods
3. **Implement quantile switching**: Based on physiological states (sleep, meals, etc.)
4. **Add new models**: Extend the framework with LSTM, GRU, Transformer, etc.
5. **Fine-tune hyperparameters**: Use the hyperparameter tuning notebook for optimization

## 📚 References

- [Original Article](https://brandonharris.io/Transforming%20Diabetes%20with%20AI%20-%20Temporal%20Fusion%20Transformers%20and%20Blood%20Glucose%20Data/)
- [Darts Documentation](https://unit8co.github.io/darts/)
- [TFT Paper](https://arxiv.org/abs/1912.09363)
- [Chronos Repository](https://github.com/amazon-science/chronos-forecasting)
- [Chronos Paper](https://arxiv.org/abs/2403.07815)

## 🤝 Contributing

This framework is designed to be extensible. To add new models:

1. Follow the base class interfaces
2. Implement the required abstract methods
3. Add your model to the comparison framework
4. Test with the existing data
5. Document your model's specific parameters

The modular architecture makes it easy to add new neural network architectures while maintaining consistency in evaluation and comparison.

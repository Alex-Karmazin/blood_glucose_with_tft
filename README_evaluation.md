# TFT Glucose Model Evaluation Scripts

This repository contains Python scripts for training and evaluating Temporal Fusion Transformer (TFT) models for blood glucose prediction, based on the work described in [Brandon Harris's article](https://brandonharris.io/Transforming%20Diabetes%20with%20AI%20-%20Temporal%20Fusion%20Transformers%20and%20Blood%20Glucose%20Data/).

## Scripts Overview

### 1. `tft_glucose_evaluation.py` (Main Evaluation Script)

**Purpose**: Comprehensive evaluation of TFT model performance with multiple metrics and quantiles.

**Features**:

- Loads pre-trained models or trains new ones
- Evaluates performance across 7 quantiles (1%, 10%, 20%, 50%, 80%, 90%, 99%)
- Calculates multiple metrics: RMSE, MAE, MAPE, SMAPE
- Generates prediction plots
- Works with both test and holdout datasets

**Usage**:

```bash
# Quick evaluation with training (5 epochs)
uv run python tft_glucose_evaluation.py --quick_train --epochs 5

# Use existing trained model
uv run python tft_glucose_evaluation.py --model_path TFT_Glucose

# Evaluate on test data instead of holdout
uv run python tft_glucose_evaluation.py --quick_train --use_test

# Custom data path
uv run python tft_glucose_evaluation.py --data_path your_data.csv --quick_train
```

### 2. `train_tft_model.py` (Training Script)

**Purpose**: Train a TFT model from scratch with configurable parameters.

**Usage**:

```bash
# Train with default parameters (100 epochs)
uv run python train_tft_model.py

# Train with custom epochs
uv run python train_tft_model.py --epochs 50

# Custom model name
uv run python train_tft_model.py --model_name MyTFTModel --epochs 100
```

### 3. `simple_tft_evaluation.py` (Simplified Evaluation)

**Purpose**: Streamlined evaluation script with core functionality.

**Usage**:

```bash
# Quick evaluation
uv run python simple_tft_evaluation.py --quick_train --epochs 1
```

## Performance Metrics Explained

### RMSE (Root Mean Square Error)

- **Range**: 0 to ∞ (lower is better)
- **Interpretation**: Average magnitude of prediction errors
- **Typical values**: 10-50 mg/dL for glucose prediction
- **Example**: RMSE of 48.71 means average error of ~49 mg/dL

### MAE (Mean Absolute Error)

- **Range**: 0 to ∞ (lower is better)
- **Interpretation**: Average absolute difference between predicted and actual values
- **Less sensitive to outliers than RMSE**
- **Example**: MAE of 44.11 means average absolute error of ~44 mg/dL

### MAPE (Mean Absolute Percentage Error)

- **Range**: 0% to ∞% (lower is better)
- **Interpretation**: Average percentage error relative to actual values
- **Example**: MAPE of 37.02% means average error is 37% of actual glucose value

### SMAPE (Symmetric Mean Absolute Percentage Error)

- **Range**: 0% to 200% (lower is better)
- **Interpretation**: Symmetric version of MAPE, less biased toward low values
- **Example**: SMAPE of 33.37% means symmetric average error of 33.37%

## Understanding Quantiles

The TFT model predicts multiple quantiles, each representing different confidence levels:

- **Q01 (1%)**: Very conservative prediction (low glucose values)
- **Q10 (10%)**: Conservative prediction
- **Q20 (20%)**: Lower confidence bound
- **Q50 (50%)**: Median prediction (most likely value)
- **Q80 (80%)**: Upper confidence bound
- **Q90 (90%)**: Optimistic prediction
- **Q99 (99%)**: Very optimistic prediction (high glucose values)

### Key Insight from Brandon Harris's Work

Different quantiles may perform better in different physiological states:

- **Sleep periods**: Higher quantiles (Q80-Q90) often perform better
- **Active periods**: Lower quantiles (Q20-Q50) often perform better
- **Meal times**: May require switching between quantiles

## Example Results

Based on the evaluation, here's what you might expect:

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

- RMSE: 48.71 mg/dL
- MAE: 44.11 mg/dL
- MAPE: 37.02%
- SMAPE: 33.37%

## Data Requirements

The scripts expect a CSV file with the following columns:

- `date_time`: Timestamp (datetime format)
- `glucose_value`: Target variable (float)
- `carbs`: Carbohydrate intake (float)
- `bolus`: Insulin bolus (float)
- `insulin_on_board`: Calculated insulin remaining (float)
- `glucose_trend_20`: 20-minute glucose trend (float)
- `last_delta`: Immediate glucose change (float)

## Model Hyperparameters

The scripts use the hyperparameters from the original notebook:

- **Input chunk length**: 35 (looks back 35 time steps = ~3 hours)
- **Hidden size**: 6
- **LSTM layers**: 3
- **Attention heads**: 2
- **Batch size**: 48
- **Learning rate**: 0.0010223
- **Dropout**: 0.1

## Output Files

The evaluation scripts generate:

1. **Console output**: Detailed performance metrics table
2. **Prediction plots**: PNG files showing actual vs predicted values
3. **Model checkpoints**: If training (saved in model directory)

## Troubleshooting

### Common Issues:

1. **"Invalid past_covariates" error**:

   - The script automatically handles this by using the full feature dataset
   - This is normal for TFT models with autoregressive prediction

2. **Memory issues with large datasets**:

   - Reduce batch size in the model parameters
   - Use fewer epochs for quick evaluation

3. **Model loading errors**:
   - Ensure the model path exists and contains valid checkpoints
   - Use `--quick_train` to train a new model instead

### Performance Tips:

1. **For quick evaluation**: Use `--epochs 1` or `--epochs 5`
2. **For production training**: Use `--epochs 100` or more
3. **For different datasets**: Adjust the split ratio in the code (currently 90% train, 5% test, 5% holdout)

## Next Steps

1. **Run evaluation**: Start with `tft_glucose_evaluation.py --quick_train`
2. **Analyze results**: Look at which quantiles perform best for different time periods
3. **Implement quantile switching**: Based on physiological states (sleep, meals, etc.)
4. **Fine-tune hyperparameters**: Use the hyperparameter tuning notebook for optimization

## References

- [Original Article](https://brandonharris.io/Transforming%20Diabetes%20with%20AI%20-%20Temporal%20Fusion%20Transformers%20and%20Blood%20Glucose%20Data/)
- [Darts Documentation](https://unit8co.github.io/darts/)
- [TFT Paper](https://arxiv.org/abs/1912.09363)

"""
Chronos-specific data handling utilities.

This module provides univariate data handling for Chronos models,
which only use glucose_value for training and prediction.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
import warnings

warnings.filterwarnings("ignore")


class ChronosDataHandler:
    """
    Handles univariate data loading and preprocessing for Chronos models.
    
    Chronos models are univariate and only use the target variable (glucose_value).
    This handler extracts only the glucose values and ignores other features.
    """
    
    def __init__(self, data_path: str = "data/t1d_glucose_data.csv"):
        """
        Initialize the Chronos data handler.
        
        Args:
            data_path: Path to the glucose data CSV file
        """
        self.data_path = data_path
        self.data = None
        self.ts_target = None
        self.scaler_target = None
        
        # Default split ratios
        self.train_split = 0.90
        self.test_split = 0.05  # 5% of remaining data
        self.holdout_split = 0.05  # 5% of remaining data
        
    def load_data(self) -> TimeSeries:
        """
        Load and preprocess the glucose data for Chronos (univariate).
        
        Returns:
            Target time series (glucose values only)
        """
        print("Loading univariate data for Chronos...")
        
        # Load data
        self.data = pd.read_csv(self.data_path)
        self.data.rename(columns={"date_time": "datetime_col"}, inplace=True)
        self.data['datetime_col'] = pd.to_datetime(self.data['datetime_col'])
        
        # Convert glucose_value to float32
        self.data['glucose_value'] = self.data['glucose_value'].astype('float32')
        
        # Create time series object (univariate - only glucose_value)
        self.ts_target = TimeSeries.from_dataframe(
            self.data, 'datetime_col', 'glucose_value', freq='5T'
        )
        
        print(f"✓ Univariate data loaded: {len(self.data)} rows")
        print(f"✓ Time range: {self.data['datetime_col'].min()} to {self.data['datetime_col'].max()}")
        print(f"✓ Target variable: glucose_value")
        
        return self.ts_target
    
    def split_data(self, ts_target: TimeSeries) -> Tuple[TimeSeries, TimeSeries, TimeSeries]:
        """
        Split univariate data into train, test, and holdout sets.
        
        Args:
            ts_target: Target time series (glucose values)
            
        Returns:
            Tuple of (train, test, holdout) time series
        """
        print("Splitting univariate data into train/test/holdout sets...")
        
        # Calculate split points
        train_size = int(len(ts_target) * self.train_split)
        split_timestamp = ts_target.to_dataframe().index[train_size]
        
        # Split target data
        ts_train, ts_temp = ts_target.split_after(split_timestamp)
        test_size = int(len(ts_temp) * 0.5)  # Split remaining 50/50 for test/holdout
        split_timestamp_test = ts_temp.to_dataframe().index[test_size]
        ts_test, ts_holdout = ts_temp.split_after(split_timestamp_test)
        
        print(f"✓ Training set: {len(ts_train)} points ({ts_train.duration})")
        print(f"✓ Test set: {len(ts_test)} points ({ts_test.duration})")
        print(f"✓ Holdout set: {len(ts_holdout)} points ({ts_holdout.duration})")
        
        return ts_train, ts_test, ts_holdout
    
    def scale_data(self, ts_train: TimeSeries, ts_test: TimeSeries, 
                   ts_holdout: TimeSeries) -> Tuple[TimeSeries, TimeSeries, TimeSeries]:
        """
        Scale the univariate data using the same scaler as training.
        
        Args:
            ts_train: Training target data
            ts_test: Test target data
            ts_holdout: Holdout target data
            
        Returns:
            Tuple of scaled time series
        """
        print("Scaling univariate data...")
        
        # Scale target data
        self.scaler_target = Scaler()
        self.scaler_target.fit(ts_train)
        ts_train_scaled = self.scaler_target.transform(ts_train)
        ts_test_scaled = self.scaler_target.transform(ts_test)
        ts_holdout_scaled = self.scaler_target.transform(ts_holdout)
        
        # Convert to float32
        ts_train_scaled = ts_train_scaled.astype(np.float32)
        ts_test_scaled = ts_test_scaled.astype(np.float32)
        ts_holdout_scaled = ts_holdout_scaled.astype(np.float32)
        
        print("✓ Univariate data scaling completed")
        
        return ts_train_scaled, ts_test_scaled, ts_holdout_scaled
    
    def prepare_sequences(self, ts_data: TimeSeries, context_length: int, 
                         prediction_length: int) -> Tuple[List, List]:
        """
        Prepare sequences for Chronos training.
        
        Args:
            ts_data: Time series data
            context_length: Length of input context
            prediction_length: Length of prediction horizon
            
        Returns:
            Tuple of (context_sequences, target_sequences)
        """
        # Convert to numpy array
        values = ts_data.values().flatten()
        
        context_sequences = []
        target_sequences = []
        
        # Create sliding window sequences
        for i in range(len(values) - context_length - prediction_length + 1):
            context = values[i:i + context_length]
            target = values[i + context_length:i + context_length + prediction_length]
            
            context_sequences.append(context)
            target_sequences.append(target)
        
        return context_sequences, target_sequences
    
    def get_data_info(self) -> dict:
        """
        Get information about the loaded data.
        
        Returns:
            Dictionary with data information
        """
        if self.data is None:
            self.load_data()
        
        return {
            'n_rows': len(self.data),
            'n_features': 1,  # Univariate
            'time_range': (self.data['datetime_col'].min(), self.data['datetime_col'].max()),
            'target_name': 'glucose_value',
            'data_type': 'univariate'
        }
    
    def inverse_transform(self, ts_scaled: TimeSeries) -> TimeSeries:
        """
        Inverse transform scaled data back to original scale.
        
        Args:
            ts_scaled: Scaled time series
            
        Returns:
            Original scale time series
        """
        if self.scaler_target is None:
            raise ValueError("Scaler not fitted. Call scale_data first.")
        
        return self.scaler_target.inverse_transform(ts_scaled)

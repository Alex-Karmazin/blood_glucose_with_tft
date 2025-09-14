"""
Data handling utilities for glucose prediction models.

This module provides common data loading, preprocessing, and splitting functionality
that can be used across different model architectures.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
import warnings

warnings.filterwarnings("ignore")


class DataHandler:
    """
    Handles data loading, preprocessing, and splitting for glucose prediction models.
    """
    
    def __init__(self, data_path: str = "data/t1d_glucose_data.csv"):
        """
        Initialize the data handler.
        
        Args:
            data_path: Path to the glucose data CSV file
        """
        self.data_path = data_path
        self.data = None
        self.ts_target = None
        self.ts_features = None
        self.scaler_target = None
        self.scaler_features = None
        
        # Default split ratios
        self.train_split = 0.90
        self.test_split = 0.05  # 5% of remaining data
        self.holdout_split = 0.05  # 5% of remaining data
        
    def load_data(self) -> Tuple[TimeSeries, TimeSeries]:
        """
        Load and preprocess the glucose data.
        
        Returns:
            Tuple of (target_time_series, features_time_series)
        """
        print("Loading and preprocessing data...")
        
        # Load data
        self.data = pd.read_csv(self.data_path)
        self.data.rename(columns={"date_time": "datetime_col"}, inplace=True)
        self.data['datetime_col'] = pd.to_datetime(self.data['datetime_col'])
        
        # Convert data types
        self.data['bolus'] = self.data['bolus'].astype('float32')
        self.data['carbs'] = self.data['carbs'].astype('float32')
        self.data['insulin_on_board'] = self.data['insulin_on_board'].astype('float32')
        
        # Convert float64 columns to float32
        int_columns = self.data.select_dtypes(include=['float64']).columns
        self.data[int_columns] = self.data[int_columns].astype('float32')
        
        # Create time series objects
        self.ts_target = TimeSeries.from_dataframe(
            self.data, 'datetime_col', 'glucose_value', freq='5T'
        )
        
        # Extract features
        feature_cols = [col for col in self.data.columns 
                       if col not in ['glucose_value', 'datetime_col']]
        df_features = self.data[['datetime_col'] + feature_cols]
        self.ts_features = TimeSeries.from_dataframe(df_features, 'datetime_col', freq='5T')
        
        print(f"✓ Data loaded: {len(self.data)} rows, {len(feature_cols)} features")
        print(f"✓ Time range: {self.data['datetime_col'].min()} to {self.data['datetime_col'].max()}")
        
        return self.ts_target, self.ts_features
    
    def split_data(self, ts_target: TimeSeries, ts_features: TimeSeries) -> Tuple:
        """
        Split data into train, test, and holdout sets.
        
        Args:
            ts_target: Target time series
            ts_features: Features time series
            
        Returns:
            Tuple of split time series
        """
        print("Splitting data into train/test/holdout sets...")
        
        # Calculate split points
        train_size = int(len(ts_target) * self.train_split)
        split_timestamp = ts_target.to_dataframe().index[train_size]
        
        # Split target data
        ts_train, ts_temp = ts_target.split_after(split_timestamp)
        test_size = int(len(ts_temp) * 0.5)  # Split remaining 50/50 for test/holdout
        split_timestamp_test = ts_temp.to_dataframe().index[test_size]
        ts_test, ts_holdout = ts_temp.split_after(split_timestamp_test)
        
        # Split feature data
        ts_features_train, ts_features_temp = ts_features.split_after(split_timestamp)
        ts_features_test, ts_features_holdout = ts_features_temp.split_after(split_timestamp_test)
        
        print(f"✓ Training set: {len(ts_train)} points ({ts_train.duration})")
        print(f"✓ Test set: {len(ts_test)} points ({ts_test.duration})")
        print(f"✓ Holdout set: {len(ts_holdout)} points ({ts_holdout.duration})")
        
        return (ts_train, ts_test, ts_holdout, 
                ts_features_train, ts_features_test, ts_features_holdout)
    
    def scale_data(self, ts_train: TimeSeries, ts_test: TimeSeries, ts_holdout: TimeSeries,
                   ts_features_train: TimeSeries, ts_features_test: TimeSeries, 
                   ts_features_holdout: TimeSeries) -> Tuple:
        """
        Scale the data using the same scalers as training.
        
        Args:
            ts_train: Training target data
            ts_test: Test target data
            ts_holdout: Holdout target data
            ts_features_train: Training feature data
            ts_features_test: Test feature data
            ts_features_holdout: Holdout feature data
            
        Returns:
            Tuple of scaled time series
        """
        print("Scaling data...")
        
        # Scale target data
        self.scaler_target = Scaler()
        self.scaler_target.fit(ts_train)
        ts_train_scaled = self.scaler_target.transform(ts_train)
        ts_test_scaled = self.scaler_target.transform(ts_test)
        ts_holdout_scaled = self.scaler_target.transform(ts_holdout)
        
        # Scale feature data
        self.scaler_features = Scaler()
        self.scaler_features.fit(ts_features_train)
        ts_features_train_scaled = self.scaler_features.transform(ts_features_train)
        ts_features_test_scaled = self.scaler_features.transform(ts_features_test)
        ts_features_holdout_scaled = self.scaler_features.transform(ts_features_holdout)
        
        # Convert to float32
        ts_train_scaled = ts_train_scaled.astype(np.float32)
        ts_test_scaled = ts_test_scaled.astype(np.float32)
        ts_holdout_scaled = ts_holdout_scaled.astype(np.float32)
        ts_features_train_scaled = ts_features_train_scaled.astype(np.float32)
        ts_features_test_scaled = ts_features_test_scaled.astype(np.float32)
        ts_features_holdout_scaled = ts_features_holdout_scaled.astype(np.float32)
        
        print("✓ Data scaling completed")
        
        return (ts_train_scaled, ts_test_scaled, ts_holdout_scaled,
                ts_features_train_scaled, ts_features_test_scaled, ts_features_holdout_scaled)
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of feature columns.
        
        Returns:
            List of feature column names
        """
        if self.data is None:
            self.load_data()
        
        feature_cols = [col for col in self.data.columns 
                       if col not in ['glucose_value', 'datetime_col']]
        return feature_cols
    
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
            'n_features': len(self.get_feature_names()),
            'time_range': (self.data['datetime_col'].min(), self.data['datetime_col'].max()),
            'feature_names': self.get_feature_names(),
            'target_name': 'glucose_value'
        }

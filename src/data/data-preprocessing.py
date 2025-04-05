import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import logging

logger = logging.getLogger(__name__)

def load_and_preprocess_data(equip_file='Equip.csv', weather_file='weather.csv'):
    """
    Focused data preprocessing with key features for convergence
    """
    logger.info("Loading and preprocessing data...")
    
    try:
        # Load datasets
        equip_df = pd.read_csv(equip_file)
        weather_df = pd.read_csv(weather_file)
        
        # Convert timestamps
        equip_df['ts'] = pd.to_datetime(equip_df['ts'])
        weather_df['ts'] = pd.to_datetime(weather_df['ts'])
        
        # Merge datasets
        merged_df = pd.merge(equip_df, weather_df, on='ts', how='inner')
        
        # Fill missing values
        merged_df = merged_df.fillna(method='ffill').fillna(method='bfill')
        
        # Create time features
        merged_df['hour'] = merged_df['ts'].dt.hour
        merged_df['is_weekend'] = (merged_df['ts'].dt.dayofweek >= 5).astype(int)
        
        # Add physical features
        if all(col in merged_df.columns for col in ['RaTemp', 'main.temp']):
            merged_df['temp_differential'] = merged_df['RaTemp'] - merged_df['main.temp']
        
        if all(col in merged_df.columns for col in ['SaTemp', 'RaTemp']):
            merged_df['supply_return_diff'] = merged_df['SaTemp'] - merged_df['RaTemp']
            merged_df['heating_mode'] = (merged_df['supply_return_diff'] > 0).astype(int)
        
        # Add lag features
        if 'Valve' in merged_df.columns:
            merged_df['Valve_lag1'] = merged_df['Valve'].shift(1)
            merged_df['Valve_lag2'] = merged_df['Valve'].shift(2)
            merged_df['valve_delta'] = merged_df['Valve'] - merged_df['Valve_lag1']
        
        for col in ['RaTemp', 'SaTemp', 'main.temp']:
            if col in merged_df.columns:
                merged_df[f'{col}_lag1'] = merged_df[col].shift(1)
        
        # Drop NaN values
        merged_df = merged_df.dropna().reset_index(drop=True)
        
        # Select features
        base_features = [col for col in [
            'RaTemp', 'SaTemp', 'ThermEnergy', 'main.temp', 'main.humidity', 
            'hour', 'is_weekend', 'temp_differential', 'supply_return_diff', 'heating_mode'
        ] if col in merged_df.columns]
        
        lag_features = [col for col in merged_df.columns if '_lag' in col]
        features = base_features + lag_features
        
        # Scale features
        scaler = RobustScaler()
        merged_df[features] = scaler.fit_transform(merged_df[features])
        
        # Define targets
        valve_target = 'Valve'
        delta_target = 'valve_delta'
        
        logger.info(f"Processed data shape: {merged_df.shape}")
        logger.info(f"Using {len(features)} features")
        
        return merged_df, features, valve_target, delta_target, scaler
        
    except Exception as e:
        logger.error(f"Error in data preprocessing: {e}")
        raise
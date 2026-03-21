"""
Synthetic Data Generator for Ensemble-Based Hybrid Disaster Prediction System
Generates realistic environmental data for training and testing
"""

import numpy as np
import pandas as pd
from typing import Optional
import config


def generate_flood_data(n_samples: int = 500, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic flood disaster data with realistic correlations.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with flood-related features and target
    """
    np.random.seed(random_state)
    
    # Generate base features
    rainfall = np.random.exponential(scale=80, size=n_samples) + np.random.uniform(0, 50, n_samples)
    rainfall = np.clip(rainfall, 0, 500)
    
    temperature = np.random.normal(28, 5, n_samples)
    temperature = np.clip(temperature, 10, 45)
    
    humidity = np.random.beta(5, 2, n_samples) * 100
    humidity = np.clip(humidity, 20, 100)
    
    # Soil moisture correlates with rainfall
    soil_moisture = 25 + 0.24 * rainfall + np.random.normal(0, 6, n_samples)
    soil_moisture = np.clip(soil_moisture, 0, 100)
    
    wind_speed = np.random.gamma(2, 15, n_samples)
    wind_speed = np.clip(wind_speed, 0, 150)
    
    # Atmospheric pressure inversely correlates with rainfall (lower pressure = storms)
    atmospheric_pressure = 1013 - 0.09 * rainfall + np.random.normal(0, 6, n_samples)
    atmospheric_pressure = np.clip(atmospheric_pressure, 900, 1050)
    
    # Previous disaster occurrence
    previous_disaster = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    
    # Generate target based on features
    # Higher risk with: high rainfall, high soil moisture, high humidity, low pressure
    risk_score = (
        0.48 * (rainfall / 500) +
        0.30 * (soil_moisture / 100) +
        0.12 * (humidity / 100) +
        0.04 * (wind_speed / 150) +
        0.04 * ((1050 - atmospheric_pressure) / 150) +
        0.02 * previous_disaster
    )
    
    # Add some noise and threshold
    risk_score += np.random.normal(0, 0.03, n_samples)
    target = (risk_score > 0.50).astype(int)
    
    df = pd.DataFrame({
        'rainfall': rainfall,
        'temperature': temperature,
        'humidity': humidity,
        'soil_moisture': soil_moisture,
        'wind_speed': wind_speed,
        'atmospheric_pressure': atmospheric_pressure,
        'previous_disaster': previous_disaster,
        'disaster_type': 'Flood',
        'target': target
    })
    
    return df


def generate_earthquake_data(n_samples: int = 500, random_state: int = 43) -> pd.DataFrame:
    """
    Generate synthetic earthquake disaster data with realistic patterns.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with earthquake-related features and target
    """
    np.random.seed(random_state)
    
    # Generate seismic activity (Richter scale)
    seismic_activity = np.random.exponential(scale=2, size=n_samples)
    seismic_activity = np.clip(seismic_activity, 0, 10)
    
    temperature = np.random.normal(25, 8, n_samples)
    temperature = np.clip(temperature, 5, 45)
    
    humidity = np.random.beta(3, 3, n_samples) * 100
    humidity = np.clip(humidity, 10, 95)
    
    wind_speed = np.random.gamma(2, 10, n_samples)
    wind_speed = np.clip(wind_speed, 0, 100)
    
    # Atmospheric pressure (some studies suggest correlation with seismic activity)
    atmospheric_pressure = 1013 + np.random.normal(0, 15, n_samples)
    atmospheric_pressure = np.clip(atmospheric_pressure, 950, 1060)
    
    # Previous disaster occurrence
    previous_disaster = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
    
    # Generate target based on features
    # Higher risk with: high seismic activity, previous occurrences
    risk_score = (
        0.60 * (seismic_activity / 10) +
        0.15 * previous_disaster +
        0.10 * (np.abs(atmospheric_pressure - 1013) / 50) +
        0.10 * np.random.uniform(0, 0.3, n_samples) +
        0.05 * (humidity / 100)
    )
    
    # Add some noise and threshold
    risk_score += np.random.normal(0, 0.08, n_samples)
    target = (risk_score > 0.35).astype(int)
    
    df = pd.DataFrame({
        'seismic_activity': seismic_activity,
        'temperature': temperature,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'atmospheric_pressure': atmospheric_pressure,
        'previous_disaster': previous_disaster,
        'disaster_type': 'Earthquake',
        'target': target
    })
    
    return df


def generate_unified_dataset(
    flood_samples: int = 500,
    earthquake_samples: int = 500,
    missing_rate: float = 0.05,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate unified dataset containing both flood and earthquake data.
    Introduces missing values for realistic preprocessing scenarios.
    
    Parameters:
    -----------
    flood_samples : int
        Number of flood samples
    earthquake_samples : int
        Number of earthquake samples
    missing_rate : float
        Proportion of values to set as missing (0-1)
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    pd.DataFrame
        Unified DataFrame with both disaster types
    """
    # Generate data for both disaster types
    flood_df = generate_flood_data(flood_samples, random_state)
    earthquake_df = generate_earthquake_data(earthquake_samples, random_state + 1)
    
    # Add missing columns to each dataset
    flood_df['seismic_activity'] = np.nan
    earthquake_df['rainfall'] = np.nan
    earthquake_df['soil_moisture'] = np.nan
    
    # Combine datasets
    df = pd.concat([flood_df, earthquake_df], ignore_index=True)
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Introduce random missing values in numeric columns
    np.random.seed(random_state)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c != 'target']
    
    for col in numeric_cols:
        mask = np.random.random(len(df)) < missing_rate
        # Only set missing where there's already a value
        valid_mask = df[col].notna()
        final_mask = mask & valid_mask
        df.loc[final_mask, col] = np.nan
    
    return df


def save_sample_dataset(output_path: Optional[str] = None):
    """
    Generate and save a sample dataset for training.
    
    Parameters:
    -----------
    output_path : str, optional
        Path to save the CSV file
    """
    if output_path is None:
        output_path = config.DATA_DIR / "disaster_dataset.csv"
    
    df = generate_unified_dataset(
        flood_samples=600,
        earthquake_samples=400,
        missing_rate=0.03
    )
    
    df.to_csv(output_path, index=False)
    print(f"✓ Dataset saved to: {output_path}")
    print(f"  Total samples: {len(df)}")
    print(f"  Flood samples: {len(df[df['disaster_type'] == 'Flood'])}")
    print(f"  Earthquake samples: {len(df[df['disaster_type'] == 'Earthquake'])}")
    print(f"  Positive class rate: {df['target'].mean():.2%}")
    
    return df


if __name__ == "__main__":
    save_sample_dataset()

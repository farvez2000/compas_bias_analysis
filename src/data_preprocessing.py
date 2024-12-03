# src/data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import os

def load_data(filepath):
    """Load raw COMPAS data from CSV."""
    data = pd.read_csv(filepath)
    return data

def clean_data(data):
    """Handle missing values and remove irrelevant columns."""
    # Drop rows with missing values in critical columns
    data_clean = data.dropna(subset=['sex', 'age', 'age_cat', 'race', 'decile_score', 'is_recid'])
    return data_clean

def encode_categorical_features(data):
    """One-Hot Encode categorical features."""
    categorical_features = ['sex', 'age_cat', 'race']
    encoder = OneHotEncoder(drop='first', sparse=False)
    encoded_cats = pd.DataFrame(encoder.fit_transform(data[categorical_features]), 
                                columns=encoder.get_feature_names_out(categorical_features),
                                index=data.index)
    data_encoded = pd.concat([data.drop(columns=categorical_features), encoded_cats], axis=1)
    return data_encoded

def scale_features(data, numerical_features):
    """Standardize numerical features."""
    scaler = StandardScaler()
    data[numerical_features] = scaler.fit_transform(data[numerical_features])
    return data

def preprocess_data(raw_filepath, processed_filepath):
    """Complete preprocessing pipeline."""
    data = load_data(raw_filepath)
    data_clean = clean_data(data)
    data_encoded = encode_categorical_features(data_clean)
    numerical_features = ['age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count']
    data_scaled = scale_features(data_encoded, numerical_features)
    data_scaled.to_csv(processed_filepath, index=False)
    print(f"Processed data saved to {processed_filepath}")

if __name__ == "__main__":
    raw_data_path = 'data/raw/compas-scores-two-years.csv'
    processed_data_path = 'data/processed/compas_processed.csv'
    preprocess_data(raw_data_path, processed_data_path)

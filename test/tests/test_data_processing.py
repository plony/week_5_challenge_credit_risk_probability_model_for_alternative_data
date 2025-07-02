# tests/test_data_processing.py

import pandas as pd
import pytest
import os

# Dummy function to simulate a part of the cleaning process for testing
# In a real scenario, you'd import a function directly from your src/ script,
# e.g., from src.proxy_target_variable_engineering import your_helper_function
def _simulate_initial_cleaning(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Simulates initial cleaning steps: dropping unnamed columns and
    converting TransactionStartTime to datetime.
    """
    df = df_raw.copy()

    # 1. Remove 'Unnamed' columns
    unnamed_cols = [col for col in df.columns if col.startswith('Unnamed:')]
    if unnamed_cols:
        df.drop(columns=unnamed_cols, inplace=True)

    # 2. Convert TransactionStartTime to datetime objects
    # Ensure this column exists before trying to convert
    if 'TransactionStartTime' in df.columns:
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce', utc=True)
        df.dropna(subset=['TransactionStartTime'], inplace=True)
    return df

# Test 1: Check if 'Unnamed' columns are dropped
def test_unnamed_columns_are_dropped():
    data = {
        'col1': [1, 2],
        'Unnamed: 0': [3, 4],
        'col2': [5, 6],
        'Unnamed: 1': [7, 8],
        'TransactionStartTime': ['2023-01-01', '2023-01-02'] # Added for _simulate_initial_cleaning to work
    }
    df_raw = pd.DataFrame(data)
    df_cleaned = _simulate_initial_cleaning(df_raw)
    assert 'Unnamed: 0' not in df_cleaned.columns
    assert 'Unnamed: 1' not in df_cleaned.columns
    assert 'col1' in df_cleaned.columns
    assert 'col2' in df_cleaned.columns
    # Expecting 3 columns: col1, col2, TransactionStartTime
    assert len(df_cleaned.columns) == 3 

# Test 2: Check if TransactionStartTime is converted to datetime and unparseable rows are dropped
def test_transaction_starttime_conversion_and_na_drop():
    data = {
        'TransactionStartTime': ['2023-01-01 10:00:00', 'invalid_date', '2023-01-02 11:00:00'],
        'col_other': [1, 2, 3],
        'Unnamed: 0': [4,5,6] # Add unnamed to ensure it's also handled by the same function
    }
    df_raw = pd.DataFrame(data)
    df_cleaned = _simulate_initial_cleaning(df_raw)

    # Check if 'invalid_date' row was dropped
    assert len(df_cleaned) == 2
    # Check if TransactionStartTime is datetime type
    assert pd.api.types.is_datetime64_any_dtype(df_cleaned['TransactionStartTime'])
    # Check if unnamed was dropped
    assert 'Unnamed: 0' not in df_cleaned.columns
    assert 'col_other' in df_cleaned.columns

# Test 3: Check if no changes occur with clean data
def test_clean_data_no_change():
    data = {
        'col1': [1, 2],
        'col2': [5, 6],
        'TransactionStartTime': ['2023-01-01 10:00:00', '2023-01-02 11:00:00']
    }
    df_raw = pd.DataFrame(data)
    df_cleaned = _simulate_initial_cleaning(df_raw)
    assert df_raw.shape == df_cleaned.shape
    assert pd.api.types.is_datetime64_any_dtype(df_cleaned['TransactionStartTime'])
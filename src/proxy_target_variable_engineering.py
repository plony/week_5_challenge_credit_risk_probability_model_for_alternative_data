# src/proxy_target_variable_engineering.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

def clean_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs initial data cleaning, calculates RFM metrics, clusters customers
    to create a proxy 'is_high_risk' target variable, and integrates it
    into the main DataFrame.

    Args:
        df (pd.DataFrame): The raw input DataFrame containing transaction data.

    Returns:
        pd.DataFrame: The DataFrame with cleaned data and the new 'is_high_risk' column.
    """

    print("Starting data cleaning and feature engineering...")

    # --- Initial Data Inspection and Cleaning ---

    # 1. Remove 'Unnamed' columns
    unnamed_cols = [col for col in df.columns if col.startswith('Unnamed:')]
    if unnamed_cols:
        df.drop(columns=unnamed_cols, inplace=True)
        print(f"Dropped unnamed columns: {unnamed_cols}")

    # 2. Convert TransactionStartTime to datetime objects
    initial_rows = df.shape[0]
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce', utc=True)
    # Drop rows where TransactionStartTime could not be parsed
    df.dropna(subset=['TransactionStartTime'], inplace=True)
    rows_after_datetime_clean = df.shape[0]
    if initial_rows != rows_after_datetime_clean:
        print(f"Dropped {initial_rows - rows_after_datetime_clean} rows due to unparseable TransactionStartTime.")

    # 3. Handle missing values for other critical columns if necessary for RFM or later use
    # Based on previous run, 'AccountId', 'CountryCode', 'ProviderId', 'Value', 'PricingStrategy' had missing values.
    # For RFM, only CustomerId, TransactionStartTime, and Amount are strictly needed.
    # It's good practice to decide on imputation or dropping for other columns based on context.
    # For this script, we proceed as long as RFM core columns are clean.

    # Ensure 'Amount' is numeric for Monetary calculation
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    df.dropna(subset=['Amount'], inplace=True) # Drop rows where Amount is not convertible


    # --- Task 4: Proxy Target Variable Engineering ---

    # Define a snapshot date for Recency calculation
    # Use the day after the latest transaction date in the dataset
    snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
    print(f"\nSnapshot date for RFM calculation: {snapshot_date}")

    # Calculate RFM Metrics
    # Ensure CustomerId exists and is suitable for grouping
    if 'CustomerId' not in df.columns:
        raise ValueError("CustomerId column not found, which is required for RFM calculation.")

    rfm_df = df.groupby('CustomerId').agg(
        Recency=('TransactionStartTime', lambda date: (snapshot_date - date.max()).days),
        Frequency=('TransactionId', 'count'),
        Monetary=('Amount', lambda amount: amount.abs().sum()) # Using absolute Amount for monetary value
    ).reset_index()

    print("\nRFM DataFrame Head:")
    print(rfm_df.head())

    # Pre-process RFM features (Scaling)
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])
    rfm_scaled_df = pd.DataFrame(rfm_scaled, columns=['Recency_scaled', 'Frequency_scaled', 'Monetary_scaled'], index=rfm_df.index)

    print("\nScaled RFM DataFrame Head:")
    print(rfm_scaled_df.head())

    # Cluster Customers using K-Means
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto') # n_init='auto' is default in newer versions
    rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)

    print("\nRFM DataFrame with Clusters Head:")
    print(rfm_df.head())

    # Analyze Clusters to define "High-Risk"
    cluster_means = rfm_df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    print("\nMean RFM values per cluster:")
    print(cluster_means)

    # Identify the high-risk cluster. This requires looking at the cluster means.
    # A high-risk cluster typically has:
    # - High Recency (last transaction long ago, so a larger number of days)
    # - Low Frequency (few transactions)
    # - Low Monetary (low total spend)
    # Sort by Recency (descending), then Frequency (ascending), then Monetary (ascending)
    # The first cluster in this sorted order is likely the high-risk one.
    sorted_clusters = cluster_means.sort_values(by=['Recency', 'Frequency', 'Monetary'], ascending=[False, True, True])
    high_risk_cluster_id = sorted_clusters.index[0]

    print(f"\nIdentified High-Risk Cluster ID: {high_risk_cluster_id}")

    # Create the is_high_risk column
    rfm_df['is_high_risk'] = (rfm_df['Cluster'] == high_risk_cluster_id).astype(int)

    print("\nRFM DataFrame with is_high_risk column head:")
    print(rfm_df.head())
    print("\nDistribution of is_high_risk in RFM DataFrame:")
    print(rfm_df['is_high_risk'].value_counts())

    # Integrate the Target Variable into the main processed dataset
    # Merge the is_high_risk column back into the original dataframe (or a copy of it)
    df_with_target = pd.merge(df, rfm_df[['CustomerId', 'is_high_risk']], on='CustomerId', how='left')

    print("\nMain DataFrame with 'is_high_risk' target variable head:")
    print(df_with_target.head())
    print("\nDistribution of is_high_risk in the main DataFrame (transaction level):")
    print(df_with_target['is_high_risk'].value_counts())

    print("\nData cleaning and feature engineering complete.")
    return df_with_target

if __name__ == "__main__":
    # Example usage:
    # Adjusted path for 'data.csv' based on your project structure: data/raw/data.csv
    try:
        raw_data = pd.read_csv('D:/10academy/week_5_challenge_credit_risk_probability_model_for_alternative_data/data/raw/data.csv') # Corrected path here
        processed_data = clean_and_engineer_features(raw_data.copy()) # Pass a copy to avoid modifying original
        processed_data.to_csv('processed_data_with_proxy_target.csv', index=False)
        print("\nProcessed data with proxy target variable saved to 'processed_data_with_proxy_target.csv'")
    except FileNotFoundError:
        print("Error: 'data.csv' not found. Please ensure the data file is in the correct directory, relative to where you run the script.")
    except Exception as e:
        print(f"An error occurred during execution: {e}")
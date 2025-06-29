import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn import set_config

# Set global output for sklearn transformers to pandas
set_config(transform_output="pandas")


# --- Define Features at Module Level ---
initial_numerical_features = ['Amount', 'Value']
categorical_features = ['ProviderId', 'ProductId', 'ProductCategory', 'ChannelId', 'PricingStrategy']

# This list must accurately represent all numerical features *after*
# TimeFeatureExtractor and CustomerAggregator have run.
# Ensure consistency with how these transformers generate column names.
all_numerical_features_for_scaling = initial_numerical_features + [
    'transaction_hour', 'transaction_day_of_week', 'transaction_month', 'transaction_year',
    'Amount_sum_per_customer', 'Amount_mean_per_customer', 'Amount_count_per_customer', 'Amount_std_per_customer',
    'Value_sum_per_customer', 'Value_mean_per_customer', 'Value_count_per_customer', 'Value_std_per_customer'
]


# --- Custom Transformers (keep your latest version with dtypes ensured) ---
class TimeFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        if not pd.api.types.is_datetime64_any_dtype(X_copy['TransactionStartTime']):
            X_copy['TransactionStartTime'] = pd.to_datetime(X_copy['TransactionStartTime'], errors='coerce')
            X_copy.dropna(subset=['TransactionStartTime'], inplace=True)
        X_copy['transaction_hour'] = X_copy['TransactionStartTime'].dt.hour
        X_copy['transaction_day_of_week'] = X_copy['TransactionStartTime'].dt.dayofweek
        X_copy['transaction_month'] = X_copy['TransactionStartTime'].dt.month
        X_copy['transaction_year'] = X_copy['TransactionStartTime'].dt.year
        X_copy = X_copy.drop('TransactionStartTime', axis=1)
        return X_copy

class CustomerAggregator(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_cols=['Amount', 'Value']):
        self.numerical_cols = numerical_cols
        self.agg_features_df = None

    def fit(self, X, y=None):
        print("Fitting CustomerAggregator...")
        if 'CustomerId' not in X.columns:
            raise ValueError("CustomerId column not found for aggregation.")
        agg_dict = {col: ['sum', 'mean', 'count', 'std'] for col in self.numerical_cols}
        self.agg_features_df = X[['CustomerId'] + self.numerical_cols].groupby('CustomerId').agg(agg_dict)
        self.agg_features_df.columns = ['_'.join(col).strip() + '_per_customer'
                                        for col in self.agg_features_df.columns.values]
        print(f"Aggregated features calculated for {len(self.agg_features_df)} unique customers.")
        return self

    def transform(self, X):
        X_copy = X.copy()
        if self.agg_features_df is None:
            raise RuntimeError("CustomerAggregator must be fitted before transforming.")
        if 'CustomerId' not in X_copy.columns:
            print("Warning: CustomerId not found in X_copy for merging aggregates. Re-check pipeline order.")
            return X_copy
        X_transformed = pd.merge(X_copy, self.agg_features_df, on='CustomerId', how='left')
        new_agg_cols = self.agg_features_df.columns.tolist()
        for col in new_agg_cols:
            if col in X_transformed.columns:
                if 'count' in col:
                    X_transformed[col] = X_transformed[col].fillna(0).astype(float)
                else:
                    median_val = self.agg_features_df[col].median()
                    fill_value = median_val if not pd.isna(median_val) else 0.0
                    X_transformed[col] = X_transformed[col].fillna(fill_value).astype(float)
        for col in self.numerical_cols:
            if col in X_transformed.columns:
                X_transformed[col] = pd.to_numeric(X_transformed[col], errors='coerce')
        return X_transformed


def get_preprocessing_pipeline():
    """
    Returns a scikit-learn Pipeline for data preprocessing and feature engineering.
    Uses globally defined feature lists and expects pandas output from transformers.
    """
    pipeline_steps = [
        ('time_features', TimeFeatureExtractor()),
        ('customer_aggregates', CustomerAggregator(numerical_cols=initial_numerical_features)),
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num_pipeline', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), all_numerical_features_for_scaling),
            ('cat_onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )

    pipeline_steps.append(('preprocessor', preprocessor))

    full_pipeline = Pipeline(pipeline_steps)
    return full_pipeline

# --- Testing Block ---
if __name__ == '__main__':
    print("--- Testing Feature Engineering Pipeline ---")

    data = {
        'TransactionId': [1, 2, 3, 4, 5, 6, 7, 8],
        'TransactionStartTime': ['2019-01-01 10:30:00', '2019-01-01 11:45:00', '2019-01-01 10:35:00',
                                 '2019-01-02 14:00:00', '2019-01-02 14:15:00', '2019-01-02 03:00:00',
                                 '2019-02-05 09:00:00', '2019-02-05 09:10:00'],
        'CustomerId': ['C1', 'C2', 'C1', 'C3', 'C1', 'C2', 'C4', 'C4'],
        'Amount': [100.0, 50.0, 200.0, 150.0, -50.0, 1000.0, 30.0, 60.0],
        'Value': [100.0, 50.0, 200.0, 150.0, 50.0, 1000.0, 30.0, 60.0],
        'ProviderId': ['P1', 'P2', 'P1', 'P3', 'P1', 'P2', 'P1', 'P1'],
        'ProductId': ['ProdA', 'ProdB', 'ProdA', 'ProdC', 'ProdA', 'ProdB', 'ProdA', 'ProdA'],
        'ProductCategory': ['CatX', 'CatY', 'CatX', 'CatZ', 'CatX', 'CatY', 'CatX', 'CatX'],
        'ChannelId': ['ChA', 'ChB', 'ChA', 'ChC', 'ChA', 'ChB', 'ChA', 'ChA'],
        'PricingStrategy': [1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 1.0],
        'FraudResult': [0, 0, 0, 1, 0, 1, 0, 0]
    }
    sample_df = pd.DataFrame(data)

    print("\n--- Original Sample Data Head ---")
    print(sample_df.head())
    print("\nOriginal Sample Data Info:")
    sample_df.info()

    preprocessing_pipeline = get_preprocessing_pipeline()
    X_sample = sample_df.drop('FraudResult', axis=1)
    y_sample = sample_df['FraudResult']

    X_transformed_df = preprocessing_pipeline.fit_transform(X_sample)

    print("\n--- Transformed Data Shape ---")
    print(X_transformed_df.shape)

    print("\n--- Transformed Data Head (as DataFrame) ---")
    print(X_transformed_df.head())
    print("\nTransformed Data Info:")
    X_transformed_df.info()

    print("\n--- Checking specific transformed features ---")
    print("Example: transaction_hour distribution (should be scaled):")
    # Corrected column name
    print(X_transformed_df['num_pipeline__transaction_hour'].describe())

    print("\nExample: Amount_mean_per_customer for Customer C1 (should be scaled):")
    # Corrected column name
    print(X_transformed_df[X_transformed_df['remainder__CustomerId'] == 'C1']['num_pipeline__Amount_mean_per_customer'].head())

    print("\nExample: One-hot encoded ProductId columns:")
    # Filter using the 'cat_onehot__' prefix
    print(X_transformed_df.filter(regex='^cat_onehot__ProductId_').head())

    print("\n--- Checking Dtypes of a few transformed numerical columns ---")
    # Corrected column names
    print(X_transformed_df[['num_pipeline__Amount', 'num_pipeline__Value', 'num_pipeline__transaction_hour', 'num_pipeline__Amount_mean_per_customer']].dtypes)

    print("\n--- Checking Dtypes of remainder columns ---")
    print(X_transformed_df[['remainder__TransactionId', 'remainder__CustomerId']].dtypes)
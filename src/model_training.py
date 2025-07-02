# src/model_training.py

import os
# IMPORTANT: These environment variables must be set before any mlflow import/call
# They instruct MLflow to store paths within its metadata as if the root is '/app',
# which directly corresponds to your Docker volume mount point.
os.environ["MLFLOW_TRACKING_URI"] = "file:///app/mlruns"
os.environ["MLFLOW_REGISTRY_URI"] = "sqlite:////app/mlruns.db"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- MLflow Configuration ---
# These set_uri calls will now reflect the environment variables set above,
# ensuring MLflow's internal path handling uses the /app/mlruns base.
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
logger.info(f"MLflow Tracking URI set to: {mlflow.get_tracking_uri()}")

mlflow.set_registry_uri(os.environ["MLFLOW_REGISTRY_URI"])
logger.info(f"MLflow Registry URI set to: {mlflow.get_registry_uri()}")

# Set experiment name
mlflow.set_experiment("Credit_Risk_Model_Training")
# --- End MLflow Configuration ---


def train_and_evaluate_model(model_name: str, model_pipeline: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, param_grid: dict):
    """
    Trains and evaluates a given model with hyperparameter tuning,
    and logs results to MLflow.

    Args:
        model_name (str): Name of the model (e.g., "Logistic Regression").
        model_pipeline (Pipeline): Scikit-learn pipeline for the model.
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.Series): Training target.
        y_test (pd.Series): Testing target.
        param_grid (dict): Hyperparameter grid for GridSearchCV.
    """
    logger.info(f"Starting training and evaluation for {model_name}...")

    with mlflow.start_run(run_name=model_name):
        # NO: mlflow.log_params(param_grid) # REMOVED THIS LINE TO PREVENT MLFLOW ERROR

        # Hyperparameter Tuning using GridSearchCV
        logger.info(f"Running GridSearchCV for {model_name}...")
        grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        logger.info(f"Best parameters for {model_name}: {best_params}")
        logger.info(f"Best cross-validation ROC-AUC for {model_name}: {best_score:.4f}")

        mlflow.log_params(best_params) # Log the best parameters found (THIS IS CORRECT)

        # Make predictions
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1] # Probability of the positive class

        # Evaluate model performance
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        logger.info(f"{model_name} Metrics:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        logger.info(f"  ROC-AUC: {roc_auc:.4f}")

        # Log metrics to MLflow
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "best_cv_roc_auc": best_score
        })

        # Log the best model
        mlflow.sklearn.log_model(best_model, "model")

        return best_model, roc_auc

if __name__ == "__main__":
    processed_data_path = 'processed_data_with_proxy_target.csv'

    try:
        df = pd.read_csv(processed_data_path)
        logger.info(f"Successfully loaded data from {processed_data_path}")
    except FileNotFoundError:
        logger.error(f"Error: {processed_data_path} not found. Please ensure the processed data file from Task 4 is in the correct location (your project root).")
        exit()
    except Exception as e:
        logger.error(f"An error occurred while loading data: {e}")
        exit()

    numerical_features = ['Amount', 'PricingStrategy']
    categorical_features = ['CurrencyCode', 'CountryCode', 'ProviderId', 'ProductId', 'ProductCategory', 'ChannelId']

    for col in numerical_features:
        if col in df.columns:
            if df[col].isnull().any():
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col].fillna(df[col].mean(), inplace=True)
                logger.info(f"Filled NaN in numerical column '{col}' with mean.")
        else:
            logger.warning(f"Numerical feature '{col}' not found in DataFrame. Skipping NaN fill for this column.")

    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].astype(str)
            if df[col].isnull().any() or 'nan' in df[col].unique():
                df[col].replace('nan', 'Missing', inplace=True)
                df[col].fillna('Missing', inplace=True)
                logger.info(f"Ensured '{col}' is string and filled NaNs/converted 'nan' strings to 'Missing'.")
        else:
            logger.warning(f"Categorical feature '{col}' not found in DataFrame. Skipping NaN fill for this column.")

    df.dropna(subset=['is_high_risk'], inplace=True)
    df['is_high_risk'] = df['is_high_risk'].astype(int)

    X = df[numerical_features + categorical_features]
    y = df['is_high_risk']

    logger.info(f"Features (X) shape: {X.shape}, Target (y) shape: {y.shape}")
    logger.info(f"Target distribution:\n{y.value_counts(normalize=True)}")

    customer_labels = df[['CustomerId', 'is_high_risk']].drop_duplicates()
    unique_customer_ids = customer_labels['CustomerId']
    unique_customer_y = customer_labels['is_high_risk']

    X_train_cust, X_test_cust, _, _ = train_test_split(
        unique_customer_ids, unique_customer_y, test_size=0.2, random_state=42, stratify=unique_customer_y
    )

    X_train_df = df[df['CustomerId'].isin(X_train_cust)]
    X_test_df = df[df['CustomerId'].isin(X_test_cust)]

    y_train = X_train_df['is_high_risk']
    y_test = X_test_df['is_high_risk']

    X_train = X_train_df[numerical_features + categorical_features]
    X_test = X_test_df[numerical_features + categorical_features]

    logger.info(f"Training data size: {X_train.shape[0]} transactions")
    logger.info(f"Testing data size: {X_test.shape[0]} transactions")
    logger.info(f"Train target distribution:\n{y_train.value_counts(normalize=True)}")
    logger.info(f"Test target distribution:\n{y_test.value_counts(normalize=True)}")

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    models = {
        "Logistic Regression": {
            "pipeline": Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced'))
            ]),
            "param_grid": {
                'classifier__C': [0.1, 1.0, 10.0],
                'classifier__penalty': ['l1', 'l2']
            }
        },
        "Random Forest": {
            "pipeline": Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
            ]),
            "param_grid": {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [10, 20, None],
                'classifier__min_samples_split': [2, 5]
            }
        }
    }

    best_roc_auc_overall = -1
    best_model_overall = None
    best_model_name_overall = ""

    for model_name, config in models.items():
        logger.info(f"\n--- Training {model_name} ---")
        model, roc_auc = train_and_evaluate_model(
            model_name=model_name,
            model_pipeline=config["pipeline"],
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            param_grid=config["param_grid"]
        )

        if roc_auc > best_roc_auc_overall:
            best_roc_auc_overall = roc_auc
            best_model_overall = model
            best_model_name_overall = model_name

    logger.info(f"\n--- Best Model Identified ---")
    logger.info(f"Overall Best Model: {best_model_name_overall} with ROC-AUC: {best_roc_auc_overall:.4f}")

    if best_model_overall:
        logger.info(f"Registering the best model '{best_model_name_overall}' to MLflow Model Registry...")
        with mlflow.start_run(run_name="Best Model Registration"):
            mlflow.sklearn.log_model(
                sk_model=best_model_overall,
                artifact_path="best_credit_risk_model",
                registered_model_name="CreditRiskProxyModel"
            )
            mlflow.log_metric("overall_best_roc_auc", best_roc_auc_overall)
            mlflow.log_param("best_model_name", best_model_name_overall)
        logger.info("Best model registered successfully.")
    else:
        logger.warning("No best model identified for registration.")

    logger.info("\nModel training and tracking complete.")
# src/api/main.py

from fastapi import FastAPI, HTTPException
from src.api.pydantic_models import InputData, PredictionOutput
import mlflow
import mlflow.sklearn
import pandas as pd
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Credit Risk Probability Model API",
    description="A REST API for predicting credit risk probability based on customer transaction data.",
    version="1.0.0",
)

# Global variable to hold the loaded model and its preprocessor
# We will load the model in the startup event to ensure it's loaded only once
model = None

# Define the name and version of the registered MLflow model
MLFLOW_REGISTERED_MODEL_NAME = "CreditRiskProxyModel"
# It's good practice to specify a version or use 'production' alias
MLFLOW_MODEL_VERSION = 1 # Use the version you just registered, or 'latest'

# Ensure MLflow tracking URI is set for model loading
# This should match how it was set in model_training.py
mlflow_tracking_uri = "file:///" + os.path.abspath(os.path.join(os.getcwd(), "mlruns")).replace("\\", "/")
mlflow.set_tracking_uri(mlflow_tracking_uri)
logger.info(f"MLflow Tracking URI for API: {mlflow_tracking_uri}")

@app.on_event("startup")
async def load_model():
    """
    Load the best model from MLflow Model Registry when the API starts up.
    """
    global model
    try:
        # Load the specific version of the model
        model_uri = f"models:/{MLFLOW_REGISTERED_MODEL_NAME}/{MLFLOW_MODEL_VERSION}"
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Model '{MLFLOW_REGISTERED_MODEL_NAME}' version {MLFLOW_MODEL_VERSION} loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model from MLflow: {e}")
        raise RuntimeError(f"Could not load model: {e}")

@app.get("/")
async def read_root():
    return {"message": "Credit Risk Probability Model API is running. Go to /docs for API documentation."}

@app.post("/predict", response_model=PredictionOutput)
async def predict_risk(data: InputData):
    """
    Predict the credit risk probability for new customer transaction data.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please try again later.")

    try:
        # Convert input data to a pandas DataFrame matching model's expected input
        input_df = pd.DataFrame([data.model_dump()]) # Use model_dump() for Pydantic v2

        # The loaded MLflow model is a Pipeline that includes preprocessing.
        # So, we can directly pass the raw input_df.
        prediction_proba = model.predict_proba(input_df)[:, 1] # Probability of the positive class

        return PredictionOutput(risk_probability=prediction_proba[0])

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
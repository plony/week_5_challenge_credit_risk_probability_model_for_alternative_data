# src/api/pydantic_models.py

from pydantic import BaseModel, Field

# Define the input data model matching your model's features
class InputData(BaseModel):
    Amount: float = Field(..., description="Transaction amount")
    PricingStrategy: float = Field(..., description="Pricing strategy used for the transaction")
    CurrencyCode: str = Field(..., description="Currency code of the transaction")
    CountryCode: str = Field(..., description="Country code where the transaction occurred")
    ProviderId: str = Field(..., description="Identifier for the transaction provider")
    ProductId: str = Field(..., description="Identifier for the product involved")
    ProductCategory: str = Field(..., description="Category of the product")
    ChannelId: str = Field(..., description="Channel through which the transaction was made")

    class Config:
        # Example for documentation (optional, but good practice)
        schema_extra = {
            "example": {
                "Amount": 100.0,
                "PricingStrategy": 1.0,
                "CurrencyCode": "KES",
                "CountryCode": "KEN",
                "ProviderId": "ProviderA",
                "ProductId": "ProductX",
                "ProductCategory": "Category1",
                "ChannelId": "Channel1"
            }
        }

# Define the output data model for the prediction
class PredictionOutput(BaseModel):
    risk_probability: float = Field(..., description="Predicted probability of high risk (0 to 1)")
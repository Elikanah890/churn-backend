from fastapi import FastAPI
from pydantic import BaseModel, Field, validator
from typing import Literal
import pandas as pd
import joblib
from fastapi.middleware.cors import CORSMiddleware

# Load trained model
model = joblib.load("telco_churn_model.pkl")

app = FastAPI(
    title="Telco Churn Prediction API",
    version="1.0",
    description="Predict churn with rule-based guidance for user inputs"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Rule-based schema
class CustomerInput(BaseModel):
    gender: Literal["Male", "Female"] = Field(..., description="Select customer's gender: Male or Female")
    SeniorCitizen: int = Field(..., description="1 if senior citizen, else 0")
    Partner: Literal["Yes", "No"] = Field(..., description="Does the customer have a partner?")
    Dependents: Literal["Yes", "No"] = Field(..., description="Does the customer have dependents?")
    tenure: int = Field(..., description="Number of months the customer has been with the company")
    PhoneService: Literal["Yes", "No"] = Field(..., description="Does the customer have phone service?")
    MultipleLines: Literal["Yes", "No", "No phone service"] = Field(..., description="Multiple phone lines?")
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(..., description="Type of internet service")
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = Field(..., description="Online security enabled?")
    OnlineBackup: Literal["Yes", "No", "No internet service"] = Field(..., description="Online backup enabled?")
    DeviceProtection: Literal["Yes", "No", "No internet service"] = Field(..., description="Device protection enabled?")
    TechSupport: Literal["Yes", "No", "No internet service"] = Field(..., description="Tech support enabled?")
    StreamingTV: Literal["Yes", "No", "No internet service"] = Field(..., description="Streams TV?")
    StreamingMovies: Literal["Yes", "No", "No internet service"] = Field(..., description="Streams Movies?")
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(..., description="Contract type")
    PaperlessBilling: Literal["Yes", "No"] = Field(..., description="Paperless billing?")
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ] = Field(..., description="Payment method")
    MonthlyCharges: float = Field(..., description="Monthly charges")
    Speed: float = Field(..., description="Internet speed in Mbps")
    DataAllowance: float = Field(..., description="Data allowance in GB")
    TenureGroup: Literal["0-1yr", "1-2yr", "2-4yr", "4-6yr", "6+yr"] = Field(..., description="Tenure group")

# Prediction endpoint
@app.post("/predict_churn")
def predict_churn_api(data: CustomerInput):
    input_df = pd.DataFrame([data.dict()])

    # Predict using pre-trained model
    churn_prob = model.predict_proba(input_df)[:, 1][0]
    churn_class = model.predict(input_df)[0]

    return {
        "churn_probability": round(float(churn_prob), 2),
        "predicted_class": "Churn" if churn_class == 1 else "No Churn"
    }
#uvicorn main:app --reload

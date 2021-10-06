import uvicorn
from fastapi import FastAPI

import pickle
import joblib
from pydantic import BaseModel


def model_read(model_filename):
    dv, model = joblib.load(model_filename)
    return dv, model

def predict(customer, dv, model):
    dicts = dv.transform([customer])
    y_pred = model.predict_proba(dicts)[0, 1]
    return y_pred


app = FastAPI()

class Customer(BaseModel):
    gender: str
    seniorcitizen: int
    partner: str
    dependents: str
    phoneservice: str
    multiplelines: str
    onlinesecurity: str
    onlinebackup: str
    deviceprotection: str
    techsupport: str
    streamingtv: str
    streamingmovies: str
    contract: str
    paperlessbilling: str
    paymentmethod: str
    tenure: int
    monthlycharges: float
    totalcharges: float



@app.post("/predict/")
def ping(customer: Customer):
    model_filename = "Model_1.0.bin"
    customer = customer.dict()

    dv, model = model_read(model_filename)
    y_pred = predict(customer, dv, model)

    churn_decision = y_pred >= 0.5
    return {
        "churn": bool(churn_decision),
        "predicted_probability": float(y_pred)
    }

if __name__ == "__main__":
    uvicorn.run("ping:app", debug=True, host = "localhost", port = 5000, reload=True)

import pickle

import uvicorn

from fastapi import FastAPI
from pydantic import BaseModel


model_file = "model1.bin"
dv_file = "dv.bin"

def open_file(model_file, dv_file):
    with open(model_file, "rb") as model_f , open(dv_file, "rb") as dv_f:
        model = pickle.load(model_f)
        dv = pickle.load(dv_f)

    return model, dv

def predict(customer, model, dv):
    X_train = dv.transform([customer])
    y_pred = model.predict_proba(X_train)[0, 1]
    return y_pred


app = FastAPI()


class Customer(BaseModel):
    contract: str
    tenure: int
    monthlycharges: int


@app.post("/predict/")
def predict_model(customer: Customer):
    customer_dict = customer.dict()
    model, dv = open_file(model_file, dv_file)

    y_pred = predict(customer_dict, model, dv)

    return {
        "churn_probability": float(y_pred)
    }



if __name__ == "__main__":
    uvicorn.run(app, host = "0.0.0.0", port=5000)

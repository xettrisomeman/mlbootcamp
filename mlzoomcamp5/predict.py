import pickle



output_file = "Model_1.0.bin"

with open(output_file, "rb") as f_in:
    dv, model = pickle.load(f_in)



customer = {
    "gender": "female",
    "seniorcitizen": 0,
    "partner": "yes",
    "dependents": "no",
    "phoneservice": "no",
    "multiplelines": "no_phone_service",
    "onlinesecurity": "no",
    "onlinebackup": "yes",
    "deviceprotection": "no",
    "techsupport": "no",
    "streamingtv" : "no",
    "streamingmovies": "no",
    "contract": "month-to-month",
    "paperlessbilling": "yes",
    "paymentmethod": "electronic_check",
    "tenure": 1,
    "monthlycharges": 29.85,
    "totalcharges": 29.85
}

def predict(customer,dv, model):
    dicts = dv.transform([customer])
    y_pred = model.predict_proba(dicts)[0, 1]
    return y_pred

print(predict(customer, dv, model))

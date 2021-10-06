import pickle


model_file = "model1.bin"

with open(model_file, "rb") as f_in:
    model = pickle.load(f_in)

dv_file = "dv.bin"
with open(dv_file, "rb") as dv_in:
    dv = pickle.load(dv_in)


# predict
customer = {"contract": "two_year", "tenure": 12, "monthlycharges": 19.7}

X_val = dv.transform([customer])

y_pred = model.predict_proba(X_val)[0, 1]

print(y_pred)
# proba that customer is churning -> 0.115495


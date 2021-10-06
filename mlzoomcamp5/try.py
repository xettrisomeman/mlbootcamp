import requests

url = "http://localhost:2323/predict"

customer_id = "xyz-2021"
customer = {
    "gender": "male",
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
    "tenure": 24,
    "monthlycharges": 29.85,
    "totalcharges": (24 * 28.85)
}


post_response = requests.post(url, json=customer).json()
print(post_response)

if post_response['churn'] == True:
    print(f"Sending email to {customer_id}")
else:
    print(f"Not sending email to {customer_id}")

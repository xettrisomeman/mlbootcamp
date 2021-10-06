import requests


url = "http://localhost:5000/predict"


customer = {"contract": "two_year", "tenure": 1, "monthlycharges": 10}


response = requests.post(url, json=customer).json()
print(response)
# answer -> {'churn_probability': 0.9988892771007961}


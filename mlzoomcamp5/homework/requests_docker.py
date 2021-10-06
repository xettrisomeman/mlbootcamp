import requests

url = "http://localhost:8000/predict"


customer = {

    "contract": "two_year",
    "tenure": 12,
    "monthlycharges": 10
}

post_sequence = requests.post(url, json=customer).json()
# print(post_sequence)
# answer -> 0.3294


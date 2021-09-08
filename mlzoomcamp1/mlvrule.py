#rule based vs machine learning
import numpy as np
import pandas as pd



spamham = pd.read_csv("spamham.csv")

# rule based predictions
# if the length of the message is equal to or more than the first ham message
# then the message is ham
decision_boundary = len(spamham[spamham['target'] == 0]['data'][1])
def get_spam_or_ham(csv_data):
    total_ham = 0
    body = csv_data['data'].values
    for text in body:
        if len(text.strip()) >= decision_boundary:
            total_ham += 1
    return total_ham
print(get_spam_or_ham(spamham))


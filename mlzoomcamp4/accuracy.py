
# Accuracy and Dummy Model
import numpy as np
import matplotlib.pyplot as plt


from data_model import y_val, y_pred
from sklearn.metrics import accuracy_score

thresholds = np.linspace(0, 1, 21)
scores = []

for t in thresholds:
    score = accuracy_score(y_val, y_pred >= t)
    scores.append(score)
    print(f"{t:.2f} {score:.3f}")

plt.plot(thresholds, scores)
plt.savefig("accuracy.jpg")
plt.show()


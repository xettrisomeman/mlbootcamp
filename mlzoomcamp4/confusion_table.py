
from data_model import y_val, y_pred

import numpy as np



actual_positive = (y_val == 1)
actual_negative = (y_val == 0)

predicted_positive = y_pred >= 0.5
predicted_negative = y_pred < 0.5

tp = (predicted_positive & actual_positive).sum()
tn = (predicted_negative & actual_negative).sum()

fp = (predicted_positive & actual_negative).sum()
fn = (predicted_negative & actual_positive).sum()

confusion_matrix = np.array([
    [tp, fn],
    [fp, tn]
])

# print(confusion_matrix)

accuracy = (tp + tn) / (tp+fp+tn+fn)
# print(accuracy)



# 4.4 Precision , Recall and F1-score
from sklearn.metrics import roc_curve, auc


# precision -> true_positive / true_positive + false positive
precision = tp / (tp+fp)
# print(precision)

# recall -> true_positive / true_positive + false negative
# recall is also called TPR or sensitivity
recall = tp / (tp+fn)
# print(recall)

f1_score = (2 * precision * recall) / (precision + recall)
# print(f1_score)



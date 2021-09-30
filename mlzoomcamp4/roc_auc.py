from roc_curves import df_ideal, df_rand, df_scores
from roc_curves import y_val, y_pred, y_ideal, y_ideal_pred, y_rand

from sklearn.metrics import auc
# shortcut
from sklearn.metrics import roc_auc_score

# print "roc score curve (auc)"
# print(auc(df_scores.fpr, df_scores.tpr))


# print "roc ideal curve (auc)"
# print(auc(df_ideal.fpr, df_ideal.tpr))


# print(roc_auc_score(y_ideal, y_ideal_pred))

import random


neg = y_pred[y_val == 0]
# print(neg)
pos = y_pred[y_val == 1]
#print(pos)

pos_ind = random.randint(0, len(pos) - 1)
neg_ind = random.randint(0, len(neg) - 1)

# print(pos_ind, neg_ind)

# print(pos[pos_ind] > neg[neg_ind])
# print(neg[neg_ind])

print(neg[pos_ind])
print(neg[neg_ind])

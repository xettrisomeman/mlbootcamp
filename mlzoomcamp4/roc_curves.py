from confusion_table import tp, fp, tn, fn
from data_model import y_pred, y_val

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

#higher is better
tpr =  tp / (tp+fn)
# print(tpr)


# lower is better
fpr = fp / (fp + tn)
# print(fpr)

thresholds = np.linspace(0, 1, 101)

scores = []

for t in thresholds:
    actual_positive = (y_val == 1)
    actual_negative = (y_val == 0)

    predict_positive = (y_pred >= t)
    predict_negative = (y_pred < t)

    tp = (predict_positive & actual_positive).sum()
    tn = (predict_negative & actual_negative).sum()

    fp = (predict_positive & actual_negative).sum()
    fn = (predict_negative & actual_positive).sum()
    scores.append((t, tp, fp, fn, tn))


# print(scores)

columns = ["threshold", "true_positive", "false_positive", "false_negative", "true_negative"]
df_scores = pd.DataFrame(scores, columns=columns)

def calculate_tpr(df):
    tpr = df.true_positive / (df.true_positive + df.false_negative)
    return tpr

def calculate_fpr(df):
    fpr = df.false_positive / (df.true_negative + df.false_positive)
    return fpr


df_scores['tpr'] = calculate_tpr(df_scores)
df_scores['fpr'] = calculate_fpr(df_scores)
# print(df_scores.head())


# plt.plot(df_scores.threshold, df_scores.tpr, label="TPR")
# plt.plot(df_scores.threshold, df_scores.fpr, label="FPR")
# plt.legend()
# plt.savefig("tpr_fpr_y_pred.jpg")
# plt.show()


# Random Model

np.random.seed(42)
y_rand = np.random.uniform(0, 1, size = len(y_val))

y_rand = y_rand.round(3)

decision = (y_rand >= 0.5)

accuracy = (y_val == decision).mean()

# print(accuracy)

thresholds = np.linspace(0, 1, 101)


def tpr_fpr_dataframe(y_val, y_pred):
    scores = []
    for t in thresholds:
        actual_positive = (y_val == 1)
        actual_negative = (y_val == 0)

        predict_positive = (y_pred >= t)
        predict_negative = (y_pred < t)

        tp = (predict_positive & actual_positive).sum()
        tn = (predict_negative & actual_negative).sum()

        fp = (predict_positive & actual_negative).sum()
        fn = (predict_negative & actual_positive).sum()
        scores.append((t, tp, fp, fn, tn))

    columns = ["threshold", "true_positive", "false_positive", "false_negative", "true_negative"]
    df_scores = pd.DataFrame(scores, columns=columns)

    def calculate_tpr(df):
        tpr = df.true_positive / (df.true_positive + df.false_negative)
        return tpr

    def calculate_fpr(df):
        fpr = df.false_positive / (df.true_negative + df.false_positive)
        return fpr


    df_scores['tpr'] = calculate_tpr(df_scores)
    df_scores['fpr'] = calculate_fpr(df_scores)

    return df_scores


df_rand = tpr_fpr_dataframe(y_val, y_rand)


# plt.plot(df_rand.threshold, df_rand.tpr, label="TPR")
# plt.plot(df_rand.threshold, df_rand.fpr, label="FPR")
# plt.legend()
# plt.savefig("random_model_tpr_fpr.jpg")
# plt.show()


# Ideal Model

num_neg = (y_val == 0).sum()
num_pos = (y_val == 1).sum()

y_ideal = np.repeat([0,1], [num_neg, num_pos])

y_ideal_pred = np.linspace(0, 1, len(y_val))



def tpr_fpr_dataframe(y_val, y_pred):
    scores = []
    for t in thresholds:
        actual_positive = (y_val == 1)
        actual_negative = (y_val == 0)

        predict_positive = (y_pred >= t)
        predict_negative = (y_pred < t)

        tp = (predict_positive & actual_positive).sum()
        tn = (predict_negative & actual_negative).sum()

        fp = (predict_positive & actual_negative).sum()
        fn = (predict_negative & actual_positive).sum()
        scores.append((t, tp, fp, fn, tn))

    columns = ["threshold", "true_positive", "false_positive", "false_negative", "true_negative"]
    df_scores = pd.DataFrame(scores, columns=columns)

    def calculate_tpr(df):
        tpr = df.true_positive / (df.true_positive + df.false_negative)
        return tpr

    def calculate_fpr(df):
        fpr = df.false_positive / (df.true_negative + df.false_positive)
        return fpr


    df_scores['tpr'] = calculate_tpr(df_scores)
    df_scores['fpr'] = calculate_fpr(df_scores)

    return df_scores


df_ideal = tpr_fpr_dataframe(y_ideal, y_ideal_pred)

# plt.plot(df_ideal.threshold, df_ideal.tpr, label="TPR")
# plt.plot(df_ideal.threshold, df_ideal.fpr, label="FPR")
# plt.legend()
# plt.savefig("random_model_tpr_fpr.jpg")
# plt.show()



# plt.figure(figsize = (9, 6))
# plt.plot(df_scores.fpr, df_scores.tpr, label = "model")
# plt.plot(df_rand.fpr, df_rand.tpr, label = "random")
# plt.plot(df_ideal.fpr, df_ideal.tpr, label="ideal")

# plt.legend()
# plt.savefig("fpr_tpr.jpg")
# plt.show()


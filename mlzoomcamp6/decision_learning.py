import pandas as pd


data = [
        [8000, "default"],
        [5000, "ok"],
        [0, "default"],
        [5000, "ok"],
        [4000, "ok"], 
        [9000, "ok"],
        [3000, 'default'],
        [2000, "default"]
]


df_examples = pd.DataFrame(data, columns = ["assets", "status"]).sort_values("assets")

# print(df_examples)

# check T
# find best T
# Calculate impurity
# choose T which gives less averaged impurity
Ts = [0, 2000, 3000, 4000, 5000, 8000]

def calculate_t(Ts):
    for T in Ts:
        print(T)
        df_left = df_examples[df_examples.assets <= T]
        df_right = df_examples[df_examples.assets > T]

        print(df_left)
        print(df_left.status.value_counts(normalize=True))

        print(df_right)
        print(df_right.status.value_counts(normalize=True))



data = [
        [8000, 3000, "default"],
        [5000, 1000, "ok"],
        [0, 1000, "default"],
        [5000, 1000, "ok"],
        [4000, 1000, "ok"], 
        [9000, 1000, "ok"],
        [3000,500 ,'default'],
        [2000, 2000,"default"]
]

df_examples = pd.DataFrame(data, columns = ["assets", "debt", "status"])

# print(df_examples)


df_examples = df_examples.sort_values("debt")


# print(df_examples)


thresholds = {
        "assets": [
            0,
            2000, 
            3000,
            4000,
            5000,
            8000
            ],
        "debt":[
            500,
            1000,
            2000
            ]

}

for feature, Ts in thresholds.items():
    # print(feature, Ts)
    print("#########################")
    print(feature)
    for T in Ts:
        print(T)
        df_left = df_examples[df_examples[feature] <= T]
        df_right = df_examples[df_examples[feature] > T]

        print(df_left)
        print(df_left.status.value_counts(normalize=True))
        print(df_right)
        print(df_right.status.value_counts(normalize=True))

        print()
    print("########################")




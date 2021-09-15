import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("AB_NYC_2019.csv")


sns.histplot(df['price'], color="green")
plt.savefig("price_long_tail.png")
plt.show()

# Answer -> Yes it does have a long tail


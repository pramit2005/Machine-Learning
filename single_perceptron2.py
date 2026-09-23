import numpy as np
import pandas as pd

df = pd.read_csv("binary_10bit_with_value.csv")
df = df.drop_duplicates()

y = df["Decimal_Value"].to_numpy()
X = df.drop(columns="Decimal_Value").to_numpy()

weight = np.zeros(10)
epoch = 1000
alpha = 0.01

for k in range(epoch):
    for x, y_t in zip(X, y):
        output = np.dot(x, weight)
        error = y_t - output
        weight += alpha * error * x

print("Learned weights:", weight)

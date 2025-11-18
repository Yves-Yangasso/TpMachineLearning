from sklearn.datasets import load_diabetes
import pandas as pd
import numpy as np

diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target


df = pd.DataFrame(X, columns=diabetes.feature_names)

print("modification test")

print(df.head())

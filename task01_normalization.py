# Task 1: Data Normalization Techniques and Z-Score
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import matplotlib.pyplot as plt

# Load dataset (replace with your dataset path)
df = pd.read_csv("sales.csv")

print("Original Data:")
print(df.head())

# Normalization Techniques
scalers = {
    "MinMaxScaler": MinMaxScaler(),
    "StandardScaler (Z-Score)": StandardScaler(),
    "RobustScaler": RobustScaler(),
    "MaxAbsScaler": MaxAbsScaler()
}

for name, scaler in scalers.items():
    scaled = scaler.fit_transform(df.select_dtypes(include=[np.number]))
    scaled_df = pd.DataFrame(scaled, columns=df.select_dtypes(include=[np.number]).columns)
    print(f"\n{name} Result (first 5 rows):")
    print(scaled_df.head())
# Boxplot
    scaled_df.plot(kind="box", figsize=(10,6), title=f"{name} - Boxplot")
    plt.show()

# Mean Centering with Z-Score 
scaler = StandardScaler(with_mean=True, with_std=True)
zscore_centered = scaler.fit_transform(df.select_dtypes(include=[np.number]))
zscore_df = pd.DataFrame(zscore_centered, columns=df.select_dtypes(include=[np.number]).columns)

print("\nZ-Score Normalization with Mean Centering:")
print(zscore_df.head())
zscore_df.plot(kind="box", figsize=(10,6), title=f"Z-Score Normalization - Boxplot")
plt.show()
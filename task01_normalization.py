# Task 01 - Data Normalization & Z-Score

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("sales.csv")
numeric_cols = df.select_dtypes(include=[np.number]).columns

print("Original Data (first 5 rows):")
print(df.head())

# Define scalers
scalers = {
    "MinMaxScaler": MinMaxScaler(),
    "StandardScaler (Z-Score)": StandardScaler(),
    "RobustScaler": RobustScaler(),
    "MaxAbsScaler": MaxAbsScaler()
}

# Apply scalers and store scaled data
scaled_data = {}
for name, scaler in scalers.items():
    scaled = scaler.fit_transform(df[numeric_cols])
    scaled_df = pd.DataFrame(scaled, columns=numeric_cols)
    scaled_data[name] = scaled_df
    
    print(f"\n{name} (first 5 rows):")
    print(scaled_df.head())
    
    # Boxplot
    plt.figure(figsize=(10,6))
    sns.boxplot(data=scaled_df)
    plt.title(f"{name} - Boxplot")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Combined KDE plots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for i, (name, scaled_df) in enumerate(scaled_data.items()):
    for col in numeric_cols:
        sns.kdeplot(scaled_df[col], label=col, ax=axes[i], fill=True)
    axes[i].set_title(f"{name} - KDE Plot")
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Density")
    axes[i].legend()

plt.tight_layout()
plt.show()

# Z-Score Normalization with Mean Centering (Optional)
zscore_scaler = StandardScaler(with_mean=True, with_std=True)
zscore_data = zscore_scaler.fit_transform(df[numeric_cols])
zscore_df = pd.DataFrame(zscore_data, columns=numeric_cols)

print("\nZ-Score Normalization with Mean Centering (first 5 rows):")
print(zscore_df.head())

# Boxplot for Z-Score normalized data
plt.figure(figsize=(10,6))
sns.boxplot(data=zscore_df)
plt.title("Z-Score Normalization - Boxplot")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# KDE Plot for Z-Score normalized data
plt.figure(figsize=(10,6))
for col in numeric_cols:
    sns.kdeplot(zscore_df[col], label=col, fill=True)
plt.title("Z-Score Normalization - KDE Plot")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()

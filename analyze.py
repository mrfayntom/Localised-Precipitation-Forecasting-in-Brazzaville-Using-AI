import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('/content/drive/MyDrive/LPFB/Dataset/Train_data.csv')

print("Basic Info:")
print(df.info())
print("Sample Data:")
print(df.head())

plt.figure(figsize=(10, 5))
sns.histplot(df['Target'], bins=100, kde=True)
plt.title('Rainfall Target Distribution')
plt.xlabel('Rainfall (mm)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
sns.histplot(np.log1p(df['Target']), bins=100, kde=True, color='orange')
plt.title('Log-Scaled Rainfall Distribution')
plt.xlabel('log(1 + Rainfall)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 8))
c_mat = df.drop(['ID', 'DATE'], axis=1).corr()
sns.heatmap(c_mat, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Feature Correlation with Rainfall')
plt.show()

df['DATE'] = pd.to_datetime(df['DATE'])
df['Month'] = df['DATE'].dt.month

plt.figure(figsize=(10, 5))
sns.boxplot(x='Month', y='Target', data=df)
plt.title('Monthly Rainfall Trend')
plt.xlabel('Month')
plt.ylabel('Rainfall (mm)')
plt.grid(True)
plt.show()

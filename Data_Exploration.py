import pandas as pd
df = pd.read_csv('titanic.csv')
print("Shape of data:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())
df.head()
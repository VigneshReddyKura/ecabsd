import pandas as pd

df = pd.read_csv('data/splits.csv')
print("Before:")
print(df['split'].unique()[:10])
print("Total rows:", len(df))

# Assign train/val/test based on fold numbers
# 80% train, 10% val, 10% test
total = len(df)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

train_end = int(0.8 * total)
val_end = int(0.9 * total)

df['split'] = 'train'
df.loc[train_end:val_end, 'split'] = 'val'
df.loc[val_end:, 'split'] = 'test'

print("\nAfter:")
print(df['split'].value_counts())

df.to_csv('data/splits.csv', index=False)
print("\nSaved fixed splits.csv")
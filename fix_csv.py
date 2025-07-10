import pandas as pd

df = pd.read_csv("gesture_data.csv", header=None)
columns = [str(i) for i in range(42)] + ['label']
df.columns = columns
df.to_csv("gesture_data.csv", index=False)
print("✅ gesture_data.csv fixed with correct headers.")

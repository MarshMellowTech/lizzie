import pandas as pd

df = pd.read_csv("training_dataset.csv")
frame = df.groupby("group")
drop = []
for group in frame:
    num = len(group[1])
    if num < 5:
        drop.append(group[0])
df = df[df.group.isin(drop) == False]
df.to_csv("training_dataset.csv", index=False)

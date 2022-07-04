import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("https://raw.githubusercontent.com/nbertagnolli/counsel-chat/master/data/20200325_counsel_chat.csv")
vec = TfidfVectorizer(stop_words="english")
l = []
d = {}
for st, sti, qid in zip(df["questionTitle"], df["questionText"], df["questionID"]):
    stin = (st + sti).replace("\xa0", " ")
    l.append(stin)
    d[qid] = stin
s = set(l)
l = list(s)
vec.fit(np.asarray(l))
features = vec.transform(np.asarray(l))
clussy = KMeans(n_clusters=5)
clussy.fit(features)
clussy.predict(features)
lab = clussy.labels_
di = {}
for label, st in zip(lab, l):
    di[st] = label
subgroups = []
for row in df.iterrows():
    subgroups.append(row[1]["topic"] + "_" + str(di[d[row[1]["questionID"]]]))
df["subGroup"] = subgroups 
df[["questionTitle", "questionText", "answerText", "subGroup"]].to_csv("subgroups.csv")

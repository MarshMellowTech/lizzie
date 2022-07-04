import gensim.downloader
import numpy as np
import pandas as pd
import re
from sklearn.decomposition import KernelPCA
import spacy

nlp = spacy.load("en_core_web_lg")
embeddings = gensim.downloader.load("word2vec-google-news-300")

df = pd.read_csv("subgroups.csv")
d = {"question": []}
group = []
for i in range(50):
    d["vec_" + str(i)] = []
vecs = []
for qTitle, qText, g in zip(df["questionTitle"], df["questionText"], df["subGroup"]):
    s = qTitle + qText
    s = re.sub("\W+", " ", s.replace("\\", "").replace("\n", " ")).encode("ascii", "ignore").decode()
    if s in d["question"]:
        continue
    d["question"].append(s)
    group.append(g)
    doc = nlp(s)
    vec = np.zeros((300,), dtype=np.float64)
    i = 0
    for token in doc:
        if token.text.lower() in embeddings:
            vec += embeddings[token.text.lower()]
            i += 1
    vec /= i
    vecs.append(vec)

pca = KernelPCA(50, kernel="linear")
feats = pca.fit_transform(vecs)

for v in feats:
    for i in range(50):
        d["vec_" + str(i)].append(v[i])

d["group"] = group
frame = pd.DataFrame.from_dict(d)
frame = frame.drop("question", axis=1)
frame.to_csv("training_dataset.csv", index=False)

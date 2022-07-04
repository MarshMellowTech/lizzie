import pandas as pd
import spacy

nlp = spacy.load("en_core_web_lg")

df = pd.read_csv("subgroups.csv")

d = {"starters": [], "subGroup": []}
for aText, sg in zip(df["answerText"], df["subGroup"]):
    doc = nlp(aText)
    for sent in doc.sents:
        if (len(sent)) >= 4:
            d["starters"].append(" ".join([sent[i].text for i in range(4)]))
            d["subGroup"].append(sg)

temp = pd.read_csv("training_dataset.csv")
valid_groups = temp.group.tolist()

frame = pd.DataFrame.from_dict(d)
frame = frame[frame.subGroup.isin(valid_groups)]
frame.to_csv("answer_starters.csv")

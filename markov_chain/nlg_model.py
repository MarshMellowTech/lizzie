# Code referenced from: https://github.com/klaudia-nazarko/nlg-text-generation/blob/main/markov_chain.ipynb

from chain import Chain
from text import Text
import pandas as pd
import pickle

df = pd.read_csv("../preprocess/subgroups.csv")
data_text = ""
for answers in df["answerText"]:
    data_text += answers + "\n"
text = Text(data_text)
chain_model = Chain(text, n=3)
print(chain_model.generate_sequence("Unless you have", 50, 1))
with open("../models/markov_chain.pickle", "wb") as p:
    pickle.dump(chain_model, p)

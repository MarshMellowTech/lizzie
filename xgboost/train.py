import multiprocessing as mp
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

df = pd.read_csv("../preprocess/training_dataset.csv")
encoder = LabelEncoder()

X = df.iloc[:, 0:50]
y = encoder.fit_transform(df["group"])

xgb_model = xgb.XGBClassifier(n_jobs=mp.cpu_count() // 2)
gscv = GridSearchCV(xgb_model, {"max_depth": [2, 4, 6], "n_estimators": [50, 100, 200, 400]}, verbose=1, n_jobs=2)
gscv.fit(X, y)

print(gscv.best_score_)
print(gscv.best_params_)
model = gscv.best_estimator_

model.save_model("../models/xgboost.{}.json".format(gscv.best_score_ * 100 // 1))

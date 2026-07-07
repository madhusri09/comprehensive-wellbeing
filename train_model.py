import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("hdi_dataset.csv")

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

encoder = LabelEncoder()
y = encoder.fit_transform(y)

model = RandomForestClassifier()

model.fit(X, y)

pickle.dump(model, open("model/hdi_model.pkl", "wb"))
pickle.dump(encoder, open("model/label_encoder.pkl", "wb"))

print("Model Saved Successfully")
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("insurance.csv")

df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1})
df = pd.get_dummies(df, columns=['region'], drop_first=True)

X = df.drop('charges', axis=1)
y = df['charges']

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))

print("Model saved")
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("insurance.csv")

# Remove sex column completely
df = df.drop('sex', axis=1)

# Encode categorical variables
df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1})
df = pd.get_dummies(df, columns=['region'], drop_first=True)

# Features and target
X = df.drop('charges', axis=1)
y = df['charges']

# Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, y)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model saved")
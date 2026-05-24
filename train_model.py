import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset

df = pd.read_csv("Crop_recommendation.csv")

# Features and target
X = df.drop("label", axis=1)
y = df["label"]

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)

# Train model
model = RandomForestClassifier()
model.fit(x_train_scaled, y_train)

# Save model and scaler
joblib.dump(model, "crop_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model saved successfully")
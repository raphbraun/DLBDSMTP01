import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

# Load the uploaded CSV file
file_path = "./equipment_anomaly_data.csv"
df = pd.read_csv(file_path)

# Display the first few rows to understand the structure
df.head()

# Select features and target
features = ['temperature', 'vibration', 'humidity']
target = 'faulty'

X = df[features]
y = df[target]

# Handle any missing values (if present)
X = X.dropna()
y = y.loc[X.index]

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# Save model and scaler for later API use
joblib.dump(model, "./anomaly_model.pkl")
joblib.dump(scaler, "./scaler.pkl")

accuracy, report

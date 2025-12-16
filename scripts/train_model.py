import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# ✅ Load features CSV
df = pd.read_csv('data/features.csv')
print("Data loaded successfully!")
print(df.head())

# ✅ Features (9 total)
X = df[['dwell_mean', 'dwell_std', 'dwell_min', 'dwell_max',
        'flight_mean', 'flight_std', 'flight_min', 'flight_max', 'n_keys']]

# ✅ Target
y = df['user_id']

# ✅ Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ✅ Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ✅ Train model
svm_model = SVC(kernel='rbf', probability=True)
svm_model.fit(X_train_scaled, y_train)

# ✅ Evaluate
y_pred = svm_model.predict(X_test_scaled)
print("\n Model training complete!\n")
print(f" Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("Classification Report:\n", classification_report(y_test, y_pred))

# ✅ Save model & scaler
os.makedirs('models', exist_ok=True)
joblib.dump(svm_model, 'models/svm_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("\n Model and Scaler saved to 'models/' folder.")

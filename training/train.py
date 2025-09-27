# training/train.py
import os
import pickle
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(OUT_DIR, exist_ok=True)
MODEL_PATH = os.path.join(OUT_DIR, 'model.pkl')
SCALER_PATH = os.path.join(OUT_DIR, 'scaler.pkl')

# Load data (Iris is simple and reproducible)
data = load_iris()
X = data['data']  # 4 features
y = (data['target'] == 0).astype(int)  # make binary problem: class 0 vs others

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

# Train a linear model
model = LogisticRegression(max_iter=200).fit(X_train_s, y_train)

# Evaluate
preds = model.predict(X_test_s)
acc = accuracy_score(y_test, preds)
print(f"Test accuracy: {acc:.4f}")

# Save model + scaler
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)
with open(SCALER_PATH, 'wb') as f:
    pickle.dump(scaler, f)

print(f"Saved model -> {MODEL_PATH}")

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import os

print("Starting model training process...")

# Load dataset
file_path = "US accident datset.csv"
print(f"Loading dataset: {file_path}")
df = pd.read_csv(file_path)

print(f"Dataset loaded. Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")


print("Mapping Severity labels to numeric (1=Low, 2=Medium, 3=High)...")
severity_mapper = {'Low': 1, 'Medium': 2, 'High': 3}
df['Severity_Level'] = df['Severity_Level'].map(severity_mapper)
df['Severity_Level'].fillna(1, inplace=True) # fallback just in case
print(f"Severity distribution:\n{df['Severity_Level'].value_counts().sort_index()}")

# ─────────────────────────────────────────────────────────
# Features: new dataset columns
# ─────────────────────────────────────────────────────────
feature_cols = [
    'Road_Type', 'Road_Condition', 'Weather_Condition', 'Vehicle_Speed', 'Speed_Limit',
    'Vehicle_Type', 'Temperature', 'Humidity', 'Visibility',
    'T_Junction', 'Crossing', 'Railway_Crossing',
    'Stop_Signal', 'Speed_Breaker'
]

X = df[feature_cols].copy()
y = df['Severity_Level']

# ─────────────────────────────────────────────────────────
# Encode categorical columns
# ─────────────────────────────────────────────────────────
encoders = {}
categorical_cols = ['Road_Type', 'Road_Condition', 'Weather_Condition', 'Vehicle_Type']

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le
    print(f"  Encoded '{col}': {list(le.classes_)}")

print(f"\nFinal feature set ({len(feature_cols)} features): {feature_cols}")
print(f"Dataset shape for training: {X.shape}")

# ─────────────────────────────────────────────────────────
# Train / Test split & model training
# ─────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining Multiple Models for Comparison...")
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "XGBoost": xgb.XGBClassifier(eval_metric='mlogloss', random_state=42)
}

best_model = None
best_acc = 0
best_pred = None

for name, clf in models.items():
    if name == "XGBoost":
        clf.fit(X_train, y_train - 1)
        preds = clf.predict(X_test) + 1
    else:
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        
    acc = accuracy_score(y_test, preds)
    print(f"{name} Accuracy: {acc * 100:.2f}%")
    if acc > best_acc:
        best_acc = acc
        best_model = clf
        best_pred = preds

print("=" * 50)
print(f"Best Model Selected: {type(best_model).__name__} with {best_acc * 100:.2f}% Accuracy")
print(f"Total records used: {len(df)}")
print("\nClassification Report (Best Model):")
print(classification_report(y_test, best_pred, zero_division=0))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, best_pred))
print("=" * 50)

model = best_model

# ─────────────────────────────────────────────────────────
# Save model artifacts
# ─────────────────────────────────────────────────────────
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/model.pkl')
joblib.dump(feature_cols, 'model/features.pkl')
joblib.dump(encoders, 'model/encoders.pkl')

print("\nModel, features, and encoders saved to 'model/' directory.")
print("Training complete!")

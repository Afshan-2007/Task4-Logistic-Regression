import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve

# Load dataset
df = pd.read_csv("data.csv")

print("Dataset Preview:")
print(df.head())

# ----------------------------
# DATA CLEANING
# ----------------------------

# Remove useless column
df = df.drop(columns=['Unnamed: 32'], errors='ignore')

# Convert target column
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Drop ID column
df = df.drop('id', axis=1)

# Fill missing values (if any)
df = df.fillna(df.mean(numeric_only=True))

# ----------------------------
# FEATURES & TARGET
# ----------------------------
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# ----------------------------
# TRAIN TEST SPLIT
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# SCALING
# ----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----------------------------
# MODEL TRAINING
# ----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ----------------------------
# PREDICTIONS
# ----------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ----------------------------
# EVALUATION
# ----------------------------
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nROC-AUC Score:", roc_auc_score(y_test, y_prob))

# ----------------------------
# ROC CURVE
# ----------------------------
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.savefig("roc_curve.png")
plt.show()

print("\n✅ Task 4 Completed Successfully!")
# Logistic Regression Binary Classifier

# Step 1: Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve

# Step 2: Loading Dataset
# Replace 'data.csv' with your actual path
df = pd.read_csv(r'C:\Users\Lenovo\Downloads\data.csv')

# Step 3: Preprocess Data
df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)  # Drop irrelevant columns
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})  # Encode labels

X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Standardize Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Step 7: Predict and Evaluate
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Confusion Matrix and Classification Report
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ROC-AUC Score
roc_auc = roc_auc_score(y_test, y_prob)
print(f"\nROC-AUC Score: {roc_auc:.4f}")

# Step 8: Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label='Logistic Regression')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()

# Step 9: Threshold Tuning Example
threshold = 0.6
y_pred_custom = (y_prob >= threshold).astype(int)

print(f"\nCustom Threshold = {threshold}")
print("Confusion Matrix with custom threshold:")
print(confusion_matrix(y_test, y_pred_custom))
print("\nClassification Report with custom threshold:")
print(classification_report(y_test, y_pred_custom))
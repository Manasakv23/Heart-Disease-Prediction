# IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)
# LOAD DATASET

df = pd.read_csv("Heart Disease.csv")

print("Original Shape:", df.shape)
df = df.drop_duplicates()
print("After Removing Duplicates:", df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nData Types:")
print(df.dtypes)

# HANDLE MISSING VALUES

for col in df.columns:

    # Check if column is numeric
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())

    else:
        df[col] = df[col].fillna(df[col].mode()[0])

#ONE-HOT ENCODING
df = pd.get_dummies(df, drop_first=True)

# FEATURE SELECTION
X = df.drop('target', axis=1)
y = df['target']

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# FEATURE SCALING
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# LOGISTIC REGRESSION MODEL
model = LogisticRegression(
    C=0.1,
    max_iter=5000,
    random_state=42
)
model.fit(X_train, y_train)

# PREDICTIONS
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

#RESULTS
print("\nAccuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nROC-AUC:", round(roc_auc_score(y_test, y_prob), 3))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# CONFUSION MATRIX
plt.figure(figsize=(8,5))
sns.heatmap(
    confusion_matrix(y_test, y_pred),
    annot=True,
    fmt='d',
    cmap='Blues'
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#PLOTS
sns.countplot(x='target', data=df)
plt.title("Target Distribution")
plt.show()

sns.histplot(df['age'], kde=True)
plt.title("Age Distribution")
plt.show()

plt.plot(y_test.values[:20], label='Actual')
plt.plot(y_pred[:20], label='Predicted')
plt.title("Actual VS Predicted")
plt.legend()
plt.show()

# FEATURE IMPORTANCE
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": abs(model.coef_[0])
}).sort_values("Importance", ascending=False)
print("\nTop Features:")
print(importance.head(10))

plt.figure(figsize=(8,5))
sns.barplot(
    data=importance.head(10),
    x="Importance",
    y="Feature"
)

plt.title("Top 10 Important Features")
plt.show()

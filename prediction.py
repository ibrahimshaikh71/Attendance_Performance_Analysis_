import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("school_attendance_performance.csv")

# -------------------------
# 1. Regression: predict marks
# -------------------------
features = [
    "Attendance_Percentage",
    "Assignment_Score",
    "Study_Hours_Per_Day",
    "Previous_Marks"
]

X = df[features]
y = df["Average_Marks"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("--- LINEAR REGRESSION ---")
print(f"MAE: {mean_absolute_error(y_test, predictions):.2f}")
print(f"R² Score: {r2_score(y_test, predictions):.2f}")

# -------------------------
# 2. Classification: performance risk
# -------------------------
df["Performance_Risk"] = pd.cut(
    df["Average_Marks"],
    bins=[-1, 49.99, 69.99, 100],
    labels=["High Risk", "Medium Risk", "Low Risk"]
)

X = df[features]
y = df["Performance_Risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

classifier = DecisionTreeClassifier(max_depth=4, random_state=42)
classifier.fit(X_train, y_train)

risk_predictions = classifier.predict(X_test)

print("\n--- DECISION TREE RISK CLASSIFICATION ---")
print(f"Accuracy: {accuracy_score(y_test, risk_predictions):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, risk_predictions))

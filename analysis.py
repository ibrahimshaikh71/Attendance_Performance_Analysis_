import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_FILE = "school_attendance_performance.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

# -------------------------
# 1. Basic information
# -------------------------
print("\n--- DATASET INFO ---")
print(df.info())

print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

# -------------------------
# 2. Data cleaning
# -------------------------
df = df.drop_duplicates()

numeric_columns = [
    "Attendance_Percentage",
    "Maths_Marks",
    "Science_Marks",
    "English_Marks",
    "Assignment_Score",
    "Study_Hours_Per_Day",
    "Previous_Marks",
    "Average_Marks"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=numeric_columns)

# Recreate category after cleaning.
df["Attendance_Category"] = pd.cut(
    df["Attendance_Percentage"],
    bins=[0, 69.99, 79.99, 89.99, 100],
    labels=["Low", "Average", "Good", "Excellent"]
)

# -------------------------
# 3. Descriptive statistics
# -------------------------
print("\n--- DESCRIPTIVE STATISTICS ---")
print(df[numeric_columns].describe().round(2))

print("\n--- AVERAGE PERFORMANCE BY ATTENDANCE CATEGORY ---")
category_result = (
    df.groupby("Attendance_Category", observed=False)["Average_Marks"]
    .agg(["count", "mean", "median"])
    .round(2)
)
print(category_result)

print("\n--- CLASS-WISE ANALYSIS ---")
print(
    df.groupby("Class")[["Attendance_Percentage", "Average_Marks"]]
    .mean()
    .round(2)
)

print("\n--- SUBJECT-WISE AVERAGE ---")
print(df[["Maths_Marks", "Science_Marks", "English_Marks"]].mean().round(2))

# -------------------------
# 4. Correlation
# -------------------------
correlation = df["Attendance_Percentage"].corr(df["Average_Marks"])
print(f"\nAttendance vs Average Marks correlation: {correlation:.3f}")

corr_columns = [
    "Attendance_Percentage",
    "Assignment_Score",
    "Study_Hours_Per_Day",
    "Previous_Marks",
    "Average_Marks"
]

# -------------------------
# 5. Visualizations
# -------------------------

# Scatter plot
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="Attendance_Percentage",
    y="Average_Marks",
    hue="Attendance_Category"
)
sns.regplot(
    data=df,
    x="Attendance_Percentage",
    y="Average_Marks",
    scatter=False
)
plt.title("Attendance vs Academic Performance")
plt.xlabel("Attendance (%)")
plt.ylabel("Average Marks")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/attendance_vs_performance.png", dpi=300)
plt.close()

# Bar chart
category_means = (
    df.groupby("Attendance_Category", observed=False)["Average_Marks"]
    .mean()
    .reindex(["Low", "Average", "Good", "Excellent"])
)

plt.figure(figsize=(8, 5))
category_means.plot(kind="bar")
plt.title("Average Marks by Attendance Category")
plt.xlabel("Attendance Category")
plt.ylabel("Average Marks")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/performance_by_attendance_category.png", dpi=300)
plt.close()

# Subject performance
subject_means = df[["Maths_Marks", "Science_Marks", "English_Marks"]].mean()

plt.figure(figsize=(8, 5))
subject_means.plot(kind="bar")
plt.title("Average Subject Performance")
plt.xlabel("Subject")
plt.ylabel("Average Marks")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/subject_performance.png", dpi=300)
plt.close()

# Class-wise attendance
class_attendance = df.groupby("Class")["Attendance_Percentage"].mean().sort_index()

plt.figure(figsize=(8, 5))
class_attendance.plot(kind="bar")
plt.title("Average Attendance by Class")
plt.xlabel("Class")
plt.ylabel("Attendance (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/class_attendance.png", dpi=300)
plt.close()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    df[corr_columns].corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png", dpi=300)
plt.close()

# -------------------------
# 6. At-risk students
# -------------------------
at_risk = df[
    (df["Attendance_Percentage"] < 70) |
    (df["Average_Marks"] < 50)
].copy()

at_risk.to_csv(f"{OUTPUT_DIR}/at_risk_students.csv", index=False)

print(f"\nAt-risk students identified: {len(at_risk)}")
print(f"Charts and results saved in: {OUTPUT_DIR}/")

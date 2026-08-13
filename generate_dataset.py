import numpy as np
import pandas as pd

np.random.seed(42)

n = 300
classes = np.random.choice(["8th", "9th", "10th"], n)
gender = np.random.choice(["Male", "Female"], n)
attendance = np.clip(np.random.normal(82, 10, n), 45, 100)
assignment = np.clip(np.random.normal(75, 12, n), 30, 100)
study_hours = np.clip(np.random.normal(2.8, 1.2, n), 0.5, 8)
previous_marks = np.clip(np.random.normal(70, 12, n), 30, 100)

# Performance has a positive association with attendance, assignments,
# study hours and previous marks, plus some random variation.
average_marks = (
    0.32 * attendance
    + 0.25 * assignment
    + 3.2 * study_hours
    + 0.25 * previous_marks
    + np.random.normal(0, 7, n)
)
average_marks = np.clip(average_marks, 20, 100)

maths = np.clip(average_marks + np.random.normal(0, 7, n), 0, 100)
science = np.clip(average_marks + np.random.normal(0, 8, n), 0, 100)
english = np.clip(average_marks + np.random.normal(0, 6, n), 0, 100)

df = pd.DataFrame({
    "Student_ID": [f"ST{i:03d}" for i in range(1, n + 1)],
    "Class": classes,
    "Gender": gender,
    "Attendance_Percentage": np.round(attendance, 1),
    "Maths_Marks": np.round(maths, 1),
    "Science_Marks": np.round(science, 1),
    "English_Marks": np.round(english, 1),
    "Assignment_Score": np.round(assignment, 1),
    "Study_Hours_Per_Day": np.round(study_hours, 1),
    "Previous_Marks": np.round(previous_marks, 1),
    "Average_Marks": np.round((maths + science + english) / 3, 1)
})

df["Attendance_Category"] = pd.cut(
    df["Attendance_Percentage"],
    bins=[0, 69.99, 79.99, 89.99, 100],
    labels=["Low", "Average", "Good", "Excellent"]
)

df.to_csv("school_attendance_performance.csv", index=False)
print("Dataset created: school_attendance_performance.csv")
print(df.head())

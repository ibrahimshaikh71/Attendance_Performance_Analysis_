import io
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(
    page_title="Attendance Performance Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_FILE = "school_attendance_performance.csv"
NUMERIC_COLUMNS = [
    "Attendance_Percentage",
    "Maths_Marks",
    "Science_Marks",
    "English_Marks",
    "Assignment_Score",
    "Study_Hours_Per_Day",
    "Previous_Marks",
    "Average_Marks",
]


@st.cache_data
def generate_sample_data(n: int = 300) -> pd.DataFrame:
    np.random.seed(42)

    classes = np.random.choice(["8th", "9th", "10th"], n)
    gender = np.random.choice(["Male", "Female"], n)
    attendance = np.clip(np.random.normal(82, 10, n), 45, 100)
    assignment = np.clip(np.random.normal(75, 12, n), 30, 100)
    study_hours = np.clip(np.random.normal(2.8, 1.2, n), 0.5, 8)
    previous_marks = np.clip(np.random.normal(70, 12, n), 30, 100)

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

    df = pd.DataFrame(
        {
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
            "Average_Marks": np.round((maths + science + english) / 3, 1),
        }
    )
    df["Attendance_Category"] = pd.cut(
        df["Attendance_Percentage"],
        bins=[0, 69.99, 79.99, 89.99, 100],
        labels=["Low", "Average", "Good", "Excellent"],
    )
    return df


@st.cache_data
def load_data(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.read_csv(DATA_FILE)

    file_name = uploaded_file.name.lower()
    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if file_name.endswith(('.xls', '.xlsx')):
        return pd.read_excel(uploaded_file)

    raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().copy()
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=NUMERIC_COLUMNS)
    df["Attendance_Category"] = pd.cut(
        df["Attendance_Percentage"],
        bins=[0, 69.99, 79.99, 89.99, 100],
        labels=["Low", "Average", "Good", "Excellent"],
    )
    return df


def build_regression_model(df: pd.DataFrame) -> tuple:
    features = [
        "Attendance_Percentage",
        "Assignment_Score",
        "Study_Hours_Per_Day",
        "Previous_Marks",
    ]
    X = df[features]
    y = df["Average_Marks"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return mae, r2


def build_risk_classifier(df: pd.DataFrame) -> tuple:
    df = df.copy()
    df["Performance_Risk"] = pd.cut(
        df["Average_Marks"],
        bins=[-1, 49.99, 69.99, 100],
        labels=["High Risk", "Medium Risk", "Low Risk"],
    )
    features = [
        "Attendance_Percentage",
        "Assignment_Score",
        "Study_Hours_Per_Day",
        "Previous_Marks",
    ]
    X = df[features]
    y = df["Performance_Risk"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    risk_predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, risk_predictions)
    report = classification_report(y_test, risk_predictions, zero_division=0)
    return accuracy, report


def render_charts(df: pd.DataFrame):
    sns.set_style("whitegrid")

    corr_columns = [
        "Attendance_Percentage",
        "Assignment_Score",
        "Study_Hours_Per_Day",
        "Previous_Marks",
        "Average_Marks",
    ]

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(
            data=df,
            x="Attendance_Percentage",
            y="Average_Marks",
            hue="Attendance_Category",
            palette="viridis",
            ax=ax,
        )
        sns.regplot(
            data=df,
            x="Attendance_Percentage",
            y="Average_Marks",
            scatter=False,
            ax=ax,
            color="black",
        )
        ax.set_title("Attendance vs Academic Performance")
        ax.set_xlabel("Attendance (%)")
        ax.set_ylabel("Average Marks")
        st.pyplot(fig)

    with col2:
        category_means = (
            df.groupby("Attendance_Category", observed=False)["Average_Marks"]
            .mean()
            .reindex(["Low", "Average", "Good", "Excellent"])
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        category_means.plot(kind="bar", ax=ax, color="#4c72b0")
        ax.set_title("Average Marks by Attendance Category")
        ax.set_xlabel("Attendance Category")
        ax.set_ylabel("Average Marks")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        st.pyplot(fig)

    col3, col4 = st.columns(2)
    with col3:
        subject_means = df[["Maths_Marks", "Science_Marks", "English_Marks"]].mean()
        fig, ax = plt.subplots(figsize=(8, 5))
        subject_means.plot(kind="bar", ax=ax, color=["#ff7f0e", "#2ca02c", "#d62728"])
        ax.set_title("Average Subject Performance")
        ax.set_xlabel("Subject")
        ax.set_ylabel("Average Marks")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        st.pyplot(fig)

    with col4:
        class_attendance = df.groupby("Class")["Attendance_Percentage"].mean().sort_index()
        fig, ax = plt.subplots(figsize=(8, 5))
        class_attendance.plot(kind="bar", ax=ax, color="#9467bd")
        ax.set_title("Average Attendance by Class")
        ax.set_xlabel("Class")
        ax.set_ylabel("Attendance (%)")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        df[corr_columns].corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax,
    )
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)


def main():
    st.title("School Attendance & Performance Dashboard")
    st.markdown(
        "Use this dashboard to explore attendance and academic performance trends, identify at-risk students, and test predictions using your own dataset or a sample dataset."
    )

    st.markdown(
        """
        <style>
        .stApp { background-color: #f7f9fc; }
        .block-container { padding: 1.5rem 2rem; }
        .stButton>button { background-color: #4c72b0; color: white; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.header("Data Controls")
    st.sidebar.write(
        "Upload your dataset in CSV format, or use the built-in sample dataset for quick exploration."
    )
    uploaded_file = st.sidebar.file_uploader(
        "Upload dataset from your device",
        type=["csv", "xls", "xlsx"],
        help="Select a CSV or Excel file with attendance and performance records.",
    )
    use_sample = st.sidebar.button("Use sample dataset")
    analyze_button = st.sidebar.button("Analyze data")

    with st.container():
        st.subheader("Upload Data")
        col_left, col_right = st.columns([2, 1])
        with col_left:
            uploaded_file_main = st.file_uploader(
                "Upload dataset from your device",
                type=["csv", "xls", "xlsx"],
                key="main_uploader",
                help="Upload a CSV or Excel file with columns like Attendance_Percentage, Average_Marks, and related student metrics.",
            )
        with col_right:
            st.write("**Quick actions**")
            if st.button("Use sample dataset", key="main_sample"):
                use_sample = True
            if st.button("Analyze data", key="main_analyze"):
                analyze_button = True

    if uploaded_file_main is not None:
        uploaded_file = uploaded_file_main

    if uploaded_file is not None and analyze_button:
        try:
            df = load_data(uploaded_file)
            st.success("Data uploaded and loaded successfully.")
        except Exception as exc:
            st.error(f"Failed to read uploaded file: {exc}")
            return
    elif use_sample:
        df = generate_sample_data()
        st.success("Sample dataset created.")
    else:
        st.info("Upload a CSV file or click 'Use sample dataset' to begin.")
        return

    df = clean_data(df)

    st.markdown("---")
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    row_col1, row_col2, row_col3 = st.columns(3)
    row_col1.metric("Rows", df.shape[0])
    row_col2.metric("Columns", df.shape[1])
    row_col3.metric(
        "At-risk students",
        int(((df["Attendance_Percentage"] < 70) | (df["Average_Marks"] < 50)).sum()),
    )

    with st.expander("Dataset summary and missing values"):
        st.write(df.describe(include="all"))
        st.write("### Missing values")
        st.write(df.isnull().sum())

    st.markdown("---")
    st.subheader("Analysis Results")
    category_result = (
        df.groupby("Attendance_Category", observed=False)["Average_Marks"]
        .agg(["count", "mean", "median"])
        .round(2)
    )
    st.write("### Average Performance by Attendance Category")
    st.dataframe(category_result)

    st.write("### Class-wise Attendance and Average Marks")
    st.dataframe(
        df.groupby("Class")[ ["Attendance_Percentage", "Average_Marks"] ]
        .mean()
        .round(2),
        use_container_width=True,
    )

    st.write("### Subject-wise Averages")
    st.dataframe(df[["Maths_Marks", "Science_Marks", "English_Marks"]].mean().round(2))

    correlation = df["Attendance_Percentage"].corr(df["Average_Marks"])
    st.info(f"Attendance vs Average Marks correlation: {correlation:.3f}")

    render_charts(df)

    at_risk = df[(df["Attendance_Percentage"] < 70) | (df["Average_Marks"] < 50)].copy()
    st.subheader("At-risk Students")
    st.dataframe(at_risk)

    csv_buffer = io.StringIO()
    at_risk.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download at-risk students",
        data=csv_buffer.getvalue(),
        file_name="at_risk_students.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.subheader("Prediction Models")
    mae, r2 = build_regression_model(df)
    accuracy, report = build_risk_classifier(df)
    st.write(f"**Regression MAE:** {mae:.2f}")
    st.write(f"**Regression R² Score:** {r2:.2f}")
    st.write(f"**Risk classifier accuracy:** {accuracy:.2f}")
    st.text(report)


if __name__ == "__main__":
    main()

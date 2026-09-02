import io
import re

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_fscore_support,
    r2_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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


def normalize_subject_name(column_name: str) -> str:
    base_name = column_name.replace("_Marks", "")
    aliases = {
        "maths": "Math",
        "science": "Science",
        "english": "English",
        "geography": "Geography",
    }
    normalized = aliases.get(base_name.lower(), base_name.replace("_", " ").title())
    return normalized


def get_subject_options(df: pd.DataFrame) -> dict:
    subject_columns = []
    for column in df.columns:
        if column.endswith("_Marks") and column not in {"Average_Marks"}:
            subject_columns.append(column)

    return {normalize_subject_name(column): column for column in subject_columns}


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
    if file_name.endswith((".xls", ".xlsx")):
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
    if df.empty or len(df) < 3:
        return 0.0, 0.0

    features = [
        "Attendance_Percentage",
        "Assignment_Score",
        "Study_Hours_Per_Day",
        "Previous_Marks",
    ]
    X = df[features]
    y = df["Average_Marks"]

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    except ValueError:
        return 0.0, 0.0

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return mae, r2


def create_risk_labels(df: pd.DataFrame) -> pd.DataFrame:
    risk_df = df.copy()
    risk_df["Performance_Risk"] = pd.cut(
        risk_df["Average_Marks"],
        bins=[-1, 49.99, 69.99, 100],
        labels=["High Risk", "Medium Risk", "Low Risk"],
    )
    return risk_df


def build_risk_model_evaluation(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < 3:
        return {"status": "warning", "message": "Not enough data to build a risk classifier."}

    risk_df = create_risk_labels(df)
    if risk_df["Performance_Risk"].isna().all() or risk_df["Performance_Risk"].nunique() < 2:
        return {"status": "warning", "message": "Not enough risk categories after filtering."}

    features = [
        "Attendance_Percentage",
        "Assignment_Score",
        "Study_Hours_Per_Day",
        "Previous_Marks",
        "Class",
        "Gender",
    ]
    X = risk_df[features].copy()
    y = risk_df["Performance_Risk"]
    label_order = ["High Risk", "Medium Risk", "Low Risk"]

    numeric_features = [
        "Attendance_Percentage",
        "Assignment_Score",
        "Study_Hours_Per_Day",
        "Previous_Marks",
    ]
    categorical_features = ["Class", "Gender"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    if y.value_counts().min() >= 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        return {
            "status": "warning",
            "message": "Not enough class diversity after filtering. Model evaluation is not meaningful for this subset.",
        }

    class_distribution_before = y.value_counts().reindex(label_order, fill_value=0)
    class_distribution_train = y_train.value_counts().reindex(label_order, fill_value=0)
    class_distribution_test = y_test.value_counts().reindex(label_order, fill_value=0)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=2,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            random_state=42,
            learning_rate=0.08,
            max_depth=4,
            max_leaf_nodes=31,
            l2_regularization=0.1,
        ),
    }

    results = []
    for model_name, model in models.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

        cv_splits = min(3, int(y_train.value_counts().min()))
        if cv_splits >= 2:
            cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
            scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                cv=cv,
                scoring="f1_macro",
            )
            cv_macro_f1 = float(np.mean(scores))
        else:
            cv_macro_f1 = 0.0

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, labels=label_order, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, labels=label_order, average="weighted", zero_division=0)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test,
            y_pred,
            labels=label_order,
            average=None,
            zero_division=0,
        )

        results.append(
            {
                "model": model_name,
                "cv_macro_f1": cv_macro_f1,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
                "pipeline": pipeline,
            }
        )

    best_result = max(results, key=lambda item: (item["cv_macro_f1"], item["macro_f1"]))
    best_pipeline = best_result["pipeline"]
    y_pred_best = best_pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best, labels=label_order)
    metric_df = pd.DataFrame(
        {
            "Class": label_order,
            "Precision": np.round(best_result["precision"], 2),
            "Recall": np.round(best_result["recall"], 2),
            "F1": np.round(best_result["f1"], 2),
            "Support": best_result["support"],
        }
    )
    summary_df = pd.DataFrame(results)
    summary_df["accuracy"] = summary_df["accuracy"].round(2)
    summary_df["macro_f1"] = summary_df["macro_f1"].round(2)
    summary_df["weighted_f1"] = summary_df["weighted_f1"].round(2)
    summary_df["cv_macro_f1"] = summary_df["cv_macro_f1"].round(2)

    return {
        "status": "ok",
        "label_order": label_order,
        "distribution_before": class_distribution_before,
        "distribution_train": class_distribution_train,
        "distribution_test": class_distribution_test,
        "best_model": best_result["model"],
        "best_pipeline": best_pipeline,
        "best_metrics": {
            "accuracy": round(float(best_result["accuracy"]), 2),
            "macro_f1": round(float(best_result["macro_f1"]), 2),
            "weighted_f1": round(float(best_result["weighted_f1"]), 2),
            "support": int(best_result["support"].sum()),
        },
        "results_df": summary_df[["model", "cv_macro_f1", "accuracy", "macro_f1", "weighted_f1"]].copy(),
        "class_metrics": metric_df,
        "confusion_matrix": cm,
        "confusion_matrix_labels": label_order,
        "classification_report": classification_report(
            y_test,
            y_pred_best,
            labels=label_order,
            target_names=label_order,
            digits=2,
            zero_division=0,
        ),
        "high_risk_warning": (
            "High Risk is a minority class in this dataset and may be difficult to learn reliably. "
            "This is a data imbalance issue, not a model bug."
            if class_distribution_before["High Risk"] < 20
            else ""
        ),
    }


def add_serial_column(df: pd.DataFrame) -> pd.DataFrame:
    preview_df = df.copy().reset_index(drop=True)
    preview_df.insert(0, "Sr.", range(1, len(preview_df) + 1))
    return preview_df


def class_sort_key(value):
    text = str(value).strip()
    match = re.search(r"(\d+)", text)
    if match:
        return (0, int(match.group(1)), text)
    return (1, 0, text)


def render_charts(df: pd.DataFrame, selected_subject: str = "All Subjects"):
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
        subject_map = get_subject_options(df)
        if selected_subject == "All Subjects":
            subject_means = pd.Series(
                {normalize_subject_name(col): df[col].mean() for col in subject_map.values()}
            )
            subject_means = subject_means.sort_index()
            fig, ax = plt.subplots(figsize=(8, 5))
            subject_means.plot(
                kind="bar",
                ax=ax,
                color=["#ff7f0e", "#2ca02c", "#d62728"][: len(subject_means)],
            )
            ax.set_title("Average Subject Performance")
            ax.set_xlabel("Subject")
            ax.set_ylabel("Average Marks")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        else:
            selected_col = subject_map[selected_subject]
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(df[selected_col], bins=10, kde=True, ax=ax, color="#1f77b4")
            ax.set_title(f"{selected_subject} Score Distribution")
            ax.set_xlabel(f"{selected_subject} Marks")
            ax.set_ylabel("Students")
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
        """
        <div class="dashboard-hero">
            <div style="font-size: 1.1rem; font-weight: 600; color: #0f172a;">Academic overview and student risk monitoring</div>
            <div style="color: #475569; margin-top: 0.35rem;">Filter by class and subject to review attendance performance, subject trends, and student risk levels in one place.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <style>
        .stApp { background: linear-gradient(180deg, #edf5ff 0%, #f8fbff 35%, #f1f5f9 100%); }
        .block-container { padding: 1.5rem 2rem 2.5rem; }
        [data-testid="stSidebar"] { background: linear-gradient(180deg, #f8fbff 0%, #eef4ff 100%); border-right: 1px solid #dfe9f7; }
        .stButton>button {
            background: linear-gradient(135deg, #2563eb, #1d4ed8);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.65rem 1.2rem;
            font-weight: 600;
            box-shadow: 0 10px 22px rgba(37, 99, 235, 0.18);
        }
        .stButton>button:hover { background: linear-gradient(135deg, #1d4ed8, #1e40af); }
        .stDataFrame { border-radius: 16px; overflow: hidden; border: 1px solid #e6edf7; }
        .stSelectbox > div > div { border-radius: 10px; }
        .stMetric { background: rgba(255,255,255,0.95); border: 1px solid #edf2f7; border-radius: 14px; padding: 0.9rem; box-shadow: 0 8px 20px rgba(15,23,42,0.04); }
        .stAlert, .stInfo, .stSuccess, .stWarning, .stError { border-radius: 12px; }
        .stTabs [role="tablist"] { gap: 0.5rem; }
        .stTabs [role="tab"] { border-radius: 10px 10px 0 0; }
        .dashboard-hero {
            background: linear-gradient(135deg, rgba(37,99,235,0.08), rgba(16,185,129,0.08));
            border: 1px solid rgba(148,163,184,0.25);
            border-radius: 18px;
            padding: 1.2rem 1.4rem;
            margin-bottom: 1rem;
            box-shadow: 0 10px 24px rgba(15,23,42,0.04);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.header("Data Controls")
    st.sidebar.write("Choose a dataset or upload your own file to begin analysis.")

    uploaded_file = st.sidebar.file_uploader(
        "Upload dataset from your device",
        type=["csv", "xls", "xlsx"],
        help="Select a CSV or Excel file with attendance and performance records.",
    )

    if uploaded_file is not None:
        try:
            df = load_data(uploaded_file)
            st.sidebar.success("Dataset uploaded successfully.")
            st.session_state["df"] = df
        except Exception as exc:
            st.sidebar.error(f"Failed to read uploaded file: {exc}")
            return
    elif st.sidebar.button("Use sample dataset", key="sample_data"):
        st.session_state["df"] = generate_sample_data()
        st.sidebar.success("Sample dataset loaded successfully.")

    if "df" not in st.session_state:
        st.info("Upload a CSV or Excel file, or click 'Use sample dataset' to begin.")
        return

    df = clean_data(st.session_state["df"])

    with st.sidebar:
        st.markdown("---")
        st.subheader("Filters")
        class_options = ["All Classes"] + sorted(
            df["Class"].dropna().astype(str).unique().tolist(), key=class_sort_key
        )
        subject_map = get_subject_options(df)
        subject_options = ["All Subjects"] + list(subject_map.keys())

        selected_class = st.selectbox("Class", class_options, index=0, key="active_class")
        selected_subject = st.selectbox("Subject", subject_options, index=0, key="active_subject")

    filtered_df = df.copy()
    if selected_class != "All Classes":
        filtered_df = filtered_df[filtered_df["Class"].astype(str) == selected_class]
    if selected_subject != "All Subjects":
        filtered_df = filtered_df[filtered_df[subject_map[selected_subject]].notna()].copy()

    if filtered_df.empty:
        st.warning(
            "No records match the current class and subject filters. Try choosing a broader filter or switching back to All Classes / All Subjects."
        )
        return

    st.markdown("---")
    st.subheader("Dataset Preview")
    preview_df = add_serial_column(filtered_df.head(10)).copy()
    st.dataframe(preview_df, use_container_width=True)

    row_col1, row_col2, row_col3 = st.columns(3)
    row_col1.metric("Rows", filtered_df.shape[0])
    row_col2.metric("Columns", filtered_df.shape[1])
    row_col3.metric(
        "At-risk students",
        int(((filtered_df["Attendance_Percentage"] < 70) | (filtered_df["Average_Marks"] < 50)).sum()),
    )

    with st.expander("Dataset summary and missing values"):
        st.write(filtered_df.describe(include="all"))
        st.write("### Missing values")
        st.write(filtered_df.isnull().sum())

    st.markdown("---")
    st.subheader("Analysis Results")
    category_result = (
        filtered_df.groupby("Attendance_Category", observed=False)["Average_Marks"]
        .agg(["count", "mean", "median"])
        .round(2)
    )
    st.write("### Average Performance by Attendance Category")
    st.dataframe(category_result)

    st.write("### Class-wise Attendance and Average Marks")
    st.dataframe(
        filtered_df.groupby("Class")[["Attendance_Percentage", "Average_Marks"]].mean().round(2),
        use_container_width=True,
    )

    if selected_subject == "All Subjects":
        subject_summary = filtered_df[
            [col for col in filtered_df.columns if col.endswith("_Marks") and col != "Average_Marks"]
        ].mean().round(2)
        subject_summary = subject_summary.rename(index=lambda col: normalize_subject_name(col))
    else:
        subject_summary = filtered_df[[subject_map[selected_subject]]].mean().round(2)
        subject_summary = subject_summary.rename(index=lambda col: normalize_subject_name(col))
    st.write("### Subject-wise Averages")
    st.dataframe(subject_summary)

    correlation = filtered_df["Attendance_Percentage"].corr(filtered_df["Average_Marks"])
    st.info(f"Attendance vs Average Marks correlation: {correlation:.3f}")

    render_charts(filtered_df, selected_subject)

    at_risk = filtered_df[(filtered_df["Attendance_Percentage"] < 70) | (filtered_df["Average_Marks"] < 50)].copy()
    at_risk = create_risk_labels(at_risk)
    at_risk_filters = ["All Risk Levels", "High Risk", "Medium Risk", "Low Risk"]
    selected_risk_level = st.selectbox("Risk Level", at_risk_filters, index=0, key="at_risk_level")
    if selected_risk_level != "All Risk Levels":
        at_risk = at_risk[at_risk["Performance_Risk"] == selected_risk_level].copy()

    st.subheader("At-risk Students")
    st.dataframe(add_serial_column(at_risk), use_container_width=True)

    csv_buffer = io.StringIO()
    at_risk.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download at-risk students",
        data=csv_buffer.getvalue(),
        file_name="at_risk_students.csv",
        mime="text/csv",
    )

    # Internal model evaluation remains in code for future refinement but is hidden from the public dashboard.
    _ = build_risk_model_evaluation(filtered_df)
    _ = build_regression_model(filtered_df)


if __name__ == "__main__":
    main()
>>>>>>> 258590e (Improve dashboard UI and add risk-level filtering)

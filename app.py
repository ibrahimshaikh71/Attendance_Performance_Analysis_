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

DATA_FILE = "school attendance performance.csv"
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


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    canonical_names = {
        name.lower(): name
        for name in [
            "Student_ID",
            "Student_Name",
            "Student_Roll_Number",
            "Class",
            "Gender",
            *NUMERIC_COLUMNS,
        ]
    }
    column_aliases = {
        "student_roll_no": "Student_Roll_Number",
        "student_roll_number": "Student_Roll_Number",
    }
    renamed_columns = {}
    for column in df.columns:
        normalized = re.sub(r"[^0-9A-Za-z]+", "_", str(column).strip()).strip("_")
        normalized = re.sub(r"_+", "_", normalized)
        renamed_columns[column] = canonical_names.get(
            normalized.lower(), column_aliases.get(normalized.lower(), normalized)
        )
    return df.rename(columns=renamed_columns)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_column_names(df).drop_duplicates().copy()
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
    sns.set_theme(
        style="darkgrid",
        rc={
            "axes.facecolor": "#101936",
            "figure.facecolor": "#0a1028",
            "axes.edgecolor": "#26345e",
            "grid.color": "#25325a",
            "text.color": "#dbe7ff",
            "axes.labelcolor": "#9fb2d9",
            "xtick.color": "#9fb2d9",
            "ytick.color": "#9fb2d9",
        },
    )

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
            palette={"Low": "#ff6b78", "Average": "#ffb454", "Good": "#32d6c0", "Excellent": "#8d72ff"},
            ax=ax,
        )
        sns.regplot(
            data=df,
            x="Attendance_Percentage",
            y="Average_Marks",
            scatter=False,
            ax=ax,
            color="#f5f7ff",
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
        category_means.plot(kind="bar", ax=ax, color=["#ff6b78", "#ffb454", "#32d6c0", "#8d72ff"])
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
                color=["#32d6c0", "#8d72ff", "#ffb454"][: len(subject_means)],
            )
            ax.set_title("Average Subject Performance")
            ax.set_xlabel("Subject")
            ax.set_ylabel("Average Marks")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        else:
            selected_col = subject_map[selected_subject]
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(df[selected_col], bins=10, kde=True, ax=ax, color="#32d6c0")
            ax.set_title(f"{selected_subject} Score Distribution")
            ax.set_xlabel(f"{selected_subject} Marks")
            ax.set_ylabel("Students")
        st.pyplot(fig)

    with col4:
        class_attendance = df.groupby("Class")["Attendance_Percentage"].mean().sort_index()
        fig, ax = plt.subplots(figsize=(8, 5))
        class_attendance.plot(kind="bar", ax=ax, color="#8d72ff")
        ax.set_title("Average Attendance by Class")
        ax.set_xlabel("Class")
        ax.set_ylabel("Attendance (%)")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        st.pyplot(fig)

    st.subheader("Study Time and Performance")
    st.caption("Explore how daily study hours relate to average marks. Hover over points to inspect individual records.")
    identifier_columns = [
        column
        for column in ["Student_ID", "Student_Name", "Student_Roll_Number"]
        if column in df.columns
    ]
    if "Student_Name" not in identifier_columns:
        name_candidates = [
            column
            for column in df.columns
            if column not in {
                "Student_ID",
                "Student_Roll_Number",
                "Class",
                "Gender",
                "Attendance_Category",
                *NUMERIC_COLUMNS,
            }
                and pd.api.types.is_string_dtype(df[column])
        ]
        if name_candidates:
            df = df.rename(columns={name_candidates[0]: "Student_Name"})
            identifier_columns.append("Student_Name")

    if "Student_Name" in df.columns and "Student_Roll_Number" in df.columns:
        df["Student_Key"] = (
            df["Student_Name"].astype(str)
            + " | Roll "
            + df["Student_Roll_Number"].astype(str)
        )
    elif "Student_Name" in df.columns:
        df["Student_Key"] = df["Student_Name"].astype(str)
    elif "Student_Roll_Number" in df.columns:
        df["Student_Key"] = "Roll " + df["Student_Roll_Number"].astype(str)
    elif "Student_ID" in df.columns:
        df["Student_Key"] = df["Student_ID"].astype(str)
    else:
        df["Student_Key"] = df.index.astype(str)

    interactive_df = df[
        ["Student_Key"]
        + [
            "Study_Hours_Per_Day",
            "Average_Marks",
            "Attendance_Category",
            "Class",
        ]
    ].dropna()
    st.vega_lite_chart(
        interactive_df,
        {
            "mark": {"type": "circle", "opacity": 0.85, "size": 90},
            "selection": {
                "grid": {
                    "type": "interval",
                    "bind": "scales",
                }
            },
            "encoding": {
                "x": {
                    "field": "Study_Hours_Per_Day",
                    "type": "quantitative",
                    "title": "Study Hours Per Day",
                },
                "y": {
                    "field": "Average_Marks",
                    "type": "quantitative",
                    "title": "Average Marks",
                },
                "color": {
                    "field": "Attendance_Category",
                    "type": "nominal",
                    "title": "Attendance Category",
                },
                "size": {
                    "field": "Average_Marks",
                    "type": "quantitative",
                    "legend": None,
                },
                "tooltip": [
                    {"field": "Student_Key", "type": "nominal", "title": "Student"},
                    {"field": "Study_Hours_Per_Day", "type": "quantitative", "title": "Study Hours"},
                    {"field": "Average_Marks", "type": "quantitative", "title": "Average Marks"},
                    {"field": "Attendance_Category", "type": "nominal", "title": "Attendance"},
                    {"field": "Class", "type": "nominal", "title": "Class"},
                ],
            },
            "height": 390,
        },
        width="stretch",
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        df[corr_columns].corr(),
        annot=True,
        fmt=".2f",
        cmap="mako",
        center=0,
        ax=ax,
    )
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)


def main():
    st.markdown(
        """
        <section class="dashboard-hero">
            <div class="hero-kicker"><span class="hero-dot"></span> INSIGHT CONSOLE <span class="hero-rule"></span> LIVE DATA VIEW</div>
            <h1 class="hero-title">School Attendance<br><span>&amp; Performance</span></h1>
            <div class="hero-footer">
                <div class="hero-summary">Academic overview and student risk monitoring</div>
                <div class="hero-description">Review attendance patterns, subject trends, and student risk levels in one focused workspace.</div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <style>
        :root { --ink: #edf3ff; --muted: #8d9bc1; --line: #27345f; --blue: #6e63ff; --teal: #32d6c0; --coral: #ff6b78; --surface: #101936; --surface-2: #151f46; }
        .stApp { background: radial-gradient(circle at 82% -10%, rgba(110,99,255,0.22), transparent 28rem), radial-gradient(circle at 0% 80%, rgba(50,214,192,0.08), transparent 24rem), #080e25; color: var(--ink); }
        .stApp::before { content: ""; display: block; position: fixed; inset: 0 0 auto; height: 3px; z-index: 1000; background: linear-gradient(90deg, var(--teal), var(--blue) 58%, var(--coral)); }
        .block-container { max-width: 1500px; padding: 1.15rem 2rem 3rem; }
        [data-testid="stHeader"] { background: rgba(8,14,37,0.78); }
        [data-testid="stSidebar"] { background: linear-gradient(180deg, #0e1634 0%, #0a1029 100%); border-right: 1px solid #1c2850; }
        [data-testid="stSidebar"] > div:first-child { padding-top: 2rem; }
        .sidebar-brand { display: flex; align-items: center; gap: 0.65rem; margin-bottom: 1.5rem; color: #f4f7ff; font-weight: 800; letter-spacing: 0.02em; }
        .sidebar-brand-mark { display: grid; place-items: center; width: 2rem; height: 2rem; border-radius: 9px; color: #081027; background: linear-gradient(135deg, var(--teal), #7182ff); box-shadow: 0 8px 18px rgba(50,214,192,0.22); }
        .sidebar-brand small { display: block; margin-top: 0.15rem; color: #7f90b8; font-size: 0.58rem; letter-spacing: 0.14em; font-weight: 700; }
        [data-testid="stSidebar"] .stMarkdown p { color: var(--muted); }
        [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stFileUploader p { color: #ffffff !important; font-weight: 700; }
        [data-testid="stSidebar"] h3, [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color: #ffffff !important; font-weight: 800; }
        .stSelectbox label, .stSelectSlider label, [data-testid="stWidgetLabel"] p { color: #ffffff !important; font-weight: 700; }
        h1, h2, h3 { color: var(--ink); letter-spacing: 0; }
        h2, h3 { border-left: 3px solid var(--teal); padding-left: 0.65rem; }
        [data-testid="stSidebar"] h2 { color: #ffffff !important; border-left: 3px solid var(--teal); padding-left: 0.65rem; font-size: 1.1rem; letter-spacing: 0.04em; text-transform: uppercase; text-shadow: 0 1px 8px rgba(50,214,192,0.2); }
        [data-testid="stSidebar"] h3 { border-left: 0; padding-left: 0; }
        .dashboard-hero { position: relative; overflow: hidden; background: linear-gradient(118deg, #111a3b 0%, #121b42 55%, #1c1b50 100%); border: 1px solid #2b3970; border-radius: 20px; padding: 1.4rem 1.8rem 1.25rem; margin: 0.2rem 0 1.25rem; box-shadow: 0 22px 50px rgba(0,0,0,0.28); }
        .dashboard-hero::after { content: ""; position: absolute; width: 20rem; height: 20rem; right: -7rem; top: -10rem; border: 1px solid rgba(50,214,192,0.26); border-radius: 50%; box-shadow: 0 0 0 2rem rgba(50,214,192,0.04), 0 0 0 4rem rgba(110,99,255,0.06); }
        .hero-kicker { position: relative; z-index: 1; color: var(--teal); font-size: 0.66rem; font-weight: 800; letter-spacing: 0.18em; }
        .hero-dot { display: inline-block; width: 7px; height: 7px; margin-right: 0.35rem; border-radius: 50%; background: var(--coral); box-shadow: 0 0 0 4px rgba(255,107,120,0.12); }
        .hero-rule { display: inline-block; width: 2.5rem; height: 1px; margin: 0 0.6rem 0.2rem; background: #51608d; }
        .hero-title { position: relative; z-index: 1; margin: 0.55rem 0 1rem; font-family: "Avenir Next", "Trebuchet MS", sans-serif; font-size: 2.85rem; line-height: 0.98; font-weight: 800; letter-spacing: -0.045em; color: #ffffff; }
        .hero-title span { color: var(--teal); }
        .hero-footer { position: relative; z-index: 1; display: grid; grid-template-columns: minmax(14rem, 0.8fr) minmax(18rem, 1.2fr); gap: 1.25rem; align-items: end; max-width: 52rem; padding-top: 0.9rem; border-top: 1px solid #2a3768; }
        .hero-summary { font-size: 1.03rem; font-weight: 700; color: #f4f7ff; }
        .hero-description { color: #9eadd0; font-size: 0.93rem; line-height: 1.5; }
        .hero-corner-mark { position: absolute; right: 1.5rem; bottom: 1.25rem; z-index: 1; color: rgba(193,210,255,0.22); font-size: 2.8rem; line-height: 0.8; font-weight: 800; text-align: right; }
        .hero-corner-mark span { font-size: 0.48rem; letter-spacing: 0.12em; line-height: 1.1; }
        .stButton>button { background: linear-gradient(135deg, #7065ff, #5248df); color: white; border: 1px solid #8178ff; border-radius: 9px; padding: 0.65rem 1.2rem; font-weight: 700; box-shadow: 0 10px 24px rgba(82,72,223,0.28); transition: transform 160ms ease, box-shadow 160ms ease; }
        .stButton>button:hover { background: linear-gradient(135deg, #8076ff, #5b50ed); transform: translateY(-1px); box-shadow: 0 14px 28px rgba(82,72,223,0.38); }
        [data-testid="stDownloadButton"] { margin: 0.75rem 0 1.25rem; }
        [data-testid="stDownloadButton"] button { width: 100%; min-height: 2.75rem; background: linear-gradient(135deg, #32d6c0, #218fbe) !important; color: #061326 !important; border: 1px solid #50e8d4 !important; border-radius: 9px; font-weight: 800; opacity: 1 !important; box-shadow: 0 10px 24px rgba(50,214,192,0.2); }
        [data-testid="stDownloadButton"] button p, [data-testid="stDownloadButton"] button span { color: #061326 !important; font-weight: 800; opacity: 1 !important; }
        [data-testid="stDownloadButton"] button:hover { background: linear-gradient(135deg, #54f0db, #37a9d8) !important; color: #041020 !important; box-shadow: 0 14px 28px rgba(50,214,192,0.32); transform: translateY(-1px); }
        [data-testid="stDownloadButton"] button:focus { outline: 2px solid #ffffff; outline-offset: 2px; }
        .stDataFrame { border-radius: 10px; overflow: hidden; border: 1px solid #26335e; box-shadow: 0 12px 28px rgba(0,0,0,0.18); }
        .stSelectbox > div > div, [data-testid="stFileUploader"] { border-radius: 9px; background: #111a3b; border-color: #2a396d; }
        .stSelectbox [data-baseweb="select"] * { color: #dce6ff; }
        .stMetric { background: linear-gradient(145deg, #141f48, #101936); border: 1px solid #29386b; border-radius: 12px; padding: 0.85rem; box-shadow: 0 12px 25px rgba(0,0,0,0.2); }
        [data-testid="stMetricLabel"] { color: #8e9fc8; }
        [data-testid="stMetricValue"] { color: #ffffff; }
        [data-testid="stHorizontalBlock"] { gap: 1rem; }
        .stAlert, .stInfo, .stSuccess, .stWarning, .stError { border-radius: 10px; background: #121d43; border-color: #2e4078; color: #dce6ff; }
        .stAlert p, .stInfo p, .stSuccess p, .stWarning p, .stError p { color: #dce6ff; }
        [data-testid="stFileUploader"] section { background: #111a3b; border: 1px dashed #344477; }
        [data-testid="stFileUploader"] button { background: #182452; border-color: #3a4c84; color: #dce6ff; }
        [data-testid="stExpander"] { background: #101936; border: 1px solid #26345f; border-radius: 10px; }
        .stCaption, [data-testid="stCaptionContainer"] { color: #8292b9; }
        .stTabs [role="tablist"] { gap: 0.5rem; }
        .stTabs [role="tab"] { border-radius: 9px 9px 0 0; color: #9eadd0; }
        @media (max-width: 768px) { .block-container { padding: 1rem 0.85rem 2rem; } .dashboard-hero { padding: 1.25rem 1.1rem 1.1rem; } .hero-title { font-size: 2.25rem; } .hero-footer { grid-template-columns: 1fr; gap: 0.45rem; } .hero-corner-mark { display: none; } }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-mark">A</div>
            <div>ATTENDANCE LAB<small>PERFORMANCE INTELLIGENCE</small></div>
        </div>
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

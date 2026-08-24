import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(
    page_title="Attendance Performance Dashboard",
    page_icon="🎓",
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
PREDICTOR_FEATURES = [
    "Attendance_Percentage",
    "Assignment_Score",
    "Study_Hours_Per_Day",
    "Previous_Marks",
]
RISK_COLORS = {"High Risk": "#ef4444", "Medium Risk": "#f59e0b", "Low Risk": "#22c55e"}
ACCENT = "#6366f1"
ACCENT_2 = "#06b6d4"

PLOTLY_TEMPLATE = "plotly_white"
CATEGORY_ORDER = ["Low", "Average", "Good", "Excellent"]


def inject_theme():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"]  {{
            font-family: 'Poppins', sans-serif;
        }}

        .stApp {{
            background: radial-gradient(1200px 600px at 10% -10%, #eef2ff 0%, rgba(238,242,255,0) 60%),
                        radial-gradient(1000px 500px at 100% 0%, #ecfeff 0%, rgba(236,254,255,0) 55%),
                        #f8fafc;
        }}

        .block-container {{
            padding: 1.2rem 2.4rem 3rem 2.4rem;
            max-width: 1400px;
        }}

        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #111827 0%, #1f2937 100%);
        }}
        section[data-testid="stSidebar"] * {{
            color: #e5e7eb !important;
        }}
        section[data-testid="stSidebar"] .stButton>button {{
            background: linear-gradient(135deg, {ACCENT} 0%, {ACCENT_2} 100%);
            color: white !important;
            border: none;
        }}

        .hero {{
            background: linear-gradient(135deg, {ACCENT} 0%, #8b5cf6 45%, {ACCENT_2} 100%);
            border-radius: 22px;
            padding: 2.2rem 2.4rem;
            color: white;
            margin-bottom: 1.6rem;
            box-shadow: 0 18px 40px -12px rgba(99, 102, 241, 0.45);
            position: relative;
            overflow: hidden;
        }}
        .hero::after {{
            content: "";
            position: absolute;
            inset: 0;
            background: radial-gradient(400px 200px at 90% 10%, rgba(255,255,255,0.25), transparent 60%);
        }}
        .hero h1 {{
            font-weight: 800;
            font-size: 2.1rem;
            margin: 0 0 .35rem 0;
            letter-spacing: -0.02em;
        }}
        .hero p {{
            font-size: 1.02rem;
            opacity: 0.92;
            margin: 0;
            max-width: 780px;
        }}

        .glass-card {{
            background: rgba(255, 255, 255, 0.85);
            border: 1px solid rgba(226, 232, 240, 0.8);
            border-radius: 18px;
            padding: 1.1rem 1.3rem;
            box-shadow: 0 8px 24px -12px rgba(15, 23, 42, 0.12);
            transition: transform 0.18s ease, box-shadow 0.18s ease;
            height: 100%;
        }}
        .glass-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 16px 32px -14px rgba(15, 23, 42, 0.22);
        }}

        div[data-testid="stMetric"] {{
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 0.9rem 1rem 0.6rem 1rem;
            box-shadow: 0 6px 18px -10px rgba(15, 23, 42, 0.15);
            transition: transform 0.15s ease;
        }}
        div[data-testid="stMetric"]:hover {{
            transform: translateY(-2px);
        }}

        .stTabs [data-baseweb="tab-list"] {{
            gap: 6px;
        }}
        .stTabs [data-baseweb="tab"] {{
            background-color: rgba(99, 102, 241, 0.06);
            border-radius: 12px 12px 0 0;
            padding: 10px 18px;
            font-weight: 600;
        }}
        .stTabs [aria-selected="true"] {{
            background: linear-gradient(135deg, {ACCENT} 0%, {ACCENT_2} 100%) !important;
            color: white !important;
        }}

        .stButton>button {{
            border-radius: 10px;
            font-weight: 600;
            transition: transform 0.12s ease, box-shadow 0.12s ease;
        }}
        .stButton>button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 8px 18px -8px rgba(99, 102, 241, 0.55);
        }}

        .risk-badge {{
            display: inline-block;
            padding: 0.35rem 0.9rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.95rem;
            color: white;
        }}

        hr {{
            border-top: 1px solid #e2e8f0;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


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
        labels=CATEGORY_ORDER,
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
        labels=CATEGORY_ORDER,
    )
    return df


def build_regression_model(df: pd.DataFrame):
    """Fit a linear regression model and return it with its evaluation metrics.

    Falls back gracefully (instead of crashing the page) when the dataset is
    too small to carve out a meaningful train/test split.
    """
    X = df[PREDICTOR_FEATURES]
    y = df["Average_Marks"]

    if len(df) < 5:
        model = LinearRegression().fit(X, y)
        preds = model.predict(X)
        return {
            "model": model,
            "mae": mean_absolute_error(y, preds),
            "r2": r2_score(y, preds),
            "warning": "Dataset is small — metrics were computed on the full sample (no holdout split).",
        }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return {
        "model": model,
        "mae": mean_absolute_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
        "warning": None,
    }


def build_risk_classifier(df: pd.DataFrame):
    """Fit a decision tree risk classifier and return it with evaluation info.

    Previously this crashed the whole dashboard whenever a dataset (e.g. a
    small or heavily imbalanced upload) didn't have enough members in every
    risk class to support a stratified split. It now degrades gracefully
    instead of taking down the page.
    """
    df = df.copy()
    df["Performance_Risk"] = pd.cut(
        df["Average_Marks"],
        bins=[-1, 49.99, 69.99, 100],
        labels=["High Risk", "Medium Risk", "Low Risk"],
    )
    X = df[PREDICTOR_FEATURES]
    y = df["Performance_Risk"]

    warning = None
    class_counts = y.value_counts()

    if len(df) < 10 or y.nunique() < 2 or class_counts.min() < 2:
        X_train, X_test, y_train, y_test = X, X, y, y
        warning = "Dataset is small or imbalanced across risk levels — showing in-sample metrics only."
    else:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            warning = "Could not stratify by risk level (a class is too rare) — used a plain random split instead."

    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    risk_predictions = model.predict(X_test)
    return {
        "model": model,
        "accuracy": accuracy_score(y_test, risk_predictions),
        "report": classification_report(y_test, risk_predictions, zero_division=0),
        "labels": sorted(y.unique().tolist(), key=lambda r: ["High Risk", "Medium Risk", "Low Risk"].index(r)),
        "y_test": y_test,
        "y_pred": risk_predictions,
        "warning": warning,
    }


def render_metric_cards(df: pd.DataFrame):
    c1, c2, c3, c4 = st.columns(4)
    at_risk_count = int(
        ((df["Attendance_Percentage"] < 70) | (df["Average_Marks"] < 50)).sum()
    )
    c1.metric("Students", df.shape[0])
    c2.metric("Avg. Attendance", f"{df['Attendance_Percentage'].mean():.1f}%")
    c3.metric("Avg. Marks", f"{df['Average_Marks'].mean():.1f}")
    c4.metric("At-risk Students", at_risk_count, delta=f"{at_risk_count/len(df)*100:.0f}% of cohort", delta_color="inverse")


def render_charts(df: pd.DataFrame):
    col1, col2 = st.columns(2)

    with col1:
        fig = px.scatter(
            df,
            x="Attendance_Percentage",
            y="Average_Marks",
            color="Attendance_Category",
            category_orders={"Attendance_Category": CATEGORY_ORDER},
            trendline="ols",
            opacity=0.75,
            template=PLOTLY_TEMPLATE,
            color_discrete_sequence=px.colors.sequential.Viridis,
            title="Attendance vs Academic Performance",
        )
        fig.update_layout(legend_title_text="Attendance", margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        category_means = (
            df.groupby("Attendance_Category", observed=False)["Average_Marks"]
            .mean()
            .reindex(CATEGORY_ORDER)
            .reset_index()
        )
        fig = px.bar(
            category_means,
            x="Attendance_Category",
            y="Average_Marks",
            color="Attendance_Category",
            category_orders={"Attendance_Category": CATEGORY_ORDER},
            color_discrete_sequence=px.colors.sequential.Viridis,
            template=PLOTLY_TEMPLATE,
            title="Average Marks by Attendance Category",
        )
        fig.update_layout(showlegend=False, margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        subject_means = (
            df[["Maths_Marks", "Science_Marks", "English_Marks"]]
            .mean()
            .reset_index()
        )
        subject_means.columns = ["Subject", "Average_Marks"]
        fig = px.bar(
            subject_means,
            x="Subject",
            y="Average_Marks",
            color="Subject",
            template=PLOTLY_TEMPLATE,
            color_discrete_sequence=["#f97316", "#22c55e", "#ef4444"],
            title="Average Subject Performance",
        )
        fig.update_layout(showlegend=False, margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        class_attendance = (
            df.groupby("Class")["Attendance_Percentage"].mean().sort_index().reset_index()
        )
        fig = px.bar(
            class_attendance,
            x="Class",
            y="Attendance_Percentage",
            color="Class",
            template=PLOTLY_TEMPLATE,
            color_discrete_sequence=px.colors.sequential.Purp,
            title="Average Attendance by Class",
        )
        fig.update_layout(showlegend=False, margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

    corr_columns = [
        "Attendance_Percentage",
        "Assignment_Score",
        "Study_Hours_Per_Day",
        "Previous_Marks",
        "Average_Marks",
    ]
    corr = df[corr_columns].corr().round(2)
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        template=PLOTLY_TEMPLATE,
        title="Correlation Heatmap",
        aspect="auto",
    )
    fig.update_layout(margin=dict(t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)


def render_prediction_tab(df: pd.DataFrame):
    st.markdown("#### Model performance")
    reg_result = build_regression_model(df)
    clf_result = build_risk_classifier(df)

    if reg_result["warning"]:
        st.warning(reg_result["warning"])
    if clf_result["warning"]:
        st.warning(clf_result["warning"])

    m1, m2, m3 = st.columns(3)
    m1.metric("Regression MAE", f"{reg_result['mae']:.2f}")
    m2.metric("Regression R²", f"{reg_result['r2']:.2f}")
    m3.metric("Risk Classifier Accuracy", f"{clf_result['accuracy']*100:.1f}%")

    with st.expander("Detailed classification report"):
        st.text(clf_result["report"])

        cm = confusion_matrix(clf_result["y_test"], clf_result["y_pred"], labels=clf_result["labels"])
        cm_fig = px.imshow(
            cm,
            x=clf_result["labels"],
            y=clf_result["labels"],
            text_auto=True,
            color_continuous_scale="Blues",
            template=PLOTLY_TEMPLATE,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            title="Confusion Matrix",
        )
        st.plotly_chart(cm_fig, use_container_width=True)

    fi1, fi2 = st.columns(2)
    with fi1:
        coef_df = pd.DataFrame(
            {"Feature": PREDICTOR_FEATURES, "Influence": reg_result["model"].coef_}
        ).sort_values("Influence")
        fig = px.bar(
            coef_df,
            x="Influence",
            y="Feature",
            orientation="h",
            template=PLOTLY_TEMPLATE,
            color="Influence",
            color_continuous_scale="RdBu",
            title="What drives Average Marks (regression coefficients)",
        )
        fig.update_layout(margin=dict(t=50, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with fi2:
        imp_df = pd.DataFrame(
            {"Feature": PREDICTOR_FEATURES, "Importance": clf_result["model"].feature_importances_}
        ).sort_values("Importance")
        fig = px.bar(
            imp_df,
            x="Importance",
            y="Feature",
            orientation="h",
            template=PLOTLY_TEMPLATE,
            color="Importance",
            color_continuous_scale="Viridis",
            title="What drives Risk classification (feature importance)",
        )
        fig.update_layout(margin=dict(t=50, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Try the predictor")
    st.caption(
        "Adjust a hypothetical student's profile and get a live prediction from the trained models."
    )

    p1, p2 = st.columns([1, 1.2])
    with p1:
        attendance_in = st.slider(
            "Attendance (%)",
            float(df["Attendance_Percentage"].min()),
            100.0,
            float(df["Attendance_Percentage"].mean()),
            0.5,
        )
        assignment_in = st.slider(
            "Assignment Score",
            0.0,
            100.0,
            float(df["Assignment_Score"].mean()),
            0.5,
        )
        study_in = st.slider(
            "Study Hours / Day",
            0.0,
            12.0,
            float(df["Study_Hours_Per_Day"].mean()),
            0.1,
        )
        previous_in = st.slider(
            "Previous Marks",
            0.0,
            100.0,
            float(df["Previous_Marks"].mean()),
            0.5,
        )

        input_row = pd.DataFrame(
            [[attendance_in, assignment_in, study_in, previous_in]],
            columns=PREDICTOR_FEATURES,
        )
        predicted_marks = float(np.clip(reg_result["model"].predict(input_row)[0], 0, 100))
        predicted_risk = clf_result["model"].predict(input_row)[0]
        risk_probabilities = clf_result["model"].predict_proba(input_row)[0]
        risk_labels = clf_result["model"].classes_

    with p2:
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=predicted_marks,
                number={"suffix": " / 100", "font": {"size": 42}},
                title={"text": "Predicted Average Marks"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": ACCENT},
                    "steps": [
                        {"range": [0, 50], "color": "#fee2e2"},
                        {"range": [50, 70], "color": "#fef3c7"},
                        {"range": [70, 100], "color": "#dcfce7"},
                    ],
                },
            )
        )
        gauge.update_layout(height=280, margin=dict(t=40, b=10, l=20, r=20))
        st.plotly_chart(gauge, use_container_width=True)

        badge_color = RISK_COLORS.get(predicted_risk, ACCENT)
        st.markdown(
            f"<span class='risk-badge' style='background:{badge_color};'>Predicted risk: {predicted_risk}</span>",
            unsafe_allow_html=True,
        )

        prob_df = pd.DataFrame({"Risk": risk_labels, "Probability": risk_probabilities})
        prob_fig = px.bar(
            prob_df,
            x="Probability",
            y="Risk",
            orientation="h",
            range_x=[0, 1],
            template=PLOTLY_TEMPLATE,
            color="Risk",
            color_discrete_map=RISK_COLORS,
            title="Risk class probabilities",
        )
        prob_fig.update_layout(showlegend=False, margin=dict(t=50, b=10, l=10, r=10), height=220)
        st.plotly_chart(prob_fig, use_container_width=True)


def main():
    inject_theme()

    st.markdown(
        """
        <div class="hero">
            <h1>🎓 School Attendance &amp; Performance Dashboard</h1>
            <p>Explore attendance and academic trends, spot at-risk students early, and get
            live predictions from trained machine-learning models — using your own dataset or
            the built-in sample data.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.header("📁 Data Controls")
    st.sidebar.write(
        "Upload your dataset in CSV/Excel format, or use the built-in sample dataset for quick exploration."
    )
    uploaded_file = st.sidebar.file_uploader(
        "Upload dataset from your device",
        type=["csv", "xls", "xlsx"],
        help="Select a CSV or Excel file with attendance and performance records.",
    )
    use_sample = st.sidebar.button("✨ Use sample dataset")
    analyze_button = st.sidebar.button("🚀 Analyze data")

    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
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
            if st.button("✨ Use sample dataset", key="main_sample"):
                use_sample = True
            if st.button("🚀 Analyze data", key="main_analyze"):
                analyze_button = True
        st.markdown("</div>", unsafe_allow_html=True)

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
        st.info("Upload a CSV file or click '✨ Use sample dataset' to begin.")
        return

    df = clean_data(df)

    if df.empty:
        st.error("No valid rows remain after cleaning. Please check your file's columns and values.")
        return

    st.markdown("<br>", unsafe_allow_html=True)
    render_metric_cards(df)

    tab_overview, tab_insights, tab_risk, tab_predict = st.tabs(
        ["📋 Overview", "📊 Insights", "⚠️ At-Risk Students", "🎯 Predict"]
    )

    with tab_overview:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
        with st.expander("Dataset summary and missing values"):
            st.write(df.describe(include="all"))
            st.write("### Missing values")
            st.write(df.isnull().sum())

        st.write("### Average Performance by Attendance Category")
        category_result = (
            df.groupby("Attendance_Category", observed=False)["Average_Marks"]
            .agg(["count", "mean", "median"])
            .round(2)
        )
        st.dataframe(category_result, use_container_width=True)

        st.write("### Class-wise Attendance and Average Marks")
        st.dataframe(
            df.groupby("Class")[["Attendance_Percentage", "Average_Marks"]]
            .mean()
            .round(2),
            use_container_width=True,
        )

    with tab_insights:
        correlation = df["Attendance_Percentage"].corr(df["Average_Marks"])
        st.info(f"📈 Attendance vs Average Marks correlation: **{correlation:.3f}**")
        render_charts(df)

    with tab_risk:
        at_risk = df[(df["Attendance_Percentage"] < 70) | (df["Average_Marks"] < 50)].copy()
        st.subheader(f"At-risk Students ({len(at_risk)})")
        st.dataframe(at_risk, use_container_width=True)

        csv_buffer = io.StringIO()
        at_risk.to_csv(csv_buffer, index=False)
        st.download_button(
            label="⬇️ Download at-risk students",
            data=csv_buffer.getvalue(),
            file_name="at_risk_students.csv",
            mime="text/csv",
        )

    with tab_predict:
        render_prediction_tab(df)


if __name__ == "__main__":
    main()

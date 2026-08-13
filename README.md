# Attendance & Performance Pattern Analysis in Schools

A beginner-friendly Data Science project using Python, Pandas, Matplotlib, Seaborn and optional Scikit-learn.

## Live dashboard (Netlify)

The `public/` folder contains a static, client-side version of the dashboard (HTML/CSS/JS with Chart.js)
that runs entirely in the browser and is what gets deployed to Netlify — see `netlify.toml`. It reproduces
the dataset preview, summary metrics, category/class/subject breakdowns, charts, correlation heatmap, and
at-risk student list from `app.py`, using either the bundled sample dataset or an uploaded CSV/Excel file.
The regression and decision-tree risk-classifier models are not part of the static dashboard; run
`prediction.py` locally with Python to see those.

## Files
- `generate_dataset.py` - creates a sample school dataset
- `analysis.py` - performs cleaning, EDA, correlation and visualizations
- `prediction.py` - optional Linear Regression and risk classification
- `requirements.txt` - required Python packages

## Run
```bash
pip install -r requirements.txt
python generate_dataset.py
python analysis.py
python prediction.py
```

The generated charts are saved in the `outputs/` folder.

## UI Dashboard
Run the Streamlit UI with:
```bash
streamlit run app.py
```

The app includes an **Upload data** button for uploading a CSV file from your system, plus sample dataset support and interactive charts.

## Deployment (Streamlit Community Cloud)

1. Push this repository to GitHub (public or private).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click *New app*, pick your repo and branch, and set `app.py` as the main file.
4. Ensure `requirements.txt` is present at the repo root (already included).
5. Deploy — Streamlit will install dependencies and run your app automatically.

Alternatively, you can deploy using Docker or other PaaS; add a `Dockerfile` if you prefer container-based deploys.

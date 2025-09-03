
import streamlit as st
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
import joblib
import seaborn as sns

st.set_page_config(page_title="SmartPath Career Trends (ML)", layout="wide")
st.title("SmartPath — Tren Karier Data di Indonesia (ML Demo)")
st.write("Upload CSV atau gunakan sample data. Model: RandomForestClassifier (demand level) + RandomForestRegressor (growth).")

uploaded = st.file_uploader("Upload CSV (data_career_trend_indonesia.csv)", type=['csv'])
if uploaded is not None:
    df = pd.read_csv(uploaded, parse_dates=['posting_date'])
else:
    df = pd.read_csv("/mnt/data/data_career_trend_indonesia.csv", parse_dates=['posting_date'])

st.sidebar.header("Filters")
cities = ['All'] + sorted(df['city'].unique().tolist())
sel_city = st.sidebar.selectbox("City", cities)
if sel_city != 'All':
    df = df[df['city']==sel_city]

st.subheader("Data sample")
st.dataframe(df.head(10))

st.subheader("Distribution: demand_index & growth_trend_percent")
col1, col2 = st.columns(2)
with col1:
    st.hist_chart = st.bar_chart(df['demand_index'].value_counts().sort_index())
with col2:
    st.line_chart = st.line_chart(df.groupby(df['posting_date'].dt.to_period('M'))['growth_trend_percent'].mean())

st.subheader("Train & Evaluate Models (Demo)")
if st.button("Train Models (This runs locally)"):
    # Simple: reuse code from notebook
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    df['demand_level'] = pd.cut(df['demand_index'], bins=[0,64,79,100], labels=['Low','Medium','High'])
    features = ['role','industry','city','avg_salary_million_idr','experience_required_years','remote_opportunity','demand_index','growth_trend_percent']
    X = df[features]
    y = df['demand_level']

    cat_cols = ['role','industry','city']
    num_cols = ['avg_salary_million_idr','experience_required_years','remote_opportunity','demand_index']

    num_pipe = Pipeline([('scaler', StandardScaler())])
    cat_pipe = Pipeline([('ohe', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer([('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf = Pipeline([('prep', preprocessor), ('rf', RandomForestClassifier(n_estimators=150, random_state=42))])
    clf.fit(X_train, y_train)
    st.success("Model trained")
    st.write("Classification report:")
    from sklearn.metrics import classification_report
    y_pred = clf.predict(X_test)
    st.text(classification_report(y_test, y_pred))

    joblib.dump(clf, 'rf_demand_level_clf.joblib')
    st.write("Saved model: rf_demand_level_clf.joblib")

st.subheader("Clustering (KMeans)")
if st.button("Run Clustering"):
    # quick clustering
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    features = ['role','industry','city','avg_salary_million_idr','experience_required_years','remote_opportunity','demand_index','growth_trend_percent']
    Xc = df[features].copy()
    cat_cols = ['role','industry','city']
    num_cols = ['avg_salary_million_idr','experience_required_years','remote_opportunity','demand_index','growth_trend_percent']
    num_pipe = Pipeline([('scaler', StandardScaler())])
    cat_pipe = Pipeline([('ohe', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer([('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)])
    Xc_trans = preprocessor.fit_transform(Xc)
    k = st.slider("K (clusters)", 2, 6, 3)
    km = KMeans(n_clusters=k, random_state=42).fit(Xc_trans)
    df['cluster'] = km.labels_
    st.write(df[['role','industry','city','avg_salary_million_idr','cluster']].head(20))
    st.success("Clustering done. Use cluster labels for curriculum mapping.")

st.caption("Demo app: replace sample CSV with real scraped job postings for production-ready analysis.")

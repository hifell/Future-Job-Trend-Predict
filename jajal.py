import os
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --------------------
# Page Config
# --------------------
st.set_page_config(page_title="Future Jobs — ML Dashboard", page_icon="📈", layout="wide")
st.title("Future Job Trend — Interactive ML Dashboard")

# --------------------
# Constants
# --------------------
MODEL_FILES = {
    "Random Forest (best)": "rf_jobtrend_reg_best.joblib",
    "Gradient Boosting": "gbr_jobtrend_reg.joblib",
}

REQUIRED_FEATURES = [
    "Year", "Job_Type", "Tech_Index", "AI_Adoption", "Automation_Level",
    "Education_Index", "Remote_Work_Index", "Market_Demand"
]
TARGET_COL = "Job_Trend_Score"

# --------------------
# Helpers
# --------------------
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return joblib.load(path)

def find_available_models() -> dict:
    available = {}
    for name, fn in MODEL_FILES.items():
        if os.path.exists(fn):
            try:
                available[name] = load_model(fn)
            except Exception as e:
                st.warning(f"Gagal memuat model {name}: {e}")
    return available

def ensure_required_columns(df: pd.DataFrame, need_target: bool=False) -> bool:
    needed = set(REQUIRED_FEATURES)
    if need_target:
        needed.add(TARGET_COL)
    missing = needed - set(df.columns)
    if missing:
        st.error(f"Kolom hilang: {sorted(list(missing))}")
        return False
    return True

def auto_extrapolate_features(df: pd.DataFrame, year: int) -> dict:
    """
    Hitung nilai fitur otomatis untuk tahun > max(df['Year'])
    berdasarkan regresi linear sederhana (linier fit).
    """
    feat_values = {}
    for col in [c for c in REQUIRED_FEATURES if c not in ["Year", "Job_Type"]]:
        try:
            temp = df[["Year", col]].dropna()
            if temp["Year"].nunique() > 1:
                z = np.polyfit(temp["Year"], temp[col], 1)
                p = np.poly1d(z)
                val = float(p(year))
                feat_values[col] = max(0.0, min(1.0, val))  # clamp [0,1]
            else:
                feat_values[col] = float(temp[col].median())
        except Exception:
            feat_values[col] = float(df[col].median())
    return feat_values

def plot_feature_importance(pipeline, X_sample: pd.DataFrame, top_n: int = 10):
    st.subheader("🔎 Feature Importance")
    try:
        model = pipeline.named_steps[[k for k in pipeline.named_steps if k != "prep"][0]]
        pre = pipeline.named_steps.get("prep")
        Xt = pre.transform(X_sample)
        feat_names = []
        for name, trans, cols in pre.transformers_:
            if hasattr(trans, "get_feature_names_out"):
                try:
                    feat_names.extend(list(trans.get_feature_names_out(cols)))
                except:
                    feat_names.extend(list(trans.get_feature_names_out()))
            else:
                feat_names.extend(cols if isinstance(cols, list) else [cols])

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            fi = pd.DataFrame({"feature": feat_names[:len(importances)], "importance": importances})
            fi = fi.sort_values("importance", ascending=False).head(top_n)
        else:
            r = permutation_importance(pipeline, X_sample, np.zeros(len(X_sample)), n_repeats=3, random_state=42)
            fi = pd.DataFrame({"feature": feat_names[:len(r.importances_mean)], "importance": r.importances_mean})
            fi = fi.sort_values("importance", ascending=False).head(top_n)

        fig, ax = plt.subplots(figsize=(7, max(4, top_n*0.35)))
        ax.barh(fi["feature"][::-1], fi["importance"][::-1])
        ax.set_title("Top Feature Importance")
        st.pyplot(fig)
        st.dataframe(fi.reset_index(drop=True))
    except Exception as e:
        st.warning(f"Tidak bisa hitung importance: {e}")

def cluster_explorer(df: pd.DataFrame):
    st.subheader("🧩 PCA Scatter")
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) < 2:
        st.info("Butuh ≥2 kolom numerik")
        return
    X = df[num_cols].dropna()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    Xp = pca.fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(6,5))
    ax.scatter(Xp[:,0], Xp[:,1], s=20, alpha=0.6)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    st.pyplot(fig)

# --------------------
# Sidebar
# --------------------
st.sidebar.header("⚙️ Pengaturan")

DATA_PATH = st.sidebar.text_input("Lokasi dataset (CSV)", value="future_job_trend_with_types.csv")

if not os.path.exists(DATA_PATH):
    st.warning("Dataset tidak ditemukan. Upload manual di bawah.")
uploaded = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False)
if uploaded is not None:
    DATA_PATH = uploaded

try:
    df = load_data(DATA_PATH)
    st.success("Dataset berhasil dimuat!")
except Exception as e:
    st.error(f"Gagal memuat dataset: {e}")
    df = pd.DataFrame()

models = find_available_models()
if not models:
    st.warning("Model *.joblib* tidak ada. Jalankan training dulu.")

# --------------------
# Tabs
# --------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔮 Single Prediction", "📦 Batch Predict", "📊 Model Analysis", 
    "📈 EDA", "🧩 PCA/Cluster", "💾 Download"
])

# --------------------
# Tab1 Single Prediction
# --------------------
with tab1:
    st.subheader("🔮 Prediksi Tunggal")
    if df.empty:
        st.info("Dataset kosong.")
    else:
        job_types = sorted(df["Job_Type"].dropna().unique()) if "Job_Type" in df else ["Data Scientist"]
        Year = st.number_input("Year", min_value=2000, max_value=2100, value=2030, step=1)
        Job_Type = st.selectbox("Job_Type", job_types)
        auto_fill = st.checkbox("Gunakan extrapolasi otomatis", value=True)

        if auto_fill and Year > df["Year"].max():
            extrap = auto_extrapolate_features(df, Year)
            Tech_Index = extrap["Tech_Index"]
            AI_Adoption = extrap["AI_Adoption"]
            Automation_Level = extrap["Automation_Level"]
            Education_Index = extrap["Education_Index"]
            Remote_Work_Index = extrap["Remote_Work_Index"]
            Market_Demand = extrap["Market_Demand"]
            st.info("Nilai fitur diisi otomatis dengan extrapolasi.")
        else:
            Tech_Index = st.slider("Tech_Index", 0.0, 1.0, 0.5)
            AI_Adoption = st.slider("AI_Adoption", 0.0, 1.0, 0.5)
            Automation_Level = st.slider("Automation_Level", 0.0, 1.0, 0.5)
            Education_Index = st.slider("Education_Index", 0.0, 1.0, 0.5)
            Remote_Work_Index = st.slider("Remote_Work_Index", 0.0, 1.0, 0.5)
            Market_Demand = st.slider("Market_Demand", 0.0, 1.0, 0.5)

        input_row = {
            "Year": Year,
            "Job_Type": Job_Type,
            "Tech_Index": Tech_Index,
            "AI_Adoption": AI_Adoption,
            "Automation_Level": Automation_Level,
            "Education_Index": Education_Index,
            "Remote_Work_Index": Remote_Work_Index,
            "Market_Demand": Market_Demand
        }
        st.json(input_row)

        model_name = st.selectbox("Model", list(models.keys()) if models else ["-"])
        if model_name in models and st.button("🚀 Prediksi"):
            try:
                pred = models[model_name].predict(pd.DataFrame([input_row]))[0]
                st.success(f"Prediksi Job_Trend_Score: **{pred:.3f}**")
            except Exception as e:
                st.error(f"Error prediksi: {e}")

# --------------------
# Tab2 Batch Predict
# --------------------
with tab2:
    st.subheader("📦 Batch Prediction")
    file_b = st.file_uploader("Upload CSV untuk batch", type=["csv"], key="batch")
    model_name_b = st.selectbox("Pilih Model", list(models.keys()) if models else ["-"], key="batch_model")
    if file_b and model_name_b in models:
        df_b = pd.read_csv(file_b)
        if ensure_required_columns(df_b):
            preds = models[model_name_b].predict(df_b[REQUIRED_FEATURES])
            df_b["Predicted_Job_Trend_Score"] = preds
            st.dataframe(df_b.head())
            st.download_button("💾 Download Hasil", df_b.to_csv(index=False).encode("utf-8"),
                               "batch_predictions.csv", "text/csv")

# --------------------
# Tab3 Model Analysis
# --------------------
with tab3:
    st.subheader("📊 Analisis Model")
    if not df.empty and models:
        model_name_m = st.selectbox("Model", list(models.keys()), key="model_analysis")
        if ensure_required_columns(df, need_target=True):
            X = df[REQUIRED_FEATURES]
            y = df[TARGET_COL]
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
            pipe = models[model_name_m]
            yhat = pipe.predict(Xte)
            rmse = mean_squared_error(yte, yhat, squared=False)
            r2 = r2_score(yte, yhat)
            st.metric("RMSE", f"{rmse:.4f}")
            st.metric("R²", f"{r2:.4f}")
            fig, ax = plt.subplots()
            ax.scatter(yte, yhat, alpha=0.6)
            ax.plot([yte.min(), yte.max()], [yte.min(), yte.max()], "--")
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            st.pyplot(fig)
            plot_feature_importance(pipe, X)

# --------------------
# Tab4 EDA
# --------------------
with tab4:
    st.subheader("📈 Exploratory Data Analysis")
    if not df.empty:
        st.write("Statistik deskriptif:")
        st.dataframe(df.describe(include="all"))
        st.write("Distribusi numerik:")
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols):
            fig, ax = plt.subplots(figsize=(7,4))
            df[num_cols].hist(ax=ax)
            st.pyplot(fig)
        st.write("Heatmap korelasi:")
        fig, ax = plt.subplots(figsize=(7,5))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

# --------------------
# Tab5 PCA/Cluster
# --------------------
with tab5:
    if not df.empty:
        cluster_explorer(df)

# --------------------
# Tab6 Download
# --------------------
with tab6:
    if not df.empty:
        st.download_button("💾 Download Dataset Asli", df.to_csv(index=False).encode("utf-8"),
                           "dataset.csv", "text/csv")

# --------------------
# Footer
# --------------------
st.markdown("---\nFell 2025")

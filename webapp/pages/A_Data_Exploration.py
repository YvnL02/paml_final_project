import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(layout="wide")
st.title("1. Data Exploration")

try:
    import kagglehub
except ImportError:
    st.error("Please install `kagglehub`: `pip install kagglehub`")
    st.stop()

@st.cache_data(show_spinner=True)
def load_kaggle_dataset():
    path = kagglehub.dataset_download("juice0lover/users-vs-bots-classification")
    csv_path = os.path.join(path, "bots_vs_users.csv")
    df = pd.read_csv(csv_path)
    return df

# Load dataset
st.markdown("Loading dataset from Kaggle...")
df = load_kaggle_dataset()
st.session_state["raw_df"] = df

st.markdown("### 1.1 Raw Dataset Preview")
st.dataframe(df.head())
st.write("Dataset shape:", df.shape)

# Class distribution
if "target" in df.columns:
    st.markdown("### 1.2 Class Distribution (Human vs Bot)")
    st.bar_chart(df["target"].value_counts())

# Distinguish column types
numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object", "bool", "category"]).columns.tolist()

# Missing value
with st.subheader("Missing Values Overview"):
    missing_df = df.isna().sum()
    missing_df = missing_df[missing_df > 0].sort_values(ascending=False)
    if not missing_df.empty:
        st.dataframe(missing_df.rename("Missing Count"))
    else:
        st.write("No missing values found.")

# Data distribution
st.header("1.3 Feature Distribution")
selected_feature = st.selectbox("Choose a feature", numeric_cols)
fig = px.histogram(df, x=selected_feature, color="target" if "target" in df else None, marginal="box", nbins=30)
st.plotly_chart(fig, use_container_width=True)

# Correlation Heatmap
with st.subheader("Correlation Heatmap (Numeric Features)"):
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough numeric features to compute correlation.")

# Scatterplot
st.subheader("1.4 Pairwise Scatterplot")
col1, col2 = st.columns(2)
with col1:
    feature_x = st.selectbox("X-axis feature", numeric_cols, key="pair_x")
with col2:
    feature_y = st.selectbox("Y-axis feature", numeric_cols, key="pair_y")

if feature_x and feature_y and feature_x != feature_y:
    fig = px.scatter(df, x=feature_x, y=feature_y, color="target" if "target" in df else None)
    st.plotly_chart(fig, use_container_width=True)


# Class statistics
with st.subheader("Feature Statistics by Class"):
    stat_feature = st.selectbox("Select feature", numeric_cols + categorical_cols, key="stat_by_class")
    if stat_feature:
        group_stats = df.groupby("target")[stat_feature].describe()
        st.dataframe(group_stats)

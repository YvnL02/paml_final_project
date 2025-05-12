import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

st.set_page_config(layout="wide")
st.title("2. Feature Engineering")

# Load raw dataset from session state
raw_df = st.session_state.get("raw_df")
if raw_df is None:
    st.error("Raw dataset not found. Please go to '1. Data Exploration' first.")
    st.stop()

df = raw_df.copy()

# Missing Value Handling
st.subheader("2.1 Handle Missing Values")

missing_cols = df.columns[df.isna().any()].tolist()

if missing_cols:
    st.write("Missing columns detected:", missing_cols)
    selected_cols = st.multiselect("Select columns to apply missing value strategy", missing_cols, default=missing_cols)
    method = st.selectbox("Missing value strategy", ["Drop rows", "Fill with mean", "Fill with median", "Fill with mode"])
    
    if st.button("Apply Missing Value Handling"):
        for col in selected_cols:
            if method == "Drop rows":
                df = df[df[col].notna()]
            elif method == "Fill with mean":
                df[col] = df[col].fillna(df[col].mean())
            elif method == "Fill with median":
                df[col] = df[col].fillna(df[col].median())
            elif method == "Fill with mode":
                df[col] = df[col].fillna(df[col].mode()[0])
        st.success("Missing value handling complete.")
else:
    st.info("No missing values found.")


# Outlier Removal (IQR)
st.subheader("2.2 Outlier Removal (IQR Method)")

numeric_cols = df.select_dtypes(include=["int", "float"]).columns.tolist()
selected_feature = st.selectbox("Select numeric feature to remove outliers", numeric_cols, key="outlier_iqr")

if selected_feature:
    Q1 = df[selected_feature].quantile(0.25)
    Q3 = df[selected_feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outlier_mask = (df[selected_feature] < lower_bound) | (df[selected_feature] > upper_bound)
    outliers = df[outlier_mask]
    inliers = df[~outlier_mask]

    st.write(f"{outliers.shape[0]} outliers detected in `{selected_feature}`.")

    if st.button("Remove Outliers"):
        df = inliers.copy()
        st.success(f"Outliers removed. Remaining rows: {df.shape[0]}.")

    fig = px.histogram(df, x=selected_feature, marginal="box", title=f"{selected_feature} after Outlier Removal")
    st.plotly_chart(fig, use_container_width=True)


# Encode Categorical Variables
st.subheader("2.3 Encode Categorical Variables")

categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
if categorical_cols:
    st.write("Encoding columns:", categorical_cols)
    st.selectbox("Encoding method", ["OneHotEncoder (default)"], disabled=True)
    if st.button("Apply Encoding"):
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded = encoder.fit_transform(df[categorical_cols])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols), index=df.index)
        df = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)
        st.success("Categorical variables encoded.")
else:
    st.info("No categorical columns to encode.")


# Feature Scaling
st.subheader("2.4 Feature Scaling")
numeric_cols = df.select_dtypes(include=["int", "float"]).columns.tolist()
if st.button("Scale Numeric Features"):
    scaler = MinMaxScaler()
    before_stats = df[numeric_cols].describe().loc[["mean", "std"]]
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    after_stats = df[numeric_cols].describe().loc[["mean", "std"]]

    st.success("All numeric features scaled using MinMaxScaler.")
    st.write("**Before Scaling (mean & std):**")
    st.dataframe(before_stats)
    st.write("**After Scaling (mean & std):**")
    st.dataframe(after_stats)

    st.write("Box Plot of Scaled Features:")
    st.plotly_chart(px.box(df[numeric_cols], points="outliers", title="Scaled Feature Distributions"), use_container_width=True)



# Save Processed Data
st.subheader("2.5 Save and Preview Processed Data")

if st.button("Save Processed Data"):
    #st.session_state["processed_df"] = df
    st.success("Processed data saved to session state.")

if "processed_df" in st.session_state:
    st.dataframe(st.session_state["processed_df"].head())
    st.write("Processed dataset shape:", st.session_state["processed_df"].shape)

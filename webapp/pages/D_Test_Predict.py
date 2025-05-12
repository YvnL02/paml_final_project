import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.metrics import accuracy_score
from model_func import LogisticRegression, SVM
from webapp.preprocess import Preprocessor

st.set_page_config(layout="wide")
st.title("4. Bot vs Human Predictor")

model_data = joblib.load("trained_model.pkl")
model_type = model_data["type"]
W = model_data["W"]
b = model_data["b"]
feature_names = model_data["feature_names"]

model = LogisticRegression() if model_type == "logistic" else SVM()
prep = Preprocessor().load()

# Load raw dataset
if "raw_df" not in st.session_state:
    st.warning("Please go to page 1 to load the dataset first.")
    st.stop()

df = st.session_state["raw_df"]
X_all = prep.transform(df.drop(columns=["target"]))
y_true = df["target"].values
y_true = np.where(y_true == 0, -1, 1)

# top feature input UI
with st.form("top_features_form"):
    st.markdown("### 🔍 Enter Values for Top Features")

    reposts_ratio = st.slider("Reposts Ratio", 0.0, 1.0, 0.2)
    
    has_status = st.selectbox("Has Status", ["Unknown", "no", "yes"])
    has_personal_data = st.selectbox("Has Personal Data", ["Unknown", "no", "yes"])
    city = st.selectbox("City", ["Saint Petersburg", "Kostomuksha", "Unknown", "Other"])
    has_hometown = st.selectbox("Has Hometown", ["Unknown", "no", "yes"])
    is_blacklisted = st.selectbox("Is Blacklisted", ["Unknown", "no", "yes"])
    is_verified = st.selectbox("Is Verified", ["Unknown", "no", "yes"])
    has_photo = st.selectbox("Has Photo", ["Unknown", "no", "yes"])
    all_posts_visible = st.selectbox("All Posts Visible", ["Unknown", "no", "yes"])

    audio_available = st.selectbox("Audio Available", ["Unknown", "yes", "no"])
    has_website = st.selectbox("Has Website", ["Unknown", "yes", "no"])

    submitted = st.form_submit_button("Predict ➜")

if submitted:
    user_features = pd.DataFrame([np.zeros(len(feature_names))], columns=feature_names)

    # Numerical
    if "reposts_ratio" in user_features.columns:
        user_features["reposts_ratio"] = reposts_ratio
    
    # One-hot
    for col in [
        f"has_status_{'1.0' if has_status == 'yes' else '0.0' if has_status == 'no' else 'Unknown'}",
        f"has_personal_data_{'1.0' if has_personal_data == 'yes' else '0.0' if has_personal_data == 'no' else 'Unknown'}",
        f"city_{city}",
        f"has_hometown_{'1.0' if has_hometown == 'yes' else '0.0' if has_hometown == 'no' else 'Unknown'}",
        f"is_blacklisted_{'1.0' if is_blacklisted == 'yes' else '0.0' if is_blacklisted == 'no' else 'Unknown'}",
                f"is_verified_{'1.0' if is_verified == 'yes' else '0.0' if is_verified == 'no' else 'Unknown'}",
        f"has_photo_{'1.0' if has_photo == 'yes' else '0.0' if has_photo == 'no' else 'Unknown'}",
        f"all_posts_visible_{'1.0' if all_posts_visible == 'yes' else '0.0' if all_posts_visible == 'no' else 'Unknown'}",
        f"audio_available_{'1.0' if audio_available == 'yes' else '0.0' if audio_available == 'no' else 'Unknown'}",
        f"has_website_{'1.0' if has_website == 'yes' else '0.0' if has_website == 'no' else 'Unknown'}"
    ]:
        if col in user_features.columns:
            user_features[col] = 1

# Predict and evaluate

    y_pred = model.predict(user_features.to_numpy(), W, b)

    result = "Bot" if y_pred[0] == 1 else "Human"
    y_all_pred = model.predict(X_all, W, b)
    accuracy = np.mean(y_all_pred == y_true)

    st.success(f"Prediction Result: **{result}**")
    st.info(f"Model accuracy on full dataset: **{accuracy:.2%}**")

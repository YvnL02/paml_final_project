import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from model_func import LogisticRegression, SVM 
from preprocess import Preprocessor

st.set_page_config(layout="wide")
st.title("4. Bot vs Human Predictor")

# Load model
model_data = joblib.load("trained_model.pkl")
model_type = model_data["type"]
W = model_data["W"]
b = model_data["b"]
feature_names = model_data["feature_names"]

model = LogisticRegression() if model_type == "logistic" else SVM()

if "raw_df" not in st.session_state:
    st.warning("Please go to page 1 to load the dataset first.")
    st.stop()

df = st.session_state["raw_df"]

# Load preprocessor
prep = Preprocessor().load()
X_all = prep.transform(df.drop(columns=["target"]))
y_true = df["target"].values
y_true = np.where(y_true == 0, -1, 1)

# Build UI
with st.form("prediction_form"):
    st.subheader("Numerical Inputs")
    friends = st.slider("Friends", 0, 1000, 120)
    posts = st.slider("Posts", 0, 1000, 84)
    hashtags = st.slider("Hashtags", 0, 100, 23)

    st.subheader("Categorical Inputs")
    has_gender = st.radio("Has Gender?", ["yes", "no"])
    has_mobile = st.radio("Has Mobile?", ["yes", "no"])
    has_birthdate = st.radio("Has Birth Date?", ["yes", "no"])
    feature_4 = st.radio("Feature 4", ["type1", "type2", "type3", "type4"])

    st.subheader("Boolean Inputs")
    private_profile = st.radio("Private Profile", ["off", "on"])
    allow_message = st.radio("Allow Message", ["off", "on"])
    allow_post = st.radio("Allow Post", ["off", "on"])

    submit = st.form_submit_button("Predict ➜")

if submit:
    user_features = pd.DataFrame([np.zeros(len(feature_names))], columns=feature_names)

    for name, value in zip(["friends", "posts", "hashtags"], [friends, posts, hashtags]):
        if name in user_features.columns:
            user_features[name] = value

    # categorical fields 
    for col in user_features.columns:
        if "has_gender" in col and has_gender in col:
            user_features[col] = 1
        if "has_mobile" in col and has_mobile in col:
            user_features[col] = 1
        if "has_birthdate" in col and has_birthdate in col:
            user_features[col] = 1
        if "feature_4" in col and feature_4 in col:
            user_features[col] = 1
        if "private_profile" in col and private_profile in col:
            user_features[col] = 1
        if "allow_message" in col and allow_message in col:
            user_features[col] = 1
        if "allow_post" in col and allow_post in col:
            user_features[col] = 1

    # Predict user input
    y_pred = model.predict(user_features.to_numpy(), W, b)
    result = "Bot" if y_pred[0] == 1 else "Human"
    y_all_pred = model.predict(X_all, W, b)
    accuracy = np.mean(y_all_pred == y_true)

    st.success(f"Prediction Result: **{result}**")
    st.info(f"Model accuracy on full dataset: **{accuracy:.2%}**")

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from model_func import LogisticRegression, SVM
from webapp.preprocess import Preprocessor

st.set_page_config(layout="wide")
st.title("3. Train Bot Detection Model")

# Check if raw data is available
if "raw_df" not in st.session_state:
    st.warning("Please go to page 1 to load the dataset first.")
    st.stop()

df = st.session_state["raw_df"]

prep = Preprocessor()
X = prep.fit(df, target_col="target")
y = df["target"].values
y = np.where(y == 0, -1, 1)

prep.save("preprocessor.pkl")
joblib.dump(prep.get_feature_names(), "feature_names.pkl")

# Train/Val Split
st.markdown("### Step 1: Configure Train/Validation Split")
split_ratio = st.slider("Train/Test Split %", min_value=50, max_value=90, value=80)
split_index = int(len(X) * split_ratio / 100)
X_train, X_val = X[:split_index], X[split_index:]
y_train, y_val = y[:split_index], y[split_index:]

st.write(f"Training on {X_train.shape[0]} samples, validating on {X_val.shape[0]} samples")

# Select model
st.markdown("### Step 2: Choose Model and Hyperparameters")
model_type = st.radio("Choose model", ["Logistic Regression", "SVM"])

# Hyperparameter configuration
if model_type == "Logistic Regression":
    lr = st.number_input("Learning rate (α)", value=0.05, format="%.3f")
    n_iter = st.number_input("Number of iterations", min_value=100, max_value=5000, value=1000, step=100)
else:
    svm_lr = st.number_input("Learning rate (η)", value=0.01, format="%.3f")
    svm_lambda = st.number_input("Regularization λ", value=0.1, format="%.3f")
    svm_iter = st.number_input("Number of iterations", min_value=100, max_value=5000, value=1000, step=100)

# Train model
if st.button("Train Model"):
    with st.spinner("Training..."):
        if model_type == "Logistic Regression":
            model = LogisticRegression(learning_rate=lr, num_iterations=n_iter)
            W, b, history = model.fit(X_train, y_train, num_iterations=n_iter, learning_rate=lr)
            st.session_state["model"] = ("logistic", W, b)
        else:
            model = SVM(learning_rate=svm_lr, lambda_param=svm_lambda, n_iterations=svm_iter)
            W, b, history = model.fit(X_train, y_train, lambda_param=svm_lambda,
                                      learning_rate=svm_lr, num_iterations=svm_iter)
            st.session_state["model"] = ("svm", W, b)

    st.success("Training completed.")

    # Show learning curve
    st.markdown("### Training Loss Curve")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Hinge Loss" if model_type == "SVM" else "Log-Likelihood")
    st.pyplot(fig, use_container_width=False)

    # Save model
    model_dict = {
    "type": model_type.lower(),
    "W": W,
    "b": b,
    "feature_names": prep.get_feature_names()
    }
    joblib.dump(model_dict, "trained_model.pkl")
    st.success("Model saved as `trained_model.pkl`")


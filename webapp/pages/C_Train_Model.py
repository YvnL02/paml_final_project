import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from model_func import LogisticRegression, SVM
from webapp.preprocess import Preprocessor

import plotly.express as px
from sklearn.metrics import accuracy_score

from model_func import SVM
from preprocess import Preprocessor

st.set_page_config(layout="wide")
st.title("3. Deploy Model")

# Check dataset
if "raw_df" not in st.session_state:
    st.warning("Please go to page 1 to load the dataset first.")
    st.stop()

df = st.session_state["raw_df"]

# Preprocess
prep = Preprocessor()
X = prep.fit(df, target_col="target")
y = df["target"].values
y = np.where(y == 0, -1, 1)

prep.save("preprocessor.pkl")
joblib.dump(prep.get_feature_names(), "feature_names.pkl")

# Train/Val split
split_index = int(len(X) * 0.8)
X_train, X_val = X[:split_index], X[split_index:]
y_train, y_val = y[:split_index], y[split_index:]

# Show model info
st.markdown("### Selected Model: **Support Vector Machine (SVM)**")
st.markdown("Based on evaluation results, SVM achieved the best F1-score and recall, and is selected for deployment with the following parameters:")

best_params = {
    "Learning rate (η)": 0.01,
    "Regularization (λ)": 0.1,
    "Iterations": 1000
}
for k, v in best_params.items():
    st.markdown(f"- **{k}**: {v}")

if "model_trained" in st.session_state:
    acc = st.session_state.get("val_acc", 0.0)
    st.success(f"✔️ Model already trained! Accuracy: {acc:.2%}")

if st.button("Train Best Model"):
    with st.spinner("Training SVM with fixed parameters..."):
        model = SVM(
            learning_rate=best_params["Learning rate (η)"],
            lambda_param=best_params["Regularization (λ)"],
            n_iterations=best_params["Iterations"]
        )
        W, b, history = model.fit(
            X_train, y_train,
            lambda_param=best_params["Regularization (λ)"],
            learning_rate=best_params["Learning rate (η)"],
            num_iterations=best_params["Iterations"]
        )

    # Validation performance
    y_val_pred = model.predict(X_val, W, b)
    val_acc = accuracy_score(y_val, y_val_pred)

    # Show training loss
    st.markdown(f"**Model training completed! Accuracy: {val_acc:.2%}**")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Hinge Loss")
    st.pyplot(fig, use_container_width=False)

    # Top 10 features
    st.markdown("### Top 10 Features by Importance")
    weights_df = pd.DataFrame({
        "feature": prep.get_feature_names(),
        "weight": W.flatten()
    })
    weights_df["abs_weight"] = weights_df["weight"].abs()
    top_features_df = weights_df.sort_values("abs_weight", ascending=False).head(10)

    fig = px.bar(
        top_features_df,
        x="feature",
        y="abs_weight",
        title="Top 10 Feature Importances (|Weight|)",
        labels={"abs_weight": "Absolute Weight"},
    )
    st.plotly_chart(fig, use_container_width=True)

    # Save model + top features
    model_dict = {
        "type": "svm",
        "W": W,
        "b": b,
        "feature_names": prep.get_feature_names()
    }
    joblib.dump(model_dict, "trained_model.pkl")
    joblib.dump(top_features_df["feature"].tolist(), "top_features.pkl")

    st.session_state["model_trained"] = True
    st.session_state["val_acc"] = val_acc

    st.success("Model saved as `trained_model.pkl`, top features saved as `top_features.pkl`.")

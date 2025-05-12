import streamlit as st

# Homepage title
st.markdown("# Social Bot Detection for VK.com")

# Project Introduction
st.markdown("""
Welcome to the **Social Media Bot Detection App**.

This tool allows you to explore, train, and evaluate machine learning models for detecting bots on [VK.com](https://vk.com), Russia's largest social media platform. Bots on social media pose threats to platform integrity by spreading misinformation, amplifying content unfairly, and impersonating users. Our system provides an interpretable, from-scratch machine learning pipeline for binary classification of user profiles as either **Bot** or **Human**.

### What does this app offer?
- **Data Exploration:** Visualize feature distributions, spot outliers, and compare human vs. bot behavior patterns.
- **Model Training:** Train a Logistic Regression or SVM model from scratch using customizable hyperparameters.
- **Prediction:** Upload or manually input VK profile features to predict whether the account is likely a bot.
- **Result Analysis:** View performance metrics (Accuracy, Precision, Recall, F1-score), and interpret model decisions using feature importance.

This app is designed for **moderators**, **data analysts**, and **researchers** interested in platform transparency and bot detection.

---
""")

st.info("To begin, select an option from the left sidebar.")

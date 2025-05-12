
# Bot Detection on VK.com  
**Machine Learning for Detecting Bots to Enhance Social Media Integrity**

This project implements an SVM-based classifier from scratch to detect bot accounts on VK.com. It features a fully functional  web app for data exploration, model training, and prediction.

---

## Project Structure

```
Streamlit web application files:
webapp/
├── pages/
│   ├── A_Data_Exploration.py        # Interactive visualizations of dataset
│   ├── B_Feature_Engineering.py     # View and apply preprocessing steps
│   ├── C_Train_Model.py             # Train SVM classifier
│   ├── D_Test_Predict.py            # Upload or input user features for prediction
├── Home.py                          # Main landing page of the Streamlit app
├── feature_names.pkl                # Saved list of feature names used in training       
├── preprocess.py                    # Core preprocessing functions
├── preprocessor.pkl                 # Serialized preprocessing pipeline
├── top_features.pkl                 # Feature importance data
├── trained_model.pkl                # Serialized trained SVM model

Jupyter Notebook and dataset:
├── bots_vs_users.csv                # Original dataset
├── code.ipynb                       # Development notebook for experimentation
├── requirements.txt                 # Python dependencies

```

---

How to run the code

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 2. Install dependencies

It’s recommended to use a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Launch the Streamlit app

```bash
streamlit run webapp/Home.py
```

Once the app is running, use the sidebar to navigate between:



## Output Files

- `trained_model.pkl`: Serialized SVM model
- `preprocessor.pkl`: Saved preprocessing pipeline
- `top_features.pkl`: Important features ranked
- `feature_names.pkl`: Ordered feature list used in training

---

## Authors

- Shuchang Cao  
- Lianyan Liu  
- Zeyu Zhang  
- Yifang Liao  
- Zhiyu Lu  

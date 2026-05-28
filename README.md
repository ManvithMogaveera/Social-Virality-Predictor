
# 🚀 Social Virality Predictor

An AI-powered Reddit engagement prediction system that estimates the virality of posts and comments using Natural Language Processing, feature engineering, and XGBoost regression.
![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge\&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-ML_Model-orange?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed_App-red?style=for-the-badge\&logo=streamlit)
![NLP](https://img.shields.io/badge/NLP-TF--IDF_&_Text_Analytics-purple?style=for-the-badge)
![Dataset](https://img.shields.io/badge/Dataset-Reddit_Social_Data-ff4500?style=for-the-badge)
![Model](https://img.shields.io/badge/Model-XGBoost_Regressor-yellow?style=for-the-badge)
![Status](https://img.shields.io/badge/Project-Production_Ready-success?style=for-the-badge)
---
🔗WEBSITE LINK:
##

https://social-virality-predictor.streamlit.app/

---
# 📌 Project Overview

This project predicts how much engagement a Reddit post or comment may receive before publishing.

The model combines:

* NLP-based title analysis
* Comment sentiment
* Posting time behavior
* Parent post momentum
* Engagement heuristics
* TF-IDF textual features

to estimate expected upvotes using machine learning.

---

# 🧠 ML Pipeline

## Data Sources

* `comments.csv`
* `posts.csv`

Dataset sourced from Reddit discussions.

---

## Core Techniques

### 🔹 Feature Engineering

Engineered features include:

* Sentiment intensity
* Word count
* Character count
* Weekend activity
* Peak posting hours
* Comment depth
* Post score influence
* Question-based titles

---

### 🔹 NLP Processing

Used TF-IDF vectorization with:

* Unigrams + bigrams
* English stopword removal
* Top 50 informative terms

---

### 🔹 Model

XGBoost Regressor trained on:

* 43k+ training samples
* Stratified virality classes

Target transformation:

```python
np.log1p(score)
```

---

# 📊 Model Performance

| Metric   | Score  |
| -------- | ------ |
| RMSE     | 0.4298 |
| R² Score | 0.1517 |

### Top Influential Features

* Post score
* Log post score
* COVID-related title terms
* Question-based posts
* Comment length

---

# 📈 Visual Insights

## Score Distribution

Shows long-tail virality behavior typical of social platforms.

## Feature Importance

Highlights which metadata and text patterns most influence engagement.

## Sentiment vs Engagement

Demonstrates weak-to-moderate relationship between emotional intensity and upvotes.

---

# 💻 Streamlit Dashboard

Interactive UI includes:

* Post virality checker
* Comment virality checker
* Real-time engagement prediction
* ML + heuristic hybrid fallback
* Dynamic engagement scoring bars

Run locally:

```bash
streamlit run app.py
```

---

# ⚙️ Installation

```bash
git clone https://github.com/yourusername/social-virality-predictor.git

cd social-virality-predictor

pip install -r requirements.txt
```

---

# ▶️ Train Model

```bash
python train.py
```

---

# ▶️ Run App

```bash
streamlit run app.py
```

---

# 📂 Requirements

Main libraries:

* pandas
* numpy
* scikit-learn
* xgboost
* streamlit
* joblib

---

# ⚠️ Limitations

* Social engagement is highly stochastic
* Reddit trends change rapidly over time
* Model performance varies across subreddits
* External events strongly affect virality
* R² remains moderate due to unpredictable human behavior

---

# 🔮 Future Improvements

* Transformer-based NLP models (BERT)
* Subreddit-specific modeling
* Time-series trend tracking
* Real-time Reddit API integration
* Deep learning sequence models

---

# 👨‍💻 Author

Manvith Mogaveera

Electronics & Telecommunication Engineering student exploring AI/ML, NLP, and intelligent social analytics systems.

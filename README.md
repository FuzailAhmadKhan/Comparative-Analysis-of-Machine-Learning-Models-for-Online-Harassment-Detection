# Comparative Study of Sentiment Analysis Classifiers for Detecting Online Harassment

_A Machine Learning and NLP based comparative analysis of traditional classifiers for detecting hateful, offensive and neutral text in social media._

---

## 🌐 Live Web Interface

The deployed harassment detection demo is hosted on **Netlify & Netlify Hosting** via **Netlify**:
🔗 **Web App:** https://fuzail7thsem.netlify.app/

---

## 🎯 Project Objective

The project aims to build a **reproducible machine learning pipeline** and perform an **algorithmic comparison** of classical text classifiers for detecting online harassment. The focus is on model evaluation over deployment.

Goals include:
- Automated classification into **Hate Speech, Offensive, Neutral**
- Fair performance comparison of multiple learning paradigms
- Interpretability through feature importance, error patterns and confidence behaviour
- Usability through a browser-accessible demo UI connected to backend API

---

## 📊 Model Evaluation Results

Model comparison was performed on a stratified test split using standard NLP features (**TF-IDF unigram + bigram**).

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Random Forest (RF)** | 0.9033 | 0.8895 | 0.9033 | 0.8902 |
| **Support Vector Machine (SVM)** | **0.9044** | 0.8888 | **0.9044** | **0.8915** |
| **Decision Tree (DT)** | 0.8755 | 0.8710 | 0.8755 | 0.8731 |
| **Naïve Bayes (Multinomial NB)** | 0.8640 | 0.8534 | 0.8640 | 0.8386 |
| **K-Nearest Neighbour (KNN)** | 0.6328 | 0.7427 | 0.6328 | 0.6604 |

### 🔍 Key Observations
- **SVM achieved the best overall performance**, particularly in Recall and F1-Balance, making it the most robust for detecting the minority Hate class.
- **Random Forest matched SVM very closely** while additionally providing strong interpretability through feature importance scores.
- **KNN failed to model sparse high-dimensional TF-IDF vectors**, confirming that instance-based distance reasoning struggles under the curse of dimensionality.
- Naïve Bayes and Decision Trees performed reasonably but lacked the stability and contextual boundary precision of SVM and RF.

All metrics can be reproduced by executing the model training script.

---

## 🧠 Feature Importance Snapshot

Feature-level impact for **Random Forest learning** reveals strong lexical cues (slurs, abusive bigrams) influencing model decisions.

```
[Insert feature importance image generated from models folder]
File name: `feature_importance.png`
```

This figure can be generated using the Random Forest joblib model.

---

## 🛠️ Machine Learning Pipeline

### 1. **Data Preprocessing**
Text normalization includes:
- lowercasing, URL removal, mention removal (`@user`), punctuation stripping
- stopword removal, Tokenization and stemming
- lightweight cleaning logic implemented using **regex and NLTK-like pipelines inside data_preprocessing.py**

### 2. **Feature Extraction**
- TF-IDF vectorizer configured with:
  - `ngram_range=(1,2)`
  - `max_features=5000`
  - `min_df=2`, `max_df=0.8`, `sublinear_tf=True`

### 3. **Model Training**
- Classical classifiers trained using **scikit-learn family algorithms inside Python**:
  - RF, SVM (Linear Kernel), Multinomial NB, Decision Tree, KNN
- Train/Test split → **80/20 stratified with fixed random seed**
- Persisted artifacts:
  - Trained models (`.joblib`)
  - TF-IDF vectorizer (`tfidf_vectorizer.joblib`)
  - Metric export CSV (`model_results.csv`)

Execution command:
```bash
python train_model.py
```

### 4. **Model Evaluation**
- Metrics are recomputed on the test split
- Common harmful lexical cues are extracted to support visual interpretation
- Results exported and visualized in front-end via confidence bars and comparison graphs

---

## 🔥 Prediction API (Flask Backend)

### Endpoint
```
POST /predict
```

### Payload format
```json
{ "text": "example message to classify" }
```

### Returns
- `prediction` (hate/offensive/neutral)
- model confidence scores
- predicted label from best model loaded

Example response:
```json
{
  "prediction": "offensive",
  "confidence": {
    "hate": 0.15,
    "offensive": 0.72,
    "neutral": 0.13
  }
}
```

---

## 🖥️ Web UI Demo Usage

The **browser interface built using UI technologies** simulates a moderation dashboard:

Features:
- Text Input Box with character sanitization
- Real time model prediction display
- Confidence bar visualization
- Model performance comparison graphs
- Harassment keyword highlighting (optional)

Front-end uses:
- **HTML**
- **CSS**
- **JavaScript**

These files directly communicate with the backend Flask API to display results.

Deployment link: https://fuzail7thsem.netlify.app/

---

## 📦 Dependencies

Create a `requirements.txt` with:

```
flask
scikit-learn
pandas
numpy
matplotlib
joblib
```

---

## 📚 Dataset Citation (Required)

Use one of the following formats depending on your dataset source.

### If using _Davidson 2017 dataset_:
```
[11] T. Davidson, D. Warmsley, M. Macy, and I. Weber, “Hate Speech and Offensive Langua

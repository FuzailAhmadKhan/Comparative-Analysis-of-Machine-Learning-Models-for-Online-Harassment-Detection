# Comparative Analysis of Machine Learning Models for Online Harassment Detection

Interactive Web Interface Link: https://fuzail7thsem.netlify.app/

---

## 🚀 Project Overview

This project builds a complete pipeline to detect online harassment using traditional ML classifiers. The goal is to compare model performance rather than deploy a single model.

### ✅ Key Highlights：
- Text preprocessing pipeline
- TF–IDF vectorization (1–2 grams)
- Training & evaluation of 5 classifiers
- Flask API backend
- Browser UI for real-time text input and prediction

### 🤖 Models Compared：
- Random Forest Classifier (RF)
- Support Vector Machine (SVM)
- Multinomial Naïve Bayes (MNB)
- Decision Tree (DT)
- K-Nearest Neighbours (KNN)

### ⭐ Best Results Achieved：
- **SVM** → Highest accuracy + most reliable across metrics
- **Random Forest** → Very close second, and most interpretable via feature importance

---

## 📁 Project Structure

```
📦 repository
├── index.html              # Front-end UI
├── styles.css             # UI styles
├── script.js              # Front-end logic
├── app.py                 # Flask Prediction API
├── data_preprocessing.py  # Text cleaning & processing
├── train_model.py         # Model training & metric export
└── model_results.csv      # Accuracy, precision, recall, F1 values
```

---

## ⚙️ Setup & Installation

Clone the project：

```bash
git clone https://github.com/your-username/harassment-detection.git
cd harassment-detection
```

Create virtual environment (optional but recommended)：

```bash
python -m venv env
source env/bin/activate  # Mac/Linux
env\Scripts\activate     # Windows
```

Install dependencies：

```bash
pip install flask scikit-learn pandas numpy matplotlib joblib
```

---

## 🛠️ Train All Models

Run the training script：

```bash
python train_model.py
```

This will：

✔ Train all 5 classifiers on TF-IDF vectors  
✔ Generate evaluation metrics  
✔ Save models (`.joblib`) and TF-IDF vectorizer  
✔ Update `model_results.csv`

---

## 🔥 Run the Flask API

Start the prediction API：

```bash
python app.py
```

Endpoint will run locally at:

```
http://127.0.0.1:5000/predict
```

### Example API Request：

````json
POST /predict
{
  "text": "You are a disgusting human"
}

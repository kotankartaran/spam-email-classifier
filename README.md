# 📧 Spam Email Classifier

A Machine Learning project that detects whether an email or SMS message is **Spam** or **Legitimate (Ham)** using Natural Language Processing (NLP) and the Multinomial Naive Bayes algorithm.

---

## 🚀 Project Overview

Spam messages are one of the most common problems in digital communication. This project uses Machine Learning and NLP techniques to automatically classify messages as:

- ✅ Legitimate (Ham)
- 🚨 Spam

The model is trained on the SMS Spam Collection Dataset and deployed using Streamlit for an interactive user experience.

---

## 🎯 Features

- Spam Detection using Machine Learning
- Natural Language Processing (NLP)
- TF-IDF Text Vectorization
- Interactive Streamlit Dashboard
- Real-Time Prediction
- Probability-Based Results
- User-Friendly Interface

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- NLP (TF-IDF)
- Streamlit
- Joblib

---

## 🤖 Machine Learning Model

### Algorithm
**Multinomial Naive Bayes**

### Text Processing
**TF-IDF Vectorization**

### Workflow

```text
Message
   ↓
TF-IDF Vectorizer
   ↓
Feature Extraction
   ↓
Multinomial Naive Bayes
   ↓
Spam / Ham Prediction
```

---

## 📊 Dataset

Dataset Used:

**SMS Spam Collection Dataset**

- Total Messages: 5,572
- Spam Messages: 747
- Legitimate Messages: 4,825

Dataset Source:

https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

---

## 📈 Model Performance

### Accuracy

```text
97.40%
```

### Classification Report

```text
Precision: 99%
Recall: 81%
F1 Score: 89%
```

---

## 📸 Screenshots

### Home Page
![Home Page](dashboard1.png)

### Spam Message Detection
![Spam Detection](dashboard2.png)

### Legitimate Message Detection
![Legitimate Message](dashboard3.png)

---

## 📂 Project Structure

```text
spam-email-classifier/
│
├── data/
│   └── spam.csv
│
├── models/
│   ├── spam_model.pkl
│   └── vectorizer.pkl
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/spam-email-classifier.git
```

### Open Project

```bash
cd spam-email-classifier
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Mac/Linux

```bash
source venv/bin/activate
```

Windows

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Train Model

```bash
python train.py
```

---

## 🌐 Run Streamlit App

```bash
streamlit run app.py
```

---

## 🧪 Sample Messages

### Spam Example

```text
Congratulations! You have won a FREE iPhone.
Click here to claim your prize now.
```

### Legitimate Example

```text
Hi Taran, the meeting is scheduled for tomorrow at 10 AM.
```

---

## 🎓 Learning Outcomes

This project helped in understanding:

- Natural Language Processing (NLP)
- Text Vectorization using TF-IDF
- Naive Bayes Classification
- Model Training & Evaluation
- Streamlit Deployment
- Machine Learning Project Workflow

---

## 👨‍💻 Author

**Taran Kotankar**

GitHub:
https://github.com/kotankartaran



---

## ⭐ If you like this project

Give this repository a star ⭐ and support the project.
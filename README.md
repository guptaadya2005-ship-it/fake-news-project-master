# 🔍 Drishya – Fake News Detection System

Drishya is an AI-powered fake news detection system designed to analyze news content and determine whether it is **Real or Fake** using Deep Learning and Natural Language Processing.

## 🚀 Live Demo

- 🌐 **Live Demo:** [Add your deployed link here]
- 💻 **GitHub Repository:** https://github.com/guptaadya2005-ship-it/fake-news-project-master

---

## ✨ Features

- 📰 **Fake News Detection**
  - Classifies news as **Real** or **Fake**.

- 🤖 **Deep Learning Based Analysis**
  - Uses a transformer-based NLP model for news classification.

- 🧠 **DistilBERT Model**
  - Uses DistilBERT for understanding and analyzing textual news content.

- 🔎 **Evidence Verification**
  - Performs an additional evidence-based verification step for real-world news.

- 📊 **Prediction Result**
  - Provides the predicted category along with the analysis result.

- 🌐 **Interactive Web Interface**
  - User-friendly interface for entering and analyzing news.

- ⚡ **Real-Time Analysis**
  - Processes the submitted news and generates a prediction.

---

## 🛠️ Tech Stack

### Frontend
- HTML
- CSS
- JavaScript
- Bootstrap / Custom CSS

### Backend
- Python
- Flask
- REST APIs

### Machine Learning & NLP
- PyTorch
- Hugging Face Transformers
- DistilBERT
- RoBERTa
- Natural Language Processing (NLP)

### Dataset
- Fake News Dataset
- `Fake.csv`
- `True.csv`

### Database / Storage
- Local file-based dataset and model storage

### Development Tools
- Git
- GitHub
- VS Code

---

## 🧠 Machine Learning Pipeline

- 📥 News article is provided as input.
- 🧹 Text is processed and prepared for analysis.
- 🔤 Tokenization is performed using the transformer tokenizer.
- 🤖 DistilBERT analyzes the textual content.
- 📊 The trained classifier predicts **Fake / Real**.
- 🔎 An additional evidence-checking step can be used for real-world evaluation.
- ✅ Final result is displayed to the user.

---

## 📁 Project Structure

```text
fake-news-project-master/
│
├── app/
│   ├── templates/
│   ├── static/
│   └── ...
│
├── deep_learning/
│   ├── model_training/
│   ├── real_world_evaluation_v2.py
│   └── ...
│
├── model_training/
│   ├── Fake.csv
│   └── True.csv
│
├── bert_model/
│   └── README / model files
│
├── datasets/
│
├── requirements.txt
├── .gitignore
├── README.md
└── run.py
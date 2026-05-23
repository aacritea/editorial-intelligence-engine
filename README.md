---
title: Editorial Intelligence Engine
emoji: 📰
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.45.1
app_file: dashboard.py
pinned: false
---

# 📰 Editorial Intelligence Engine

> AI-powered editorial intelligence platform for engagement-aware news ranking and front-page optimization using NLP, behavioral analytics, and learning-to-rank systems.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-green)
![NLP](https://img.shields.io/badge/NLP-SentenceTransformers-orange)
![Status](https://img.shields.io/badge/Status-Active-success)

---

# 📌 Overview

Editorial Intelligence Engine is a machine learning system designed to simulate AI-assisted newsroom decision-making by predicting article engagement and optimizing front-page story placement.

Traditional editorial workflows rely heavily on manual curation and subjective prioritization. This project introduces a scalable, data-driven ranking pipeline that leverages:

- NLP feature engineering
- behavioral interaction analytics
- learning-to-rank models
- engagement prediction
- front-page visibility optimization

to dynamically rank stories for maximum audience traction.

---

# ✨ Features

- 🧠 Learning-to-rank pipeline using LightGBM Ranker
- 📰 Newspaper-style Streamlit dashboard
- 📊 Engagement-aware article ranking
- 🔍 Explainable ranking decisions
- 📈 Behavioral analytics and CTR modeling
- 🧾 NLP feature engineering for headlines
- ⚡ Visibility-aware front-page optimization
- 🎯 Ranking evaluation using NDCG@K and MAP
- 🧪 Synthetic fallback data generation for demos

---

# 🏗️ System Architecture

```text
News Articles
      ↓
Data Preprocessing Pipeline
      ↓
NLP Feature Engineering
      ↓
Engagement Prediction
      ↓
Learning-to-Rank Engine
      ↓
Front-Page Optimization
      ↓
Interactive Editorial Dashboard
```

---

# 🧠 Core Concepts

This project focuses on:

- Editorial Intelligence
- Learning-to-Rank Systems
- Recommendation Systems
- Engagement Prediction
- NLP Feature Engineering
- Behavioral Analytics
- Front-Page Optimization

Unlike traditional classification-based news systems, this platform treats editorial ranking as a ranking optimization problem.

---

# 📂 Project Structure

```text
editorial-intelligence-engine/
│
├── data/
│   ├── raw/
│   │   ├── news.tsv
│   │   └── behaviors.tsv
│   │
│   └── processed/
│       └── mind_clean.parquet
│
├── models/
│
├── notebooks/
│
├── src/
│   ├── mind_preprocessing.py
│   ├── feature_engineering.py
│   ├── embeddings.py
│   ├── ranker.py
│   ├── optimizer.py
│   ├── inference.py
│   └── dashboard.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 📊 Dataset

## Microsoft MIND Dataset

This project uses the Microsoft MIND dataset, a large-scale benchmark dataset for news recommendation and ranking systems.

### Includes

- News headlines
- Article metadata
- User impressions
- Click interactions
- Reading history
- Engagement signals

### Dataset Link

https://msnews.github.io/

---

# ⚙️ Tech Stack

## Machine Learning

- LightGBM Ranker
- XGBoost
- Scikit-learn

## NLP

- Sentence Transformers
- Hugging Face Transformers
- spaCy

## Backend

- Python
- Pandas
- NumPy

## Frontend

- Streamlit
- Plotly

---

# 🔬 Feature Engineering

The ranking engine extracts a combination of linguistic, behavioral, and contextual features.

## Headline Features

- Headline length
- Sentiment score
- Readability score
- Clickbait probability
- Urgency score
- Punctuation intensity

## Article Features

- Topic/category
- Named entities
- Recency
- Metadata encoding

## Behavioral Features

- Historical CTR
- Global click statistics
- Category popularity
- User interaction frequency
- Session diversity metrics

---

# 📈 Ranking Models

The project prioritizes ranking systems instead of standard classification models.

## Models Used

- LightGBM Ranker
- XGBoost Ranker
- LambdaMART

## Embeddings

Sentence embeddings are generated using:

```python
all-MiniLM-L6-v2
```

via Sentence Transformers.

---

# 📊 Ranking Metrics

Evaluation focuses on ranking quality rather than binary accuracy.

## Metrics

- NDCG@K
- MAP
- MRR

These metrics better reflect real-world editorial ranking performance.

---

# 📰 Editorial Dashboard

The Streamlit dashboard provides:

- Front-page article ranking visualization
- Engagement score analytics
- Explainable ranking insights
- Topic/category filtering
- Visibility-aware layout scoring
- Interactive newspaper-style UI

---

# 🚀 Installation

## 1. Clone Repository

```bash
git clone https://github.com/aacritea/editorial-intelligence-engine.git
cd editorial-intelligence-engine
```

---

## 2. Create Virtual Environment

### Mac/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Download spaCy Model

```bash
python -m spacy download en_core_web_sm
```

---

# 🧪 Data Preprocessing

Run preprocessing pipeline:

```bash
python src/mind_preprocessing.py \
  --news data/raw/news.tsv \
  --behaviors data/raw/behaviors.tsv \
  --output data/processed/mind_clean.parquet
```

---

# ▶️ Run Dashboard

```bash
streamlit run src/dashboard.py
```

---

# 📌 Sample Engineered Features

| Feature | Description |
|---|---|
| `global_ctr` | Historical click-through rate |
| `history_len` | User reading history length |
| `session_size` | Number of candidate articles |
| `title_len` | Headline character length |
| `history_cat_diversity` | Diversity of user reading interests |

---

# 🎯 Project Goals

This project was designed to demonstrate:

- Practical ML engineering
- Ranking systems
- Recommendation pipelines
- Behavioral interaction modeling
- Explainable AI systems
- NLP feature engineering
- Product-oriented AI design

---

# 🔮 Future Improvements

- Real-time news ingestion APIs
- Personalized recommendation systems
- Temporal trend forecasting
- Multi-objective optimization
- Reinforcement-learning-based layout optimization
- A/B testing simulation
- User personalization engine

---

# 📚 Research & Engineering Focus

The project emphasizes:

- scalable ranking systems
- recommendation system engineering
- practical NLP pipelines
- engagement-aware optimization
- production-style ML architecture

rather than heavyweight deep learning experimentation.

---

# 👩‍💻 Author

## Aakriti Jain

- GitHub: https://github.com/aacritea
- Portfolio: https://aacritea.framer.wiki
- LinkedIn: https://linkedin.com/in/aacritea

---

# ⭐ Acknowledgements

- Microsoft Research for the MIND Dataset
- Hugging Face
- Streamlit
- LightGBM
- Sentence Transformers

---
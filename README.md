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
- transformer embeddings
- learning-to-rank models
- engagement prediction
- front-page visibility optimization

to dynamically rank stories for maximum audience traction.

---

# ✨ Features

- 🧠 LightGBM LambdaRank learning-to-rank pipeline
- 📰 Newspaper-style Streamlit editorial dashboard
- 📊 Engagement-aware article ranking
- 🔍 Explainable AI ranking insights
- 📈 Behavioral analytics and CTR modeling
- 🧾 NLP-driven headline intelligence
- ⚡ Visibility-aware front-page optimization
- 🎯 NDCG and MRR ranking evaluation
- 🧪 Semantic embeddings using Sentence Transformers
- 📉 Feature importance analytics
- 🧠 Editorial-quality scoring engine
- 🗂️ Category-aware article zoning system

---

# 🏗️ System Architecture

```text
News Articles
      ↓
Data Preprocessing Pipeline
      ↓
Editorial NLP Feature Engineering
      ↓
Sentence Embedding Generation
      ↓
Behavioral Signal Aggregation
      ↓
LightGBM LambdaRank Model
      ↓
Front-Page Optimization Engine
      ↓
Interactive Editorial Dashboard
```

---

# 🧠 Core Concepts

This project focuses on:

- Editorial Intelligence
- Recommendation Systems
- Learning-to-Rank Systems
- Engagement Prediction
- NLP Feature Engineering
- Semantic Search
- Behavioral Analytics
- Explainable AI
- Front-Page Optimization

Unlike traditional classification-based news systems, this platform treats editorial ranking as a ranking optimization problem.

---

# 📂 Project Structure

```text
news_editorial/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── cache/
│
├── models/
│
├── dashboard.py
├── embeddings.py
├── feature_engineering.py
├── mind_preprocessing.py
├── optimizer.py
├── ranker.py
│
├── requirements.txt
├── runtime.txt
├── Dockerfile
└── README.md
```

---

# 📊 Dataset

## Microsoft MIND Dataset

This project uses the Microsoft MIND dataset, a large-scale benchmark dataset for news recommendation and editorial ranking systems.

### Includes

- News headlines
- Article metadata
- User impressions
- Click interactions
- Reading history
- Behavioral engagement signals

### Dataset Scale

- 11.2M+ impression–article interactions
- 100K+ news articles
- Large-scale clickstream behavior
- Multi-category news coverage

### Dataset Link

https://msnews.github.io/

---

# ⚙️ Tech Stack

## Machine Learning

- LightGBM Ranker
- LambdaMART
- Scikit-learn

## NLP

- Sentence Transformers
- Hugging Face Transformers
- spaCy

## Backend

- Python
- Pandas
- NumPy
- PyArrow

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
- Transformer embeddings

## Behavioral Features

- Historical CTR
- Global click statistics
- Category popularity
- Session diversity
- User interaction frequency
- Reading history depth

---

# 🧠 Embedding Pipeline

Semantic article representations are generated using:

```python
all-MiniLM-L6-v2
```

via Sentence Transformers.

## Embedding Optimizations

- Deduplicated embedding generation
- MPS acceleration on Apple Silicon
- Cached transformer inference
- Reduced-dimensional ranking embeddings
- Title + abstract fused embeddings

---

# 📈 Ranking Models

The project prioritizes ranking systems over standard classification models.

## Models Used

- LightGBM Ranker
- LambdaMART
- Gradient Boosted Decision Trees

## Objective

The system optimizes article ranking quality rather than binary prediction accuracy.

---

# 📊 Model Evaluation

Evaluation focuses on ranking quality and recommendation effectiveness.

## Final Validation Metrics

| Metric | Score |
|---|---|
| NDCG@5 | 0.155 |
| NDCG@10 | 0.163 |
| NDCG@20 | 0.164 |
| MRR | 0.146 |

## Training Dataset Scale

| Component | Scale |
|---|---|
| Impression–Article Pairs | 11.2M+ |
| Training Rows | 1.5M |
| Unique Articles | 100K+ |
| Embedding Dimensions | 64 |
| Engineered Features | 70+ |

## Top Predictive Features

| Feature | Importance |
|---|---|
| global_ctr | Highest |
| global_clicks | Very High |
| global_impressions | Very High |
| session_size | High |
| history_len | Moderate |
| semantic embeddings | Significant |

The ranking model demonstrates strong engagement prediction capability while leveraging semantic and behavioral ranking signals.

---

# 📰 Editorial Dashboard

The Streamlit dashboard provides:

- Front-page article ranking visualization
- Engagement score analytics
- Explainable ranking decisions
- Topic/category filtering
- Visibility-aware layout scoring
- Feature importance analytics
- Interactive newspaper-style UI
- Real-time editorial controls

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

# 🧪 Full ML Pipeline

## 1. Preprocess MIND Dataset

```bash
python mind_preprocessing.py \
  --news MINDlarge_train/news.tsv \
  --behaviors MINDlarge_train/behaviors.tsv \
  --output data/processed/mind_clean.parquet
```

---

## 2. Generate Semantic Embeddings

```bash
python embeddings.py \
  --input data/processed/mind_clean.parquet \
  --output data/processed/mind_embedded.parquet \
  --fuse-abstract \
  --device mps \
  --batch-size 128
```

---

## 3. Train Ranking Model

```bash
python ranker.py train \
  --input data/processed/mind_embedded.parquet \
  --model-dir models/ranker \
  --num-leaves 31 \
  --lr 0.05 \
  --n-estimators 800
```

---

## 4. Launch Dashboard

```bash
streamlit run dashboard.py
```

---

# 📌 Sample Engineered Features

| Feature | Description |
|---|---|
| global_ctr | Historical click-through rate |
| history_len | Reading history depth |
| session_size | Candidate article count |
| history_cat_diversity | Diversity of user interests |
| emb_23 | Learned semantic representation |
| emb_43 | Learned semantic representation |

---

# 🎯 Project Goals

This project was designed to demonstrate:

- Learning-to-rank systems
- Practical ML engineering
- Recommendation pipelines
- NLP feature engineering
- Behavioral interaction modeling
- Explainable AI systems
- Editorial optimization workflows
- Product-oriented AI design

---

# 🔮 Future Improvements

- Personalized recommendation systems
- Real-time news ingestion APIs
- Cross-encoder reranking
- Temporal trend forecasting
- Reinforcement-learning layout optimization
- Multi-objective ranking systems
- A/B testing simulation
- User personalization engine

---

# 📚 Research & Engineering Focus

The project emphasizes:

- scalable ranking systems
- recommendation engineering
- semantic retrieval pipelines
- engagement-aware optimization
- explainable editorial AI
- production-style ML architecture

rather than heavyweight deep learning experimentation.

---

# 🌐 Live Demo

## Hugging Face Space

https://aacritea-editorial-intelligence-engine.hf.space

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
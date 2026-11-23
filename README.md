# Fake News Detection via Text Feature Integration and Convolutional Neural Networks

<p align="center">
  <strong>A Deep Learning Approach for Automated Fake News Classification</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Domain-NLP-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Deep%20Learning-CNN-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Task-Fake%20News%20Detection-red?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Python-ML-yellow?style=for-the-badge&logo=python" />
</p>

## 📌 Overview

This repository contains the implementation of our research work, **"Fake News Detection via Text Feature Integration and Convolutional Neural Networks."**

The rapid spread of misinformation across digital platforms has created a growing need for reliable and automated fake news detection systems.

This research proposes a custom Convolutional Neural Network (CNN) architecture that integrates textual information from both news titles and full articles to distinguish between fake and real news.

The proposed model uses an embedding layer followed by multiple one-dimensional convolutional layers to extract meaningful linguistic patterns, contextual information, and semantic features from news content.

The model achieves a reported accuracy of **99.03%**, demonstrating the effectiveness of CNN-based approaches for automated text classification.

## 🎯 Research Objectives

- Develop a custom CNN architecture for fake news detection.
- Integrate textual features from news titles and full articles.
- Extract meaningful linguistic and semantic patterns using convolutional layers.
- Identify deceptive narratives and stylistic irregularities in news content.
- Compare the proposed model with traditional machine learning classifiers.
- Develop an efficient deep learning framework for automated fake news classification.

## 🧠 Proposed CNN Architecture

The proposed model follows a sequential deep learning architecture designed for binary text classification.

### Model Components

**1. Text Preprocessing**

- Tokenization of news articles.
- Integer encoding of text sequences.
- Sequence padding to a fixed length of 200 tokens.

**2. Embedding Layer**

- Embedding dimension: 100
- Vocabulary size: vocab_size + 1
- Converts integer-encoded words into dense vector representations.

**3. Convolutional Layers**

The model uses two consecutive 1D convolutional layers:

| Layer | Filters | Kernel Size | Activation |
|---|---:|---:|---|
| Conv1D | 128 | 5 | ReLU |
| MaxPooling1D | - | 2 | - |
| Conv1D | 256 | 3 | ReLU |
| MaxPooling1D | - | 2 | - |

These layers capture local and fine-grained linguistic patterns from the news text.

**4. Global Max Pooling**

GlobalMaxPooling1D extracts the most prominent features from the convolutional feature maps and produces a fixed-size representation.

**5. Dense Layers and Dropout**

- Dense layer: 128 neurons, ReLU activation.
- Dropout: 0.5.
- Dense layer: 64 neurons, ReLU activation.
- Dropout: 0.3.

**6. Output Layer**

A single neuron with sigmoid activation generates a probability score between 0 and 1.

- Score above 0.5: Fake News.
- Score at or below 0.5: Real News.

## 🔄 Model Workflow

```text
     News Dataset
          |
          v
   Data Loading
          |
          v
   Text Preprocessing
          |
          v
     Tokenization
          |
          v
    Sequence Padding
     (Length = 200)
          |
          v
    Embedding Layer
          |
          v
   Conv1D (128 Filters)
          |
          v
    Max Pooling
          |
          v
   Conv1D (256 Filters)
          |
          v
    Max Pooling
          |
          v
 Global Max Pooling
          |
          v
    Dense + Dropout
          |
          v
    Dense + Dropout
          |
          v
  Sigmoid Classification
          |
          v
     Fake / Real News
```

## 📊 Dataset Description

The research uses a large-scale dataset containing real and fake news articles sourced from two separate CSV files:

- `Fake.csv`
- `True.csv`

The dataset contains news titles and full article content.

Both textual sources are integrated to provide the model with more comprehensive linguistic information.

## ⚙️ Model Training Configuration

| Parameter | Configuration |
|---|---|
| Architecture | Custom CNN |
| Embedding Dimension | 100 |
| Input Sequence Length | 200 |
| Optimizer | Adam |
| Loss Function | Binary Cross-Entropy |
| Batch Size | 64 |
| Training Epochs | 5 |
| Output Activation | Sigmoid |
| Classification | Binary |

## 📈 Experimental Results

The proposed CNN model was evaluated against traditional machine learning classifiers.

### Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 93.47% | 93.11% | 92.85% | 92.98% |
| Naive Bayes | 91.29% | 90.74% | 89.92% | 90.32% |
| Random Forest | 95.26% | 94.98% | 94.60% | 94.78% |
| **Proposed CNN** | **99.03%** | **98.89%** | **98.94%** | **98.90%** |

### Class-wise Performance

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Fake News | 99.61% | 98.30% | 98.95% |
| Real News | 98.16% | 99.58% | 98.86% |
| Average | 98.89% | 98.94% | 98.90% |

The model demonstrates strong classification performance across both fake and real news categories.

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Training and Validation Accuracy
- Training and Validation Loss

## 📉 Visualizations

The research includes:

- Proposed CNN architecture.
- Fake news detection workflow.
- Confusion matrix.
- Training and validation accuracy curves.
- Training and validation loss curves.
- Comparative distribution of real and fake news topics.

## 🛠️ Technologies and Tools

- Python
- Convolutional Neural Networks (CNN)
- Natural Language Processing (NLP)
- Keras
- Deep Learning
- Text Tokenization
- Word Embeddings
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## 📁 Repository Structure

```text
Fake-News-Detection-CNN/
│
├── README.md
│
├── notebooks/
│   └── fake_news_detection.ipynb
│
├── data/
│   └── README.md
│
├── results/
│   ├── confusion_matrix.png
│   ├── accuracy.png
│   ├── loss.png
│   └── topic_distribution.png
│
├── requirements.txt
│
└── models/
    └── README.md
```

*Note: This is a suggested structure. Adjust the filenames and directories according to the actual implementation.*

## 🚀 Getting Started

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd YOUR_REPOSITORY_NAME
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open the fake news detection notebook and follow the implementation steps.

### Dataset Preparation

Place the following files in the dataset directory:

- `Fake.csv`
- `True.csv`

Follow the dataset paths specified in the notebook.

## 🔬 Limitations and Future Research

The current research primarily focuses on English-language news content.

Potential future improvements include:

- Extending the model to multilingual news detection.
- Evaluating performance across different domains and news sources.
- Integrating textual and visual information.
- Improving robustness against adversarial text manipulation.
- Exploring transformer-based architectures.
- Evaluating generalization on independent and more recent datasets.

## 📄 Research Paper

**Title:** Fake News Detection via Text Feature Integration and Convolutional Neural Networks

**Research Area:** Natural Language Processing and Deep Learning

**Main Contribution:** A custom CNN architecture integrating news titles and article text for automated fake news classification.

*Add the official publication venue, year, DOI, and author information here if applicable.*

## ⭐ Acknowledgements

This research explores the application of deep learning and natural language processing to address misinformation through automated text classification.

If you find this project useful, consider giving this repository a ⭐ star.

---

**Research Areas:** Artificial Intelligence | Natural Language Processing | Deep Learning | CNN | Fake News Detection | Text Classification

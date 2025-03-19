# sentimental_analysis
# Sentiment Analysis in Python

## Overview
This project is a sentiment analysis tool that classifies text reviews as positive or negative. It utilizes Natural Language Processing (NLP) techniques with machine learning models to predict sentiment. The dataset consists of user reviews with labeled sentiments.

## Features
- Preprocessing of text data (lowercasing, punctuation removal, etc.)
- TF-IDF Vectorization for text representation
- Logistic Regression model for classification
- Accuracy evaluation using Scikit-learn

## Installation
To run this project, install the necessary dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib nltk
```

## Dataset
Ensure your dataset (`review.csv`) contains the following columns:
- `review`: The text data containing user reviews
- `sentiment`: Labels indicating sentiment ("positive" or "negative")

## Usage
1. Load the dataset and preprocess text:
   - Convert text to lowercase
   - Remove punctuation
   - Map sentiment labels to binary values (1 for positive, 0 for negative)

2. Train the machine learning model using TF-IDF and Logistic Regression.
3. Evaluate the model using accuracy metrics.

## Running the Project
To run the project, execute the Python script containing the following:
```python
import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(f"[{string.punctuation}]", "", text)
        return text
    return ""

df = pd.read_csv("review.csv")
df["review"] = df["review"].apply(preprocess_text)
df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})
df = df.dropna()
```

## Results
The trained model achieves high accuracy in predicting sentiment and can be further improved with additional data and hyperparameter tuning.

## Future Improvements
- Implement deep learning models for better accuracy
- Integrate with a web interface for real-time sentiment analysis

## License
This project is open-source and available under the MIT License.


import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import re
import os

# ============================================
# TASK 2: Apply FinBERT on Recent Headlines
# ============================================

model_path = "./finbert_finetuned"

# Check if model directory exists
if not os.path.exists(model_path) or not os.path.isdir(model_path):
    print(f"ERROR: Model directory '{model_path}' not found!")
    print(f"   Please train the model first by running: python train.py")
    exit(1)

# Check if required model files exist
required_files = ["config.json", "pytorch_model.bin"]
if not all(os.path.exists(os.path.join(model_path, f)) for f in required_files):
    # Try alternative file names (safetensors format)
    if not os.path.exists(os.path.join(model_path, "model.safetensors")):
        print(f"ERROR: Model files not found in '{model_path}'!")
        print(f"   Please train the model first by running: python train.py")
        exit(1)

print(f"Loading model from '{model_path}'...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True, num_labels=3)
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Make sure you have run train.py first to create the model.")
    exit(1)

headlines_df = pd.read_csv("data/merged_headlines_sp500.csv")
print(f"\nLoaded {len(headlines_df)} headlines")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\$[A-Za-z]+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

headlines_df['clean_headline'] = headlines_df['headline'].apply(clean_text)

def predict_sentiment(text):
    inputs = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = predictions[0][predicted_class].item()
    
    return predicted_class, confidence

print("\nPredicting sentiment for headlines...")
sentiments = []
confidences = []

for idx, text in enumerate(headlines_df['clean_headline']):
    if idx % 200 == 0:
        print(f"Processing {idx}/{len(headlines_df)}")
    
    sentiment, confidence = predict_sentiment(text)
    sentiments.append(sentiment)
    confidences.append(confidence)

headlines_df['sentiment_prediction'] = sentiments
headlines_df['sentiment_confidence'] = confidences

sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
headlines_df['sentiment_label'] = headlines_df['sentiment_prediction'].map(sentiment_map)

print("\nSentiment Distribution:")
print(headlines_df['sentiment_label'].value_counts())

headlines_df.to_csv("headlines_with_sentiment.csv", index=False)
print("\nSaved: headlines_with_sentiment.csv")

# ============================================
# TASK 3: Daily Sentiment & Market Trend
# ============================================

headlines_df['date'] = pd.to_datetime(headlines_df['date'])
headlines_df['date_only'] = headlines_df['date'].dt.date

daily_sentiment = headlines_df.groupby('date_only').agg({
    'sentiment_prediction': 'mean',
    'sentiment_confidence': 'mean',
    'sp500_return': 'first',
    'sp500_close': 'first'
}).reset_index()

daily_sentiment['sentiment_score'] = daily_sentiment['sentiment_prediction'] - 1

def calc_weighted_sentiment(group):
    weights = group['sentiment_confidence']
    scores = group['sentiment_prediction'] - 1
    return np.average(scores, weights=weights)

daily_weighted = headlines_df.groupby('date_only').apply(calc_weighted_sentiment).reset_index()
daily_weighted.columns = ['date_only', 'weighted_sentiment']
daily_sentiment = daily_sentiment.merge(daily_weighted, on='date_only')

print("\nDaily Sentiment Summary:")
print(daily_sentiment.head(10))

valid_data = daily_sentiment.dropna(subset=['sentiment_score', 'sp500_return'])

correlation = None
if len(valid_data) > 1:
    correlation, p_value = pearsonr(valid_data['sentiment_score'], valid_data['sp500_return'])
    print(f"\nCorrelation Analysis:")
    print(f"Pearson Correlation: {correlation:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    weighted_corr, weighted_p = pearsonr(valid_data['weighted_sentiment'], valid_data['sp500_return'])
    print(f"\nWeighted Sentiment Correlation: {weighted_corr:.4f}")
    print(f"P-value: {weighted_p:.4f}")
else:
    print("\nWarning: Not enough valid data for correlation analysis")

daily_sentiment.to_csv("daily_sentiment_sp500.csv", index=False)
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Sentiment over time
axes[0].plot(daily_sentiment['date_only'], daily_sentiment['sentiment_score'], 
             marker='o', color='blue', alpha=0.7)
axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Sentiment Score')
axes[0].set_title('Daily Average Sentiment Over Time')
axes[0].grid(True, alpha=0.3)
axes[0].tick_params(axis='x', rotation=45)

# Sentiment vs Returns
axes[1].scatter(daily_sentiment['sentiment_score'], daily_sentiment['sp500_return'], alpha=0.6)
axes[1].set_xlabel('Sentiment Score')
axes[1].set_ylabel('S&P 500 Daily Return (%)')
corr_label = f" (Correlation: {correlation:.4f})" if correlation is not None else ""
axes[1].set_title(f'Sentiment vs S&P 500 Returns{corr_label}')
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[1].axvline(x=0, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('sentiment_analysis_plots.png', dpi=300, bbox_inches='tight')

import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import torch


# load validation data
val_path = "data/val_processed.csv"
val_df = pd.read_csv(val_path)
val_df = val_df.rename(columns={"processed_text": "text", "sentiment_encoded": "label"})

# remove rows with missing values
val_df = val_df.dropna(subset=["text", "label"])

# create dataset
val_dataset = Dataset.from_pandas(val_df)

# load the fine-tuned model and tokenizer
model_path = "./finbert_finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)


def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )


# tokenize validation dataset
tokenized_val_dataset = val_dataset.map(tokenize, batched=True)
tokenized_val_dataset = tokenized_val_dataset.remove_columns(["text"])
tokenized_val_dataset = tokenized_val_dataset.rename_column("label", "labels")
tokenized_val_dataset.set_format("torch")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    # calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')

    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


# run evaluation
print("Evaluating model...")
eval_results = trainer.evaluate(eval_dataset=tokenized_val_dataset)



# print results
print("\nResults:")
for metric, value in eval_results.items():
    print(f"{metric}: {value:.4f}")



# get predictions for detailed analysis
predictions = trainer.predict(tokenized_val_dataset)
predicted_labels = np.argmax(predictions.predictions, axis=1)
true_labels = predictions.label_ids


# confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)
print("\nConfusion Matrix:")
print("True\\Predicted  0(Negative)  1(Neutral)  2(Positive)")
for i, row in enumerate(cm):
    label_name = ["Negative", "Neutral ", "Positive"][i]
    print(f"{label_name}      {row[0]:8d}  {row[1]:8d}  {row[2]:9d}")



# per-class metrics
precision, recall, f1, support = precision_recall_fscore_support(true_labels, predicted_labels, average=None)
print("\nPer-Class:")
class_names = ["Negative", "Neutral", "Positive"]
for i, class_name in enumerate(class_names):
    print(f"{class_name}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}, Support={support[i]}")

print("\nDone.")
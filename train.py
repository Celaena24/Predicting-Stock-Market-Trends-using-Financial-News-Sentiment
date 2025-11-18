import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer


# Load training data
train_path = "data/train_processed.csv"
train_df = pd.read_csv(train_path)
train_df = train_df.rename(columns={"processed_text": "text", "sentiment_encoded": "label"})


# Remove rows with missing values
train_df = train_df.dropna(subset=["text", "label"])


# Create dataset
train_dataset = Dataset.from_pandas(train_df)


# Initialize model and tokenizer
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )


# Tokenize dataset
tokenized_dataset = train_dataset.map(tokenize, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["text"])
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
tokenized_dataset.set_format("torch")


# Initialize model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3
)


# Training arguments
training_args = TrainingArguments(
    output_dir="./finbert_results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    logging_steps=100,
    save_strategy="epoch",
)


# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)


# Train model
trainer.train()


# Save model and tokenizer
trainer.save_model("./finbert_finetuned")
tokenizer.save_pretrained("./finbert_finetuned")


print("Training completed")

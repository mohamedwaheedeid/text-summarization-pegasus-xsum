#!/usr/bin/env python
# coding: utf-8

# Text summerization project with pegasus and xsum dataset

# In[ ]:


# import necessary libraries
import os
import re
import nltk
import spacy
import torch
import evaluate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from bs4 import BeautifulSoup
from datasets import load_dataset
from nltk.corpus import stopwords
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)

# Download NLTK data for preprocessing
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Load Spacy for lemmatization
try:
    nlp = spacy.load("en_core_web_sm")
except:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

stop_words = set(stopwords.words('english'))
rouge = evaluate.load("rouge")


local_path = os.environ.get("XSUM_LOCAL", "")
data_dir = Path(local_path) if local_path else None

try:
    if data_dir and data_dir.is_dir():
        paths = {"train": "train.json", "validation": "validation.json", "test": "test.json"}
        xsum = load_dataset("json", data_files={k: str(data_dir/v) for k,v in paths.items()})
    else:
        print("Fetching XSum from Hugging Face Hub...")
        # We select a small subset (1000) for testing to avoid long wait times
        xsum = load_dataset("xsum", split={'train': 'train[:1000]', 'validation': 'validation[:100]', 'test': 'test[:100]'})
    print("Dataset Loaded Successfully")
except Exception as e:
    print(f"Failed to load dataset: {e}")
    raise


def clean_text(text):
    soup = BeautifulSoup(text, 'html.parser')
    cleaned = soup.get_text(separator=' ')
    cleaned = re.sub(r"\s+", ' ', cleaned).strip()
    return cleaned

# Apply cleaning and create a new column
print("Cleaning dataset...")
xsum = xsum.map(lambda x: {'document_clean': clean_text(x['document'])})

# Visualization: Insight into document lengths
lengths = [len(d.split()) for d in xsum['train']['document_clean']]
plt.figure(figsize=(10, 4))
sns.histplot(lengths, bins=30, color='skyblue')
plt.axvline(512, color='red', linestyle='--', label='Pegasus Limit (512)')
plt.title("Document Word Count Distribution")
plt.legend()
plt.show()


model_name = "google/pegasus-xsum"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading Model on {device}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

def tokenize_function(examples):
    model_inputs = tokenizer(examples['document_clean'], max_length=512, truncation=True)
    labels = tokenizer(text_target=examples['summary'], max_length=128, truncation=True)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenized_datasets = xsum.map(tokenize_function, batched=True)




def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    
    return {k: round(v, 4) for k, v in result.items()}

training_args = Seq2SeqTrainingArguments(
    output_dir="./pegasus-fine-tuned",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True,
    fp16=True if torch.cuda.is_available() else False,
    logging_steps=10,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    compute_metrics=compute_metrics,
)


print("✅ Trainer Initialized. Ready to Train!")
trainer.train()
print("✅ Training Completed!")


# save the fine-tuned model
trainer.save_model("./pegasus-fine-tuned")
print("✅ Model Saved to ./pegasus-fine-tuned")


print("📊 Evaluating model on the test set...")
test_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])

# Display results in a readable format
df_results = pd.DataFrame([test_results])
print(df_results[['eval_rouge1', 'eval_rouge2', 'eval_rougeL', 'eval_loss']])

# Visualization: Compare metrics
plt.figure(figsize=(8, 4))
metrics = ['rouge1', 'rouge2', 'rougeL']
values = [test_results[f'eval_{m}'] for m in metrics]
sns.barplot(x=metrics, y=values, palette='viridis')
plt.title("Final ROUGE Scores on Test Set")
plt.ylim(0, 1) 
plt.show()


def generate_summary(text, model, tokenizer):
    # Prepare the text
    inputs = tokenizer(text, max_length=512, truncation=True, padding="max_length", return_tensors="pt").to(device)
    
    # Generate summary IDs
    summary_ids = model.generate(
        inputs["input_ids"], 
        num_beams=4, 
        max_length=60, 
        early_stopping=True
    )
    
    # Decode back to text
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Pick a random example from the test set
sample_idx = 10 
raw_document = xsum['test'][sample_idx]['document']
actual_summary = xsum['test'][sample_idx]['summary']

# Generate our model's summary
generated_summary = generate_summary(raw_document, model, tokenizer)

print(f"{'='*30} TEST EXAMPLE {'='*30}")
print(f"Original Document (First 300 chars): {raw_document[:300]}...")
print(f"\n✅ Generated Summary: {generated_summary}")
print(f"Actual Summary: {actual_summary}")
print(f"{'='*75}")


save_path = "./final_pegasus_model"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"Model saved to {save_path}. You can now use it for your graduation project or portfolio!")


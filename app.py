from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os

# Absolute path to the trained model
MODEL_PATH =  r"C:\Users\hp\Desktop\txtsummerizationproject"

print("Loading model...")

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)

print("Model loaded successfully")

app = FastAPI(title="PEGASUS Text Summarization API")

class TextRequest(BaseModel):
    text: str


def generate_summary(text):

    inputs = tokenizer(
        text,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    ).to(device)

    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=4,
        max_length=60,
        early_stopping=True
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary


@app.get("/")
def home():
    return {"message": "PEGASUS Summarization API is running"}


@app.post("/summarize")
def summarize(request: TextRequest):

    summary = generate_summary(request.text)

    return {"summary": summary}
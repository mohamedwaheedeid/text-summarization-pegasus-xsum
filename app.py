from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os

# Absolute path to the trained model
MODEL_PATH = r"C:\Users\hp\Desktop\txtsummerizationproject"

print("Loading model...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Model & Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)
print(f"Model loaded successfully on {device}")

app = FastAPI(title="PEGASUS Text Summarization API")

# --- ADDED: CORS Middleware ---
# This allows your Streamlit app to communicate with the API without security blocks
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    return {"status": "online", "model": "Pegasus-XSum"}

@app.post("/summarize")
def summarize(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        summary = generate_summary(request.text)
        return {"summary": summary}
    except Exception as e:
        print(f"Error during summarization: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error during model inference")

if __name__ == "__main__":
    import uvicorn
    # Using 0.0.0.0 makes it accessible within Docker and your local network
    uvicorn.run(app, host="0.0.0.0", port=8000)
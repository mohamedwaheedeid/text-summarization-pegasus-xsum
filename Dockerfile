# Use a Python image that includes building tools
FROM python:3.10-slim

# Install system-level dependencies for ML libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install requirements
COPY requirements.txt .

# 1. Fix the dependency conflict explicitly 
# 2. Install the rest of the requirements
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "fsspec>=2023.1.0,<=2025.10.0" "datasets>=4.5.0" && \
    pip install --no-cache-dir -r requirements.txt

# Download SpaCy model
RUN python -m spacy download en_core_web_sm

# Copy your code (The .dockerignore will handle skipping the 2GB file)
COPY . .

EXPOSE 8000

# Start the API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
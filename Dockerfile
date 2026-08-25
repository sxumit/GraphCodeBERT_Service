# GraphCodeBERT embedding service — container for cloud deployment
FROM python:3.11-slim

WORKDIR /app

# Install system deps needed by torch/transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Pre-download the model into the image so cold-start on the cloud is fast
# (comment out if you'd rather download on first request instead)
RUN python -c "from transformers import AutoTokenizer, AutoModel; \
    AutoTokenizer.from_pretrained('microsoft/graphcodebert-base'); \
    AutoModel.from_pretrained('microsoft/graphcodebert-base')"

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

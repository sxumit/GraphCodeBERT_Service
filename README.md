# ELARA — GraphCodeBERT Embedding Service

This service is a component of ELARA project, an AI-powered bug-assignment system. It wraps Microsoft's pretrained graphcodebert-base model in a lightweight FastAPI REST API, exposing endpoints (/embed, /embed_batch) that convert source code snippets into 768-dimensional semantic embeddings. These embeddings capture both the textual and structural meaning of code (not just surface-level text similarity), and are used by ELARA to index a GitHub repository's codebase and match incoming bug reports to the developers most likely to understand the relevant code. The model is used as-is, without fine-tuning, and the service is containerized with Docker for cloud deployment.

## Run locally

```bash
cd graphcodebert_service
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

Test it:
```bash
curl -X POST http://localhost:8000/embed \
     -H "Content-Type: application/json" \
     -d '{"code": "def add(a, b):\n    return a + b"}'
```

We get back a JSON response with `"dimension": 768` and a list of 768 numbers.
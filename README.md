# ELARA — GraphCodeBERT Embedding Service

Turns code snippets into 768-dimensional embeddings using `microsoft/graphcodebert-base`,
exposed as a REST API for the rest of the ELARA pipeline to call.

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
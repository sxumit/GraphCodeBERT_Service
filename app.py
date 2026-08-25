"""
GraphCodeBERT Embedding Service for ELARA
-------------------------------------------
Loads microsoft/graphcodebert-base locally and exposes a REST API
that converts code (or text) into a 768-dimensional embedding vector.

Run locally:
    uvicorn app:app --host 0.0.0.0 --port 8000

Test:
    curl -X POST http://localhost:8000/embed \
         -H "Content-Type: application/json" \
         -d '{"code": "def add(a, b):\n    return a + b"}'
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch

MODEL_NAME = "microsoft/graphcodebert-base"

app = FastAPI(title="ELARA GraphCodeBERT Embedding Service")

# Load model + tokenizer once at startup (not per-request — that would be slow)
print(f"Loading {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()  # inference mode, no training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model loaded on {device}.")


class EmbedRequest(BaseModel):
    code: str


class EmbedResponse(BaseModel):
    embedding: list[float]
    dimension: int


class BatchEmbedRequest(BaseModel):
    codes: list[str]


class BatchEmbedResponse(BaseModel):
    embeddings: list[list[float]]
    dimension: int


def get_embedding(text: str) -> list[float]:
    """Tokenize code/text, run through GraphCodeBERT, mean-pool token vectors
    into a single 768-dim embedding for the whole snippet."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,   # GraphCodeBERT's max input length
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Mean-pooling over token embeddings (last_hidden_state) is a standard,
    # simple way to get one fixed-size vector per input.
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embedding.cpu().tolist()


@app.get("/")
def health_check():
    return {"status": "ok", "model": MODEL_NAME, "device": str(device)}


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    if not req.code or not req.code.strip():
        raise HTTPException(status_code=400, detail="`code` field cannot be empty")
    vec = get_embedding(req.code)
    return EmbedResponse(embedding=vec, dimension=len(vec))


@app.post("/embed_batch", response_model=BatchEmbedResponse)
def embed_batch(req: BatchEmbedRequest):
    """Batch endpoint — useful when indexing a whole repo (hundreds/thousands
    of files) so ELARA's backend doesn't have to make one HTTP call per file."""
    if not req.codes:
        raise HTTPException(status_code=400, detail="`codes` list cannot be empty")
    vectors = [get_embedding(c) for c in req.codes]
    return BatchEmbedResponse(embeddings=vectors, dimension=len(vectors[0]) if vectors else 0)

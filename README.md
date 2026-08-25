# ELARA — GraphCodeBERT Embedding Service

Turns code snippets into 768-dimensional embeddings using `microsoft/graphcodebert-base`,
exposed as a REST API for the rest of the ELARA pipeline to call.

## 1. Run locally first (to make sure it works before deploying)

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

You should get back a JSON response with `"dimension": 768` and a list of 768 numbers.

## 2. Deploy to the cloud

You have a few options depending on budget/simplicity. Pick ONE:

### Option A — Hugging Face Spaces (easiest, has a free tier)
1. Create a free account at huggingface.co
2. Create a new "Space" → choose "Docker" as the SDK
3. Push this folder's contents (app.py, requirements.txt, Dockerfile) to the Space's git repo
4. HF Spaces builds and hosts it automatically — you get a public URL like:
   `https://<your-username>-<space-name>.hf.space`
5. Your API endpoint becomes: `https://<your-username>-<space-name>.hf.space/embed`

Best option if you want zero AWS/cloud billing complexity for a student project.

### Option B — Render.com (simple, has a free/cheap tier)
1. Push this folder to a GitHub repo
2. On Render.com: New → Web Service → connect your repo
3. Render auto-detects the Dockerfile and builds it
4. You get a URL like `https://elara-graphcodebert.onrender.com`

Note: Render's free tier has limited RAM — GraphCodeBERT (~500MB) should fit, but if it's slow/crashing, upgrade to their cheapest paid tier (~$7/month).

### Option C — AWS (matches your original architecture diagram)
1. Build and push the Docker image to Amazon ECR:
   ```bash
   docker build -t elara-graphcodebert .
   aws ecr create-repository --repository-name elara-graphcodebert
   docker tag elara-graphcodebert:latest <your-ecr-url>/elara-graphcodebert:latest
   docker push <your-ecr-url>/elara-graphcodebert:latest
   ```
2. Deploy via AWS App Runner (simplest AWS option — auto-scales, gives you a public URL directly) or ECS Fargate if you want more control.
3. App Runner will give you a URL like `https://xxxxx.us-east-1.awsapprunner.com`

More setup work, but matches the "Deployment: Docker + AWS" line in your tech stack doc — worth doing if you want the demo to visually match your architecture diagram.

## 3. Give your teammate the final endpoint

Once deployed, the two endpoints your teammate needs are:

- **Single embedding:** `POST <your-url>/embed` — body: `{"code": "..."}`
- **Batch embedding** (for indexing a whole repo at once, faster than one-by-one):
  `POST <your-url>/embed_batch` — body: `{"codes": ["...", "...", "..."]}`

Both return a `768`-dimensional vector (or list of vectors for batch), ready to be stored in Qdrant.

## Notes

- **No fine-tuning needed** (as specified) — this uses the pretrained model as-is via `AutoModel.from_pretrained(...)`, no training loop included.
- **Mean-pooling** is used to turn GraphCodeBERT's per-token outputs into one fixed 768-dim vector per code snippet — this is the standard, simplest approach. (If accuracy needs improving later, CLS-token pooling or a learned pooling layer are alternatives worth experimenting with.)
- **Max input length is 512 tokens** — for very large files, you'll want to chunk the file into smaller pieces before embedding (this matches the "subgraph, depth=2" chunking idea from the pseudocode your mentor shared earlier).
- Add authentication (an API key header) before giving teammates/production access, especially if deploying somewhere with a public URL and no built-in access control.

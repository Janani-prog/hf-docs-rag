# Stage 1: Builder — installs dependencies and pre-downloads models
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Pre-download embedding and reranker models during build
# so the container starts instantly without downloading on first query
COPY src/ ./src/
RUN python -c "
from sentence_transformers import SentenceTransformer, CrossEncoder
SentenceTransformer('all-MiniLM-L6-v2')
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print('Models cached.')
"

# Stage 2: Runtime — lean final image
FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /install /usr/local
COPY --from=builder /root/.cache /root/.cache
COPY . .

EXPOSE 7860

ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

CMD ["python", "app.py"]
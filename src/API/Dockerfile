# Base image: Debian slim with Python 3.12 (pydub needs audioop, removed in 3.13).
FROM python:3.12-slim

# Don't write .pyc files; don't buffer stdout so logs appear immediately.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# pydub needs the real ffmpeg binary (the pip "ffmpeg" package is not it).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch: the default wheel bundles CUDA and is several GB larger.
RUN pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu

# Copy requirements first so this slow layer stays cached until requirements change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the embedding model at build time so containers don't fetch it on every start.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')"

# Source code changes often, so it goes last.
COPY . .

EXPOSE 8000

# 0.0.0.0 = accept connections from outside the container.
CMD ["uvicorn", "claudeMain:app", "--host", "0.0.0.0", "--port", "8000"]

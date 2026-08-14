# ── Base image ───────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        netcat-openbsd \
        git \
        gcc \
        g++ \
        python3-dev \
        cython3 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .
# numpy 2.x + cython antes do restante (madmom-modern exige NumPy 2.x; modelos .pkl
# referenciam numpy._core). PyTorch CPU-only evita dependências CUDA em uma imagem local.
# madmom-modern compilado por último, contra o NumPy runtime.
RUN pip install --no-cache-dir "numpy>=2.2,<3" cython \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchaudio \
    && grep -vE '^madmom(-modern)? @' requirements.txt > /tmp/requirements-docker.txt \
    && pip install --no-cache-dir -r /tmp/requirements-docker.txt \
    && pip install --no-cache-dir --no-build-isolation \
        "madmom-modern @ git+https://github.com/imcmurray/madmom-modern.git"

# ── Application source ────────────────────────────────────────────────────────
COPY . .

# Make the entrypoint script executable
RUN chmod +x entrypoint.sh

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Entrypoint & default command ──────────────────────────────────────────────
ENTRYPOINT ["./entrypoint.sh"]
CMD ["gunicorn", "config.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "1", "--timeout", "120"]

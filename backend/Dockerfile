FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    CHECKPOINT_DIR=/app/backend/data/checkpoints \
    MAX_UPLOAD_SIZE_MB=50

WORKDIR /app/backend

COPY render-requirements.txt ./render-requirements.txt

# Install CPU-only PyTorch so the image does not pull CUDA libraries.
RUN python -m pip install --upgrade pip && \
    python -m pip install --index-url https://download.pytorch.org/whl/cpu \
      --extra-index-url https://pypi.org/simple \
      torch==2.14.1+cpu torchvision==0.29.1+cpu && \
    python -m pip install -r render-requirements.txt

COPY app ./app
COPY data/checkpoints/resnet18_647c05a4_best.pth ./data/checkpoints/resnet18_647c05a4_best.pth

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]

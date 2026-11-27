FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/hf_cache \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY inscription_vlm_lora/ ./inscription_vlm_lora/
COPY inscriptions_dataset.csv .
COPY image_embeddings_full.npz .
COPY prepare_models.py .

# Pre-download SigLIP + CLIP and pre-merge Qwen2.5-VL-3B + LoRA into ./merged_vlm
RUN TRANSFORMERS_OFFLINE=0 HF_HUB_OFFLINE=0 python prepare_models.py && rm -f prepare_models.py

COPY predict_inscription.py .
COPY main.py .

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
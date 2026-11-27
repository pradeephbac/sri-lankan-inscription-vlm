FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download model files directly to disk cache (<200MB RAM used, no OOM during build)
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('Qwen/Qwen2.5-VL-3B-Instruct'); \
    snapshot_download('google/siglip-base-patch16-224'); \
    snapshot_download('openai/clip-vit-base-patch32')"

COPY inscription_vlm_lora/ ./inscription_vlm_lora/
COPY image_embeddings_full.npz inscriptions_dataset.csv predict_inscription.py main.py ./

EXPOSE 8080
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
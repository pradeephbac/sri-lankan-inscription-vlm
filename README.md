# 🏛️ Sri Lankan Inscription Recognition & Archival Search

[![Live App](https://img.shields.io/badge/Live_Demo-slinscriptionsmlbe.web.app-FF6B00?style=for-the-badge&logo=firebase&logoColor=white)](https://slinscriptionsmlbe.web.app/)
[![Model](https://img.shields.io/badge/VLM-Qwen2.5--VL--3B--LoRA-1E3A8A?style=for-the-badge)](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
[![Dataset](https://img.shields.io/badge/Corpus-db.inscriptions.lk-0F766E?style=for-the-badge)](https://db.inscriptions.lk)

> **Try the Live Application:** 👉 <a href="https://slinscriptionsmlbe.web.app/" target="_blank" rel="noopener noreferrer"><strong>https://slinscriptionsmlbe.web.app/</strong></a>
> Upload a rubbing or field photograph of an ancient Sri Lankan Brahmi or Sinhala stone inscription to get an instant AI transliteration, English translation, paleographic classification, and visually matched archival plates.
---

## ✨ Application & Training Showcase

| Desktop Interface (`web-view.png`) | Mobile Field View (`mobile-view.png`) |
| :---: | :---: |
| <img src="docs/screenshots/web-view.png" alt="Desktop Web Interface" width="100%"/> | <img src="docs/screenshots/mobile-view.png" alt="Mobile Field View" width="260"/> |

---

## 🔍 Overview & Architecture
Scattered across Sri Lanka’s cave drip-ledges, rock faces, and stone pillars are thousands of inscriptions dating back to the 3rd century BCE. Sourced from **[db.inscriptions.lk](https://db.inscriptions.lk)**, this project takes a specialized, limited-size historical dataset through the **complete AI engineering lifecycle**—from data curation and Vision-Language Model fine-tuning to a fast, low-cost serverless web app anyone can use in the field.

- **Domain Image Preprocessing:** Automatic contrast stretching (`autocontrast`) and edge sharpening tailored to weathered granite carvings and low-contrast estampage rubbings.
- **Fine-Tuned Vision-Language Model:** `Qwen/Qwen2.5-VL-3B-Instruct` merged with a custom LoRA adapter (`inscription_vlm_lora`) to generate structured JSON (transliteration, English translation, paleographic footnotes, script, period, and likely site).
- **Archival Visual Retrieval:** `google/siglip-base-patch16-224` cosine similarity search across digitized plates from `db.inscriptions.lk`, surfacing the top 3 verified historical matches alongside AI predictions.
- **Zero-Shot Input Guardrails:** `openai/clip-vit-base-patch32` classification + archival similarity thresholding (`>= 0.28`) to reject non-inscription photos before VLM inference.
- **Serverless Production Deployment:** Mobile-first dark-mode frontend on **Firebase Hosting** (`public/index.html`) backed by a containerized **FastAPI** service on **Google Cloud Run** (`backend/`) configured with scale-to-zero (`--min-instances 0`), per-IP rate limiting, and domain-locked CORS.

---

## ⚙️ End-to-End Execution Order (Data Collection to Deployment)

### Step 1: Data Collection (`pipeline/crawl_inscriptions.py`)
Crawls digitized inscription plates and epigraphic metadata from `db.inscriptions.lk`, saving raw images into `inscription_images/` and tabular records into `data/inscriptions_dataset.csv`.
```bash
python pipeline/crawl_inscriptions.py
```

### Step 2: Dataset Preparation & Augmentation (`pipeline/prepare_dataset.py`)
Applies contrast/rotation augmentations for low-resource generalization and splits the records (`80/10/10`) into structured prompt-response pairs (`data/train.jsonl`, `data/val.jsonl`, `data/test.jsonl`).
```bash
python pipeline/prepare_dataset.py
```

### Step 3: Vision-Language Model Fine-Tuning (`pipeline/train_vlm.py`)
Fine-tunes `Qwen/Qwen2.5-VL-3B-Instruct` using LoRA (executed on Google Colab GPU) and saves the lightweight adapter weights and tokenizer configs into `inscription_vlm_lora/`.
```bash
python pipeline/train_vlm.py
```

### Step 4: Held-Out Test Evaluation (`pipeline/evaluate_test.py`)
Evaluates the merged LoRA model against `data/test.jsonl` to benchmark JSON validity, transliteration accuracy, and period/script classification.
```bash
python pipeline/evaluate_test.py
```

### Step 5: Local Backend & Inference Testing (`backend/main.py` & `backend/predict_inscription.py`)
Starts the local FastAPI inference server (which loads the merged Qwen2.5-VL model, builds/loads SigLIP archival embeddings, and enables CLIP guardrails).
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

### Step 6: Production Backend Deployment on Google Cloud Run (`backend/Dockerfile` & `backend/cloudbuild.yaml`)
Builds the container image with pre-cached Hugging Face model weights and deploys to Cloud Run with scale-to-zero (`--min-instances 0`) and CPU throttling for minimal cost.
```bash
gcloud builds submit --config backend/cloudbuild.yaml backend/

gcloud run deploy inscriptions-api \
  --image gcr.io/<YOUR_PROJECT_ID>/inscriptions-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 16Gi \
  --cpu 4 \
  --timeout 300 \
  --min-instances 0 \
  --max-instances 2 \
  --cpu-throttling
```

### Step 7: Frontend Deployment on Firebase Hosting (`public/index.html` & `firebase.json`)
Deploys the mobile-responsive single-page web interface to Firebase Hosting (`https://slinscriptionsmlbe.web.app/`).
```bash
firebase deploy --only hosting
```

---

## 📁 Repository Structure
```text
├── public/                 # Mobile-responsive dark-mode web UI (Firebase Hosting)
├── backend/                # FastAPI inference service, Dockerfile & Cloud Run config
├── pipeline/               # Data crawling (db.inscriptions.lk), 80/10/10 prep, LoRA training & eval
├── data/                   # Lightweight tabular metadata and train/val/test JSONL splits
├── inscription_vlm_lora/   # Fine-tuned LoRA adapter weights and tokenizer configs
├── docs/screenshots/       # Application UI and training screenshots
└── firebase.json           # Firebase Hosting deployment configuration
```

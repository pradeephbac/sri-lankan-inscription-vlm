import io
import time
import tempfile
import threading
import traceback
from collections import defaultdict
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, ImageEnhance
from predict_inscription import InscriptionAnalyzer, UnreadableInscriptionError

MAX_FILE_BYTES = 5 * 1024 * 1024  # 5 MB limit
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
CLIENT_APP_TOKEN = "sl-inscriptions-web"

RATE_LIMIT_MAX = 10        # max analyses per IP
RATE_LIMIT_WINDOW = 3600   # per 1 hour (seconds)
ip_requests: dict[str, list[float]] = defaultdict(list)

analyzer: InscriptionAnalyzer | None = None
load_error: str | None = None


def check_rate_limit(client_ip: str) -> None:
    now = time.time()
    recent = [t for t in ip_requests[client_ip] if now - t < RATE_LIMIT_WINDOW]
    if len(recent) >= RATE_LIMIT_MAX:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded (max 10 analyses per hour). Please try again later.",
        )
    recent.append(now)
    ip_requests[client_ip] = recent


def _load_models() -> None:
    global analyzer, load_error
    try:
        print("[Startup] Loading InscriptionAnalyzer models in background thread...")
        analyzer = InscriptionAnalyzer(train_only_index=False)
        print("[Startup] InscriptionAnalyzer ready!")
    except Exception:
        load_error = traceback.format_exc()
        print(f"[Startup ERROR] Failed to load models:\n{load_error}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_load_models, daemon=True).start()
    yield


app = FastAPI(title="Sri Lankan Epigraphy AI API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://slinscriptionsml.web.app",
        "https://slinscriptionsml.firebaseapp.com",
        "https://slinscriptionsmlbe.web.app",
        "https://slinscriptionsmlbe.firebaseapp.com",
        "http://localhost:5000",
        "http://127.0.0.1:5000",
    ],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Content-Type", "X-Client-App"],
)


def enhance_stone_contrast(pil_img: Image.Image) -> Image.Image:
    img = pil_img.convert("RGB")
    img = ImageOps.autocontrast(img, cutoff=2)
    img = ImageEnhance.Contrast(img).enhance(1.35)
    img = ImageEnhance.Sharpness(img).enhance(1.5)
    return img


@app.get("/api/health")
def health():
    return {
        "status": "error" if load_error else ("ok" if analyzer is not None else "warming_up"),
        "model_loaded": analyzer is not None,
        "error": load_error,
    }


@app.post("/api/analyze")
async def analyze(
    request: Request,
    file: UploadFile = File(...),
    enhance: bool = Form(True),
    x_client_app: str | None = Header(default=None),
):
    if x_client_app != CLIENT_APP_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized client.")

    forwarded_for = request.headers.get("x-forwarded-for")
    client_ip = (
        forwarded_for.split(",")[0].strip()
        if forwarded_for
        else (request.client.host if request.client else "unknown")
    )
    check_rate_limit(client_ip)

    if load_error:
        raise HTTPException(status_code=500, detail=f"Model failed to initialize: {load_error}")
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Model is still warming up. Please retry in 15 seconds.")

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Only JPEG, PNG, or WebP images are supported.")

    raw_bytes = await file.read()
    if len(raw_bytes) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 5 MB limit.")

    try:
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or corrupted image file.")

    processed_img = enhance_stone_contrast(img) if enhance else img

    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
            processed_img.save(tmp.name, format="JPEG", quality=95)
            result = analyzer.predict(tmp.name, stream=False)
    except UnreadableInscriptionError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return result
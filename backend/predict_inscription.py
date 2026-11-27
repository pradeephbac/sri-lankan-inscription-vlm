import os
import re
import json
import time
import platform
import torch
import numpy as np
import pandas as pd
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    AutoImageProcessor,
    AutoModel,
    CLIPProcessor,
    CLIPModel,
    TextStreamer,
)
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
MERGED_MODEL_DIR = "./merged_vlm"
LORA_DIR = "./inscription_vlm_lora"
EMBED_MODEL = "google/siglip-base-patch16-224"
GUARD_MODEL = "openai/clip-vit-base-patch32"
CSV_PATH = "inscriptions_dataset.csv"

MIN_ARCHIVE_SIMILARITY = 0.28

GUARDRAIL_LABELS = [
    # Valid historic inscription categories (indices 0, 1, 2)
    "an ancient stone inscription or rock carving with historic letters",
    "an archaeological black and white rubbing or estampage of ancient script",
    "weathered ancient stone pillar or cave wall with engraved symbols",
    # Invalid / meme / modern text categories (indices 3, 4, 5, 6)
    "a funny internet meme with modern English text",
    "a modern digital screenshot, social media post, or typed English words",
    "a modern photograph of a person, selfie, animal, food, or everyday object",
    "a blank screen, solid color, or vulgar inappropriate image",
]


class UnreadableInscriptionError(ValueError):
    """Raised when an uploaded image is a meme, modern English text, or non-inscription photo."""


class InscriptionAnalyzer:
    def __init__(self, train_only_index: bool = False):
        t0 = time.time()
        self.vlm_device = "cpu"
        self.embed_device = "cpu"

        num_threads = 2 if platform.system() == "Darwin" else (os.cpu_count() or 8)
        torch.set_num_threads(num_threads)
        print(f"Initializing CPU-Optimized InscriptionAnalyzer (threads: {num_threads})...")

        df = pd.read_csv(CSV_PATH).fillna("")
        if train_only_index and os.path.exists("train.jsonl"):
            with open("train.jsonl", "r", encoding="utf-8") as f:
                train_ids = {str(json.loads(line)["inscriptionNo"]) for line in f}
            df = df[df["inscriptionNo"].astype(str).isin(train_ids)].reset_index(drop=True)
        self.df = df

        # 1. Load SigLIP for archival similarity search
        print("[Startup 1/4] Loading SigLIP archive encoder...")
        self.embed_processor = AutoImageProcessor.from_pretrained(EMBED_MODEL, local_files_only=True)
        self.embed_model = (
            AutoModel.from_pretrained(EMBED_MODEL, local_files_only=True)
            .to(self.embed_device)
            .eval()
        )
        cache_file = "image_embeddings_train.npz" if train_only_index else "image_embeddings_full.npz"
        self.db_embeddings, self.db_indices = self._load_or_build_index(cache_file)

        # 2. Load lightweight CLIP zero-shot guardrail
        print("[Startup 2/4] Loading CLIP visual guardrail...")
        self.guard_processor = CLIPProcessor.from_pretrained(GUARD_MODEL, local_files_only=True)
        self.guard_model = (
            CLIPModel.from_pretrained(GUARD_MODEL, local_files_only=True)
            .to(self.embed_device)
            .eval()
        )

        # 3. Load Pre-Merged VLM + Apply INT8 Dynamic Quantization for fast x86 CPU inference
        print("[Startup 3/4] Loading & quantizing Qwen2.5-VL-3B for CPU...")
        if os.path.exists(MERGED_MODEL_DIR):
            self.vlm_processor = AutoProcessor.from_pretrained(
                MERGED_MODEL_DIR,
                min_pixels=128 * 28 * 28,
                max_pixels=256 * 28 * 28,
                local_files_only=True,
            )
            vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                MERGED_MODEL_DIR,
                torch_dtype=torch.float32,
                attn_implementation="sdpa",
                low_cpu_mem_usage=True,
                local_files_only=True,
            ).eval()
        else:
            self.vlm_processor = AutoProcessor.from_pretrained(
                BASE_MODEL,
                min_pixels=128 * 28 * 28,
                max_pixels=256 * 28 * 28,
                local_files_only=True,
            )
            base_vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.float32,
                attn_implementation="sdpa",
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
            vlm_model = PeftModel.from_pretrained(base_vlm, LORA_DIR).merge_and_unload().eval()

        # Quantize Linear layers to INT8 on Linux x86_64 (Cloud Run) for 2.5x-4x faster CPU decoding
        if platform.system() == "Linux":
            print("  Applying INT8 dynamic quantization to Linear layers...")
            vlm_model = torch.ao.quantization.quantize_dynamic(
                vlm_model, {torch.nn.Linear}, dtype=torch.qint8
            )

        self.vlm = vlm_model.to(self.vlm_device).eval()

        # 4. Warmup pass so PyTorch CPU kernels & KV-cache buffers are pre-allocated
        print("[Startup 4/4] Running CPU warmup pass...")
        self._warmup()
        print(f"[Startup Complete] Ready in {time.time() - t0:.2f}s!")

    @torch.inference_mode()
    def _warmup(self) -> None:
        dummy = Image.new("RGB", (224, 224), color=(128, 128, 128))
        _ = self._embed_image(dummy)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Read."},
                ],
            }
        ]
        text_input = self.vlm_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.vlm_processor(
            text=[text_input], images=[dummy], return_tensors="pt"
        ).to(self.vlm_device)
        _ = self.vlm.generate(**inputs, max_new_tokens=4, do_sample=False, use_cache=True)

    @torch.inference_mode()
    def _check_is_historic_inscription(self, img: Image.Image, max_archive_score: float) -> None:
        inputs = self.guard_processor(
            text=GUARDRAIL_LABELS,
            images=img.convert("RGB"),
            return_tensors="pt",
            padding=True,
        ).to(self.embed_device)

        outputs = self.guard_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=-1).cpu().numpy()[0]

        valid_prob = float(probs[:3].sum())
        invalid_prob = float(probs[3:].sum())
        top_label = GUARDRAIL_LABELS[int(np.argmax(probs))]

        print(
            f"[Guardrail] ArchiveSim={max_archive_score:.3f} | "
            f"HistoricProb={valid_prob:.2f} vs NonHistoricProb={invalid_prob:.2f} | "
            f"Top='{top_label}'"
        )

        if invalid_prob > 0.65 or max_archive_score < MIN_ARCHIVE_SIMILARITY:
            raise UnreadableInscriptionError(
                "Cannot read this image. Please upload a photograph or rubbing of a historic stone inscription."
            )

    @torch.inference_mode()
    def _embed_image(self, img: Image.Image) -> np.ndarray:
        inputs = self.embed_processor(images=img.convert("RGB"), return_tensors="pt").to(self.embed_device)
        feats = self.embed_model.get_image_features(**inputs)
        if not isinstance(feats, torch.Tensor):
            feats = feats.pooler_output
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()[0]

    def _load_or_build_index(self, cache_file: str):
        if os.path.exists(cache_file):
            data = np.load(cache_file)
            return data["embeddings"], data["indices"]

        print(f"Building visual similarity index ({cache_file})...")
        embeddings, valid_indices = [], []
        total = len(self.df)
        for idx, row in self.df.iterrows():
            path = str(row["locallink"]).strip()
            if path and os.path.exists(path):
                emb = self._embed_image(Image.open(path))
                embeddings.append(emb)
                valid_indices.append(idx)
            if (idx + 1) % 200 == 0 or (idx + 1) == total:
                print(f"  Embedded {idx + 1}/{total} images...")

        emb_arr = np.vstack(embeddings)
        idx_arr = np.array(valid_indices)
        np.savez(cache_file, embeddings=emb_arr, indices=idx_arr)
        return emb_arr, idx_arr

    def find_similar_inscriptions(self, query_emb: np.ndarray, top_k: int = 3) -> tuple[list[dict], float]:
        scores = self.db_embeddings @ query_emb
        max_score = float(scores.max())
        top_pos = np.argsort(scores)[::-1][:top_k]

        results = []
        for pos in top_pos:
            row = self.df.iloc[self.db_indices[pos]]
            results.append({
                "inscriptionNo": str(row["inscriptionNo"]),
                "similarity_score": round(float(scores[pos]), 4),
                "Site": row["Site"],
                "Period": row["Period"],
                "Language": row["Language"],
                "Sinhala Prakrit": row["Sinhala Prakrit"],
                "Script": row["Script"],
                "Publication": row["Publication"],
                "realInscriptionUrl": row["realInscriptionUrl"],
                "Transliteration": row["Transliteration"],
                "Translation": row["Translation"],
                "Footnotes": row["Footnotes"],
            })
        return results, max_score

    @torch.inference_mode()
    def predict(self, user_image_path: str, stream: bool = False) -> dict:
        t_start = time.time()
        img = Image.open(user_image_path).convert("RGB")
        query_emb = self._embed_image(img)
        similar_records, max_archive_score = self.find_similar_inscriptions(query_emb, top_k=3)

        self._check_is_historic_inscription(img, max_archive_score)

        prompt = (
            "Expert Sri Lankan epigraphist. Return ONLY compact JSON with keys: "
            "transliteration, translation, footnotes, period, language, sinhala_prakrit, "
            "script, likely_site, reference_publication."
        )

        img_resized = img.copy()
        img_resized.thumbnail((336, 336))

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text_input = self.vlm_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.vlm_processor(
            text=[text_input], images=[img_resized], return_tensors="pt"
        ).to(self.vlm_device)

        streamer = TextStreamer(self.vlm_processor.tokenizer, skip_prompt=True) if stream else None
        output_ids = self.vlm.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            use_cache=True,
            streamer=streamer,
        )
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        raw_output = self.vlm_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        match = re.search(r"\{.*\}", raw_output, re.DOTALL)
        try:
            predicted_data = json.loads(match.group(0) if match else raw_output)
        except json.JSONDecodeError:
            predicted_data = {"transliteration": raw_output}

        elapsed = round(time.time() - t_start, 2)
        print(f"[VLM] CPU Prediction finished in {elapsed}s")

        return {
            "prediction": predicted_data,
            "related_inscriptions": similar_records,
            "inference_seconds": elapsed,
        }
import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    AutoImageProcessor,
    AutoModel,
    CLIPProcessor,
    CLIPModel,
)
from peft import PeftModel

print("Downloading SigLIP and CLIP...")
AutoImageProcessor.from_pretrained("google/siglip-base-patch16-224")
AutoModel.from_pretrained("google/siglip-base-patch16-224")
CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

print("Downloading Qwen2.5-VL-3B and merging LoRA adapter...")
proc = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)
merged = PeftModel.from_pretrained(base, "./inscription_vlm_lora").merge_and_unload()
merged.save_pretrained("./merged_vlm", safe_serialization=True)
proc.save_pretrained("./merged_vlm")
print("Successfully pre-merged VLM weights!")
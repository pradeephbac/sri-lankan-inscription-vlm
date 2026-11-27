import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from PIL import Image
from datasets import load_dataset
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
TRAIN_FILE = "train.jsonl"
VAL_FILE = "val.jsonl"
OUTPUT_DIR = "./inscription_vlm_lora"

SYSTEM_PROMPT = (
    "You are an expert Sri Lankan epigraphist. Analyze the uploaded inscription image "
    "and return a valid JSON object containing: transliteration, translation, footnotes, "
    "period, language, sinhala_prakrit, script, likely_site, and reference_publication."
)


def main():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds = load_dataset("json", data_files=TRAIN_FILE, split="train")
    val_ds = load_dataset("json", data_files=VAL_FILE, split="train")

    def collate_fn(batch):
        texts = []
        images = []
        for item in batch:
            img = Image.open(item["image_path"]).convert("RGB")
            img.thumbnail((512, 512))
            images.append(img)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": SYSTEM_PROMPT},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": item["target_json"]}],
                },
            ]
            text = processor.apply_chat_template(messages, tokenize=False)
            texts.append(text)

        inputs = processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        labels = inputs["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        return inputs

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        prediction_loss_only=True,
        eval_accumulation_steps=1,
        fp16=True,
        bf16=False,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
    )

    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
    
# import torch
# from PIL import Image
# from datasets import load_dataset
# from transformers import (
#     Qwen2_5_VLForConditionalGeneration,
#     AutoProcessor,
#     TrainingArguments,
#     Trainer,
# )
# from peft import LoraConfig, get_peft_model

# MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
# TRAIN_FILE = "train.jsonl"
# VAL_FILE = "val.jsonl"
# OUTPUT_DIR = "./inscription_vlm_lora"

# SYSTEM_PROMPT = (
#     "You are an expert Sri Lankan epigraphist. Analyze the uploaded inscription image "
#     "and return a valid JSON object containing: transliteration, translation, footnotes, "
#     "period, language, sinhala_prakrit, script, likely_site, and reference_publication."
# )


# def main():
#     processor = AutoProcessor.from_pretrained(MODEL_ID)
#     model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#         MODEL_ID,
#         torch_dtype=torch.float16,
#         device_map="auto",
#     )

#     lora_config = LoraConfig(
#         r=16,
#         lora_alpha=32,
#         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
#         lora_dropout=0.05,
#         bias="none",
#         task_type="CAUSAL_LM",
#     )
#     model = get_peft_model(model, lora_config)
#     model.print_trainable_parameters()

#     train_ds = load_dataset("json", data_files=TRAIN_FILE, split="train")
#     val_ds = load_dataset("json", data_files=VAL_FILE, split="train")

#     def collate_fn(batch):
#         texts = []
#         images = []
#         for item in batch:
#             img = Image.open(item["image_path"]).convert("RGB")
#             img.thumbnail((768, 768))
#             images.append(img)

#             messages = [
#                 {
#                     "role": "user",
#                     "content": [
#                         {"type": "image"},
#                         {"type": "text", "text": SYSTEM_PROMPT},
#                     ],
#                 },
#                 {
#                     "role": "assistant",
#                     "content": [{"type": "text", "text": item["target_json"]}],
#                 },
#             ]
#             text = processor.apply_chat_template(messages, tokenize=False)
#             texts.append(text)

#         inputs = processor(
#             text=texts,
#             images=images,
#             padding=True,
#             return_tensors="pt",
#         )
#         labels = inputs["input_ids"].clone()
#         labels[labels == processor.tokenizer.pad_token_id] = -100
#         inputs["labels"] = labels
#         return inputs

#     training_args = TrainingArguments(
#         output_dir=OUTPUT_DIR,
#         per_device_train_batch_size=1,
#         gradient_accumulation_steps=8,
#         num_train_epochs=4,
#         learning_rate=2e-4,
#         lr_scheduler_type="cosine",
#         warmup_steps=60,
#         logging_steps=10,
#         eval_strategy="epoch",
#         save_strategy="epoch",
#         fp16=True,
#         bf16=False,
#         remove_unused_columns=False,
#     )

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_ds,
#         eval_dataset=val_ds,
#         data_collator=collate_fn,
#     )

#     trainer.train()
#     model.save_pretrained(OUTPUT_DIR)
#     processor.save_pretrained(OUTPUT_DIR)
#     print(f"Model saved to {OUTPUT_DIR}")


# if __name__ == "__main__":
#     main()
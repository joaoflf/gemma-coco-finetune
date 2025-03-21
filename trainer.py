import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from PIL import Image
from dataset import CocoDataset


model_id = "google/gemma-3-4b-pt"
tokenizer_id = "google/gemma-3-4b-it"

model_kwargs = dict(
    attn_implementation="eager", torch_dtype=torch.bfloat16, device_map="auto"
)

model_kwargs["quantization_config"] = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
    bnb_4bit_quant_storage=model_kwargs["torch_dtype"],
)

model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
processor = AutoProcessor.from_pretrained(tokenizer_id)

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
    modules_to_save=["lm_head", "embed_tokens"],
)


args = SFTConfig(
    output_dir="gemma-pose-estimation",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    logging_steps=5,
    save_strategy="epoch",
    learning_rate=2e-4,
    bf16=False,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    push_to_hub=True,
    hub_model_id="JoaoFLF/gemma3-pose-estimation",
    hub_strategy="every_save",
    report_to="tensorboard",
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataset_text_field="",
    dataset_kwargs={"skip_prepare_dataset": True},
)
args.remove_unused_columns = False


def process_vision_info(messages: list[dict]) -> list[Image.Image]:
    image_inputs = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]

        for element in content:
            if isinstance(element, dict) and (
                "image" in element or element.get("type") == "image"
            ):
                if "image" in element:
                    image = element["image"]
                else:
                    image = element
                image_inputs.append(image.convert("RGB"))
    return image_inputs


def collate_fn(examples):
    texts = []
    images = []

    for example in examples:
        image_inputs = process_vision_info(example["messages"])
        text = processor.apply_chat_template(
            example["messages"], add_generation_prompt=False, tokenize=False
        )
        texts.append(text.strip())
        images.append(image_inputs)

        batch = processor(
            text=texts, images=images, padding="max_length", return_tensors="pt"
        )

        labels = batch["input_ids"].clone()  # Create a copy to avoid modifying original

        # Find the correct image token - adjust this based on your model's tokens
        # Check if 'boi_token' exists in special_tokens_map
        image_token_id = None
        if "boi_token" in processor.tokenizer.special_tokens_map:
            image_token_id = processor.tokenizer.convert_tokens_to_ids(
                processor.tokenizer.special_tokens_map["boi_token"]
            )
        elif "<image>" in processor.tokenizer.special_tokens_map.values():
            # Try finding by value instead
            for key, value in processor.tokenizer.special_tokens_map.items():
                if value == "<image>":
                    image_token_id = processor.tokenizer.convert_tokens_to_ids(value)
                    break

        # Apply masking for image token if found
        if image_token_id is not None:
            labels[labels == image_token_id] = -100

        # Apply masking for pad tokens
        labels[labels == 262144] = -100

        batch["labels"] = labels
        return batch


dataset = [sample for sample in CocoDataset().get_dataset()]
# dataset = CocoDataset().get_dataset()
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=processor,
    data_collator=collate_fn,
)

trainer.train()

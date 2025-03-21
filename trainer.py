import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig


model_id = "google/gemma-3-4b-pt"
tokenizer_id = "google/gemma-3-4b-it"

model_kwargs = dict(
    attn_implementation="eager", torch_dtype=torch.bfloat16, device_map="mps"
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

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
import torch
import os

def load_model(model_type="peft"):
    """
    Loads the specified FLAN-T5 model and its tokenizer.

    Args:
        model_type (str): Type of model to load.
                          Can be "peft" (LoRA fine-tuned),
                          "full" (fully fine-tuned), or
                          "original" (base FLAN-T5 model).

    Returns:
        tuple: A tuple containing the loaded model and its tokenizer.
    """
    base_model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    if model_type == "peft":
        base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16)
        peft_model_path = os.path.join("models", "peft_ft")
        model = PeftModel.from_pretrained(base_model, peft_model_path, torch_dtype=torch.bfloat16)
    elif model_type == "original":
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16)
    elif model_type == "full":
        full_model_path = os.path.join("models", "full_ft")
        model = AutoModelForSeq2SeqLM.from_pretrained(full_model_path, torch_dtype=torch.bfloat16)
    else:
        raise ValueError("Unsupported model_type. Use 'peft', 'full', or 'original'.")

    model.eval()
    return model, tokenizer

def generate_summary(text, model, tokenizer, max_tokens=200):
    """
    Generates a summary for the given text using the provided model and tokenizer.

    Args:
        text (str): The input dialogue text to summarize.
        model: The loaded Hugging Face model.
        tokenizer: The loaded Hugging Face tokenizer.
        max_tokens (int): Maximum number of new tokens to generate for the summary.

    Returns:
        str: The generated summary.
    """
    # Prepare the input text for the model
    input_text = f"Summarize the following conversation.\n\n{text}\n\nSummary:"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
    
    # Generate the summary without tracking gradients
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_tokens)
    return tokenizer.decode(output[0], skip_special_tokens=True)
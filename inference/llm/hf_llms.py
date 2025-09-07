from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.generation.configuration_utils import GenerationConfig
<<<<<<< HEAD
import torch



def has_working_chat_template(tokenizer):
    return hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None


def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    generation_config = GenerationConfig.from_pretrained(model_name)

    
    use_chat_template = has_working_chat_template(tokenizer)
    
    return model, tokenizer, generation_config, use_chat_template


def generate(model_tokenizer_config, model_input):
    model, tokenizer, generation_config, use_chat_template = model_tokenizer_config
    full_prompt = f"{model_input['prompt']}\n{model_input['sample']}"
    
    if use_chat_template:
        messages = [{"role": "user", "content": full_prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)
        input_len = inputs.shape[-1]
        
        with torch.inference_mode():
            generation = model.generate(
                inputs,
                max_new_tokens=4096,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            generation = generation[0][input_len:]
        
        return tokenizer.decode(generation, skip_special_tokens=True).strip()
    
    inputs = tokenizer(full_prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    generate_ids = model.generate(
        **inputs,
        max_new_tokens=4096,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(
        generate_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
    ).strip()
=======


def load_model(model_name):
    """Load the specified Hugging Face text LLM model.
    
    Args:
        model_name: The model identifier (e.g., 'CohereLabs/aya-expanse-32b', 'google/gemma-3-12b-it', 'Unbabel/Tower-Plus-9B', etc.)
    
    Returns:
        Tuple of (model, tokenizer, generation_config)
    
    Raises:
        Exception: If the model is not found on Hugging Face
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
        )
        
        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        generation_config = GenerationConfig.from_pretrained(model_name)

        return model, tokenizer, generation_config
        
    except Exception as e:
        print(f"Model '{model_name}' was not found on Hugging Face or could not be loaded.")
        print(f"Original error: {e}")
        raise


def generate(model_tokenizer_config, model_input):
    """Generate text using the loaded model."""
    model, tokenizer, generation_config = model_tokenizer_config

    # Concatenate prompt and input with newline
    full_prompt = f"{model_input['prompt']}\n{model_input['sample']}"
    
    # Tokenize input
    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda:0")
    
    # Generate translation
    generate_ids = model.generate(
        **inputs,
        max_new_tokens=4096,
        generation_config=generation_config,
    )
    
    # Extract only the generated part (remove input prompt)
    generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
    translation = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    return translation
>>>>>>> 7145533 (merge the model-specific modules into one huggingface text module)

#!/usr/bin/env python3
"""
Download MoE models for Project 2

This script downloads the required Mixture of Experts models from HuggingFace:
- DeepSeek-MoE-16B-base
- Qwen1.5-MoE-A2.7B
"""

import os
from pathlib import Path

# Set up cache directory
PROJECT_DIR = Path(__file__).parent
MODEL_CACHE_DIR = PROJECT_DIR / "models" / "huggingface_cache"
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

os.environ['HF_HOME'] = str(MODEL_CACHE_DIR)
os.environ['TRANSFORMERS_CACHE'] = str(MODEL_CACHE_DIR)
os.environ['HF_DATASETS_CACHE'] = str(MODEL_CACHE_DIR)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

print(f"Model cache directory: {MODEL_CACHE_DIR}")
print(f"HuggingFace mirror: {os.environ.get('HF_ENDPOINT')}")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

def download_model(model_name, short_name):
    """Download a model from HuggingFace"""
    print(f"\n{'='*60}")
    print(f"Downloading: {short_name} ({model_name})")
    print(f"{'='*60}")
    
    try:
        print(f"Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=str(MODEL_CACHE_DIR)
        )
        print(f"Tokenizer downloaded successfully")
        
        print(f"Downloading config...")
        config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=str(MODEL_CACHE_DIR)
        )
        print(f"Config downloaded successfully")
        
        print(f"Downloading model weights (this may take a while)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=str(MODEL_CACHE_DIR),
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True
        )
        print(f"{short_name} model downloaded successfully!")
        
        # Clean up memory
        del model
        del tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        import gc
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"Failed to download {short_name}: {e}")
        return False

def main():
    print("Starting MoE model download...")
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    models = [
        ("deepseek-ai/deepseek-moe-16b-base", "DeepSeek-16B"),
        ("Qwen/Qwen1.5-MoE-A2.7B", "Qwen-MoE"),
    ]
    
    results = {}
    for model_name, short_name in models:
        success = download_model(model_name, short_name)
        results[short_name] = success
    
    print("\n" + "="*60)
    print("Download Summary:")
    print("="*60)
    for name, success in results.items():
        status = "Success" if success else "Failed"
        print(f"  {name}: {status}")
    
    if all(results.values()):
        print("\nAll models downloaded successfully!")
    else:
        print("\nSome models failed to download. Please check your network connection and try again.")

if __name__ == "__main__":
    main()

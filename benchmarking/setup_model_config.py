#!/usr/bin/env python3
"""
Helper script to quickly configure test_config.py for different models.
"""

import os

# Model configurations based on your setup instructions
MODEL_CONFIGS = {
    "med-llama3-8b": {
        "model_name": "med-llama3-8b",
        "model_type": "pretrain",
        "description": "Me-LLaMA (YBXL/Med-LLaMA3-8B)"
    },
    "qwen3-32b": {
        "model_name": "qwen3-32b", 
        "model_type": "instruct",
        "description": "Qwen3 without reasoning (Qwen/Qwen3-32B)"
    },
    "gpt-oss-20b": {
        "model_name": "gpt-oss-20b",
        "model_type": "instruct", 
        "description": "GPT-OSS (openai/gpt-oss-20b)"
    },
    "medgemma-27b-text-it": {
        "model_name": "medgemma-27b-text-it",
        "model_type": "instruct",
        "description": "MedGemma (google/medgemma-27b-text-it)"
    }
}

def update_config(model_key: str, base_url: str = "http://localhost:1234/v1"):
    """Update test_config.py with the specified model configuration."""
    
    if model_key not in MODEL_CONFIGS:
        print(f"❌ Unknown model: {model_key}")
        print(f"Available models: {list(MODEL_CONFIGS.keys())}")
        return False
    
    config = MODEL_CONFIGS[model_key]
    
    config_content = f'''#!/usr/bin/env python3
"""
Configuration file for quick testing.
Modify these settings for your server setup.
"""

# vLLM Server Configuration
MODEL_NAME = "{config['model_name']}"  # {config['description']}
BASE_URL = "{base_url}"  # Change if using different port/host
API_KEY = "EMPTY"  # Usually "EMPTY" for vLLM

# Model Type Configuration
# For Me-LLaMA: use "pretrain"
# For Qwen3, Gpt-oss, MedGemma: use "instruct"
MODEL_TYPE = "{config['model_type']}"  # Auto-configured for {config['description']}

# Testing Configuration
SAMPLE_SIZE = 5  # Number of samples per dataset (minimum 5)
MAX_SAMPLE_SIZE = 10  # Maximum samples to prevent long tests
TEMPERATURE = 0.0  # Generation temperature
MAX_TOKENS = 256  # Maximum tokens to generate

# Directory Configuration
DATA_DIR = "grouped_deduplicated_data"  # Where your final datasets are
OUTPUT_DIR = "test_results"  # Where to save test results

# Dataset groups to test
GROUPS_TO_TEST = [
    "literature_mc",
    "literature_open", 
    "exam_mc",
    "exam_open"
]
'''
    
    # Write the configuration
    with open("test_config.py", "w") as f:
        f.write(config_content)
    
    print(f"✅ Configuration updated for {config['description']}")
    print(f"   Model Name: {config['model_name']}")
    print(f"   Model Type: {config['model_type']}")
    print(f"   Base URL: {base_url}")
    print(f"\nYou can now run: python quick_test.py")
    
    return True

def main():
    """Interactive configuration setup."""
    
    print("=== vLLM Model Configuration Helper ===\n")
    
    print("Available models:")
    for i, (key, config) in enumerate(MODEL_CONFIGS.items(), 1):
        print(f"{i}. {key} - {config['description']} ({config['model_type']})")
    
    print(f"{len(MODEL_CONFIGS) + 1}. Custom configuration")
    
    try:
        choice = input(f"\nSelect model (1-{len(MODEL_CONFIGS) + 1}): ").strip()
        choice_num = int(choice)
        
        if choice_num == len(MODEL_CONFIGS) + 1:
            # Custom configuration
            model_name = input("Enter model name: ").strip()
            model_type = input("Enter model type (instruct/pretrain): ").strip()
            base_url = input("Enter base URL [http://localhost:1234/v1]: ").strip()
            
            if not base_url:
                base_url = "http://localhost:1234/v1"
            
            # Create custom config
            config_content = f'''#!/usr/bin/env python3
"""
Configuration file for quick testing.
Modify these settings for your server setup.
"""

# vLLM Server Configuration
MODEL_NAME = "{model_name}"  # Custom model
BASE_URL = "{base_url}"  # Change if using different port/host
API_KEY = "EMPTY"  # Usually "EMPTY" for vLLM

# Model Type Configuration
# For Me-LLaMA: use "pretrain"
# For Qwen3, Gpt-oss, MedGemma: use "instruct"
MODEL_TYPE = "{model_type}"  # Custom configuration

# Testing Configuration
SAMPLE_SIZE = 5  # Number of samples per dataset (minimum 5)
MAX_SAMPLE_SIZE = 10  # Maximum samples to prevent long tests
TEMPERATURE = 0.0  # Generation temperature
MAX_TOKENS = 256  # Maximum tokens to generate

# Directory Configuration
DATA_DIR = "grouped_deduplicated_data"  # Where your final datasets are
OUTPUT_DIR = "test_results"  # Where to save test results

# Dataset groups to test
GROUPS_TO_TEST = [
    "literature_mc",
    "literature_open", 
    "exam_mc",
    "exam_open"
]
'''
            
            with open("test_config.py", "w") as f:
                f.write(config_content)
            
            print(f"✅ Custom configuration created")
            print(f"   Model Name: {model_name}")
            print(f"   Model Type: {model_type}")
            print(f"   Base URL: {base_url}")
            
        elif 1 <= choice_num <= len(MODEL_CONFIGS):
            # Predefined configuration
            model_keys = list(MODEL_CONFIGS.keys())
            selected_key = model_keys[choice_num - 1]
            
            base_url = input("Enter base URL [http://localhost:1234/v1]: ").strip()
            if not base_url:
                base_url = "http://localhost:1234/v1"
            
            update_config(selected_key, base_url)
        else:
            print("❌ Invalid choice")
            return
        
        print(f"\nYou can now run: python quick_test.py")
        
    except (ValueError, KeyboardInterrupt):
        print("\n❌ Configuration cancelled")

if __name__ == "__main__":
    main()

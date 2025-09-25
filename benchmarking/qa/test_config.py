#!/usr/bin/env python3
"""
Configuration file for quick testing.
Modify these settings for your server setup.
"""

# vLLM Server Configuration
MODEL_NAME = "med-llama3-8b"  # Me-LLaMA (YBXL/Med-LLaMA3-8B)
BASE_URL = "http://localhost:1234/v1"  # Change if using different port/host
API_KEY = "EMPTY"  # Usually "EMPTY" for vLLM

# Model Type Configuration
# For Me-LLaMA: use "pretrain"
# For Qwen3, Gpt-oss, MedGemma: use "instruct"
MODEL_TYPE = "pretrain"  # Auto-configured for Me-LLaMA (YBXL/Med-LLaMA3-8B)

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

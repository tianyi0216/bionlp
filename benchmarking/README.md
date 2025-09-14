# Biomedical QA Benchmarking

Simple benchmarking setup for evaluating language models on biomedical question-answering tasks.

## How to run the experiment

Here is the quickest way to run the experiment:

Already finished runnning MedLLaMA3-8B with vLLM and GPT-OSS-20B with vLLM.

Need to run qwen3-32b with vLLM and medgemma-27b-text-it with vLLM.

You can change the hyperparameters in the `run.sh` script, and then run the experiment. The script runs inference on all datasets and saves the results to the `results/` directory for one specific model.

```bash

# 1. Set up environment only
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv vllm --python 3.12
source vllm/bin/activate
uv pip install vllm==0.10.1 --torch-backend=auto
uv pip install -r requirements.txt

# or use pip
python -m venv vllm
source vllm/bin/activate
pip install vllm==0.10.1 --torch-backend=auto
pip install -r requirements.txt

# 2. Run the complete experiment (includes deployment + evaluation)
./run.sh
```

## Quick Setup

### **1. Environment Setup:**

**Create virtual environment:**
```bash
# Option 1: Using uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv vllm --python 3.12
source vllm/bin/activate

# Option 2: Using conda
conda create -n vllm python=3.12
conda activate vllm

# Option 3: Using pip
python -m venv vllm
source vllm/bin/activate  # On Windows: vllm\Scripts\activate
```

### **2. Install Dependencies:**

**Install vLLM (for model deployment):**
```bash
# Using uv
uv pip install vllm==0.10.1 --torch-backend=auto

# Using pip
pip install vllm==0.10.1
```

**Install evaluation requirements:**
```bash
# Using uv
uv pip install -r requirements.txt

# Using pip
pip install -r requirements.txt
```

### **3. Deploy a Model (Example):**

**For CHTC servers:**
```bash
export no_proxy="localhost,127.0.0.1,::1"
```

**Deploy Me-LLaMA (pretrain format):**
```bash
CUDA_VISIBLE_DEVICES=0,1 python3 -m vllm.entrypoints.openai.api_server \
  --model YBXL/Med-LLaMA3-8B \
  --served-model-name med-llama3-8b \
  --tensor-parallel-size 2 \
  --dtype auto \
  --host 127.0.0.1 \
  --port 1234
```

### **4. Configure and Test:**

**Set up model configuration:**
```bash
# Interactive configuration (recommended)
python setup_model_config.py

# Or manually edit test_config.py
# Set MODEL_TYPE = "pretrain" for Me-LLaMA
# Set MODEL_TYPE = "instruct" for Qwen3, GPT-OSS, MedGemma
```

**Run quick test to verify everything works:**
```bash
# Test with 5 samples per dataset group
python quick_test.py
```

This will:
- Test server connection
- Sample 5 questions from each dataset group
- Run evaluation pipeline
- Show accuracy/F1 scores and save results to `test_results/`

## Automated Deployment and Evaluation

### **Complete Pipeline with `run.sh`:**

For fully automated deployment and evaluation, use the `run.sh` script:

```bash
# After environment setup, run everything automatically
./run.sh
```

**What `run.sh` does:**
1. **Deploys vLLM server** with Me-LLaMA model automatically
2. **Waits for server** to be ready (up to 10 minutes)
3. **Runs evaluation** on all 4 dataset groups
4. **Saves results** to `results/` directory
5. **Cleans up** server process when finished

### **Customizing the Run:**

**Change model (set before running):**
```bash
# Use Qwen3 instead of Me-LLaMA
MODEL_NAME="Qwen/Qwen3-32B" SERVED_MODEL_NAME="qwen3-32b" \
USE_INSTRUCT="true" CHAT_TEMPLATE="qwen3_nonthinking" ./run.sh
```

**Run specific datasets only:**
```bash
# Run only exam datasets
DATASETS="exam_mc,exam_open" ./run.sh

# Run only multiple choice datasets  
DATASETS="literature_mc,exam_mc" ./run.sh
```

**Adjust evaluation parameters:**
```bash
# Use different sample size and temperature
SAMPLE_SIZE="50" TEMPERATURE="0.3" ./run.sh
```

**All configurable parameters:**
- `MODEL_NAME`: HuggingFace model path (default: "YBXL/Med-LLaMA3-8B")
- `SERVED_MODEL_NAME`: Model name for API (default: "med-llama3-8b")  
- `USE_INSTRUCT`: "true" for instruct, "false" for pretrain (default: "false")
- `DATASETS`: Comma-separated list (default: "literature_mc,literature_open,exam_mc,exam_open")
- `SAMPLE_SIZE`: Samples per dataset (default: "100")
- `TEMPERATURE`: Generation temperature (default: "0.0")
- `MAX_TOKENS`: Max tokens to generate (default: "256")
- `TENSOR_PARALLEL_SIZE`: GPU parallelism (default: "2")
- `CHAT_TEMPLATE`: Template for Qwen3 (use "qwen3_nonthinking")

---

## 1. Dataset Selection

We created a balanced 10K dataset through deduplication and quality-based sampling:

### **Dataset Structure:**
- **Total**: 10,000 samples
- **4 Groups**: 2,500 samples each
  - `literature_mc`: Literature-based multiple choice (hoc, PubMedQA)
  - `literature_open`: Literature-based open-ended (NFCorpus, BioNLI, BC5CDR)  
  - `exam_mc`: Medical exam multiple choice (MedMCQA, JAMA, MedBullets)
  - `exam_open`: Medical exam open-ended (MedQA-USMLE, LiveQA, MedicationQA, MedQuAD, MeQSum)

### **Quality Control:**
- **Deduplication**: Removed similar questions using BiomedBERT embeddings
- **Quality Ranking**: Prioritized high-quality datasets (JAMA > MedMCQA, PubMedQA > hoc)
- **Diversity Sampling**: Ensured representation from all source datasets

## 2. Evaluation Metrics

### **Multiple Choice Questions:**
- **Accuracy**: Percentage of correct answers
- **Top-3 Accuracy**: Model's top 3 choices include correct answer
- **Confidence Scores**: Probability distribution over options

### **Open-Ended Questions:**
- **F1 Score**: Token-level overlap with reference answers
- **ROUGE-1/2/L**: N-gram and longest common subsequence overlap
- **BLEU Score**: Translation-quality metric adapted for QA
- **Medical Similarity**: Semantic similarity using BiomedBERT embeddings

## 3. Evaluation Methods

### **Prompt-Based Evaluation:**
Models generate text responses that are parsed for answers.

**Multiple Choice Prompt:**
```
You are a medical expert. Answer the following medical question by selecting the most appropriate option.

Question: [question text]

Options:
A. [option A]
B. [option B] 
C. [option C]
D. [option D]

Respond with exactly one letter (A, B, C, D, or E).
```

**Open-Ended Prompt:**
```
You are a medical expert. Provide a clear, accurate, and concise answer to the following medical question.

Question: [question text]

Answer:
```

### **Dataset-Specific Formats:**
- **PubMedQA**: Yes/No/Maybe options with specific instructions
- **HOC**: Multiple cancer hallmark selection (A-K options)
- **Standard MC**: Single letter selection (A/B/C/D/E)

### **Domain-Specific Instructions:**
- **Medical**: "You are a medical expert..."
- **Literature**: "You are analyzing biomedical literature..."  
- **Exam**: "You are taking a medical exam..."

## 4. Files and Usage

### **Core Evaluation Files:**
- **`evaluation.py`**: Main evaluation script for vLLM/CHTC deployment
- **`eval_functions.py`**: Core prompt creation and answer extraction functions
- **`metrics.py`**: Comprehensive evaluation metrics (accuracy, F1, ROUGE, BLEU, medical similarity)
- **`quick_test.py`**: Quick testing script for vLLM server validation

### **Data Processing Files:**
- **`convert_qa_format.py`**: Converts raw datasets to standardized format with MC options
- **`deduplicate_grouped_qa.py`**: Deduplication and quality-based sampling pipeline

### **Configuration Files:**
- **`test_config.py`**: Configuration for quick testing (model type, server settings)
- **`setup_model_config.py`**: Interactive setup for test configuration

### **Testing Files:**
- **`test_mc_conversion.py`**: Tests MC dataset format conversion
- **`test_mc_answers.py`**: Validates MC answer formats
- **`test_metrics_integration.py`**: Tests metric calculation integration

### **Deployment Files:**
- **`Dockerfile`**: Container setup for CHTC deployment
- **`deploy.sh`**: Deployment script for HTCondor
- **`run.sh`**: Job execution script
- **`requirements.txt`**: Python dependencies

## 5. Additional Model Deployments

**Other supported models:**

**Qwen3 (without reasoning) - Use `instruct` format:**
```bash
# Download non-thinking template first
wget https://qwen.readthedocs.io/en/latest/_downloads/c101120b5bebcc2f12ec504fc93a965e/qwen3_nonthinking.jinja

# Deploy
CUDA_VISIBLE_DEVICES=0,1 python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-32B \
  --served-model-name qwen3-32b \
  --tensor-parallel-size 2 \
  --chat-template ./qwen3_nonthinking.jinja \
  --dtype auto \
  --host 127.0.0.1 \
  --port 1234
```

**GPT-OSS - Use `instruct` format:**
```bash
CUDA_VISIBLE_DEVICES=0,1 python3 -m vllm.entrypoints.openai.api_server \
  --model openai/gpt-oss-20b \
  --served-model-name gpt-oss-20b \
  --tensor-parallel-size 2 \
  --async-scheduling \
  --dtype auto \
  --host 127.0.0.1 \
  --port 1234
```

**MedGemma - Use `instruct` format:**
```bash
CUDA_VISIBLE_DEVICES=0,1 python3 -m vllm.entrypoints.openai.api_server \
  --model google/medgemma-27b-text-it \
  --served-model-name medgemma-27b-text-it \
  --tensor-parallel-size 2 \
  --dtype auto \
  --host 127.0.0.1 \
  --port 1234
```

**Model Type Reference:**
| Model | Format | Configuration |
|-------|--------|---------------|
| Qwen3 (non-reasoning) | `instruct` | `MODEL_TYPE = "instruct"` |
| GPT-OSS | `instruct` | `MODEL_TYPE = "instruct"` |
| MedGemma | `instruct` | `MODEL_TYPE = "instruct"` |
| Me-LLaMA | `pretrain` | `MODEL_TYPE = "pretrain"` |

## 6. Usage Examples

### **For vLLM Testing:**
```bash
# 1. Configure your model type
python setup_model_config.py

# 2. Run quick test (samples 5 questions per group)
python quick_test.py

# 3. Run full evaluation
python evaluation.py --model_name "your-model-name"
```

### **For Custom Integration:**
```python
from eval_functions import create_mc_prompt, extract_mc_answer, load_dataset
from metrics import evaluate_mc_complete, evaluate_open_complete

# Load data
data = load_dataset("exam_mc")

# Create prompts
prompt = create_mc_prompt(question, options, "medical", "MedMCQA")

# Get model response (your implementation)
response = your_model(prompt)

# Extract answer
answer = extract_mc_answer(response, valid_options, "MedMCQA")

# Calculate metrics
results = evaluate_mc_complete(predictions, targets, dataset_names)
```

Simple and clean evaluation for biomedical language models.

# Biomedical QA Benchmarking

Simple benchmarking setup for evaluating language models on biomedical question-answering tasks.

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
- **Confidence Scores**: Probability distribution over options (logit-based)

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

Answer: The correct answer is
```

**Open-Ended Prompt:**
```
You are a medical expert. Provide a clear, accurate, and concise answer to the following medical question.

Question: [question text]

Answer:
```

### **Logit-Based Evaluation:**
Extract probability distributions over answer choices directly from model logits (for multiple choice only).

### **Domain-Specific Instructions:**
- **Medical**: "You are a medical expert..."
- **Literature**: "You are analyzing biomedical literature..."  
- **Exam**: "You are taking a medical exam..."

## Usage

1. **Load dataset**: Use `load_dataset(group_name)` 
2. **Create prompts**: Use `create_mc_prompt()` or `create_open_prompt()`
3. **Get model response**: Call your model with the prompt
4. **Extract answer**: Use `extract_mc_answer()` for MC questions
5. **Calculate metrics**: Use functions from `metrics.py`

Simple and clean evaluation for biomedical language models.

# SCOTUS AI Tokenization Pipeline

This module is responsible for the critical task of converting raw textual data—justice biographies and case descriptions—into a numerical format suitable for deep learning models. It employs a multi-stage pipeline that tokenizes text using `SentenceTransformer` models and serializes the output for efficient use during model training and pretraining.

## 🎯 Core Objective

The primary goal of this pipeline is to **pre-tokenize** all necessary text files referenced in the main `case_dataset.json`. Instead of tokenizing text on-the-fly during training, which would create a significant computational bottleneck, this process generates tokenized representations ahead of time. The output is saved in compressed `pickle` files, allowing for rapid loading of training batches.

This approach accelerates the training cycle by offloading the tokenization overhead to a one-time preprocessing step.

## 🏗️ Architecture & Workflow

The tokenization process is orchestrated by a main script that intelligently manages several specialized worker scripts. This design ensures modularity and allows for targeted re-tokenization if necessary.

![Tokenization Workflow](https://storage.googleapis.com/agent-tools-prod.appspot.com/tool-results/v1/files/42071d02-23f2-4547-8898-d102123d9b4b)

**1. Orchestrator (`main_encoder.py`)**
This is the central entry point for the pipeline. It does not perform tokenization itself but instead manages the entire workflow:
-   **Analyzes `case_dataset.json`**: It first determines the complete set of unique biography and case description files required for training.
-   **Checks Existing State**: It intelligently detects which files have already been tokenized by checking the contents of existing output files.
-   **Creates a Plan**: Based on the analysis, it builds a plan to tokenize only the missing files, making resumption from interruption or partial completion highly efficient.
-   **Delegates to Workers**: It invokes the appropriate worker scripts (`encode_bios.py`, `encode_descriptions.py`) with a list of files to process.

**2. Worker Scripts**
-   **`encode_bios.py`**: Handles the tokenization of all justice biographies.
-   **`encode_descriptions.py`**: Handles the tokenization of all legal case descriptions.
-   **`encode_pretraining.py`**: A specialized script that prepares the data specifically for the [Contrastive Justice Pretraining](../pretraining/README.md). It tokenizes both the "truncated" and "full" biographies and extracts temporal metadata (appointment year) for each justice, saving them into dedicated files.

**3. Tokenizer**
All worker scripts utilize the `AutoTokenizer` from the Hugging Face `transformers` library. The specific pre-trained model used for tokenization is defined in the configuration.

## ⚙️ Configuration (`config.py`, `config.env`)

All aspects of the tokenization pipeline are controlled via the `scripts/tokenization/config.env` file. This allows for easy modification of models, paths, and batch sizes without changing the code.

```bash
# Model selection for different text types
BIO_MODEL_NAME=sentence-transformers/all-roberta-large-v1
DESCRIPTION_MODEL_NAME=sentence-transformers/all-roberta-large-v1

# Batching and device configuration
BIO_BATCH_SIZE=16
DESCRIPTION_BATCH_SIZE=8
DEVICE=auto # 'cuda', 'cpu', or 'auto'

# Input and Output file paths
BIO_INPUT_DIR=data/processed/bios
DESCRIPTION_INPUT_DIR=data/processed/case_descriptions
BIO_OUTPUT_FILE=data/processed/encoded_bios.pkl
DESCRIPTION_OUTPUT_FILE=data/processed/encoded_descriptions.pkl

# Paths for the contrastive pretraining task data
PRETRAINING_TRUNC_BIO_FILE=data/processed/encoded_pre_trunc_bios.pkl
PRETRAINING_FULL_BIO_FILE=data/processed/encoded_pre_full_bios.pkl
PRETRAINING_DATASET_FILE=data/processed/pretraining_dataset.json
```

## 📊 Output Format

The pipeline generates several `.pkl` files, which are Python pickle files containing serialized dictionaries. This format is highly efficient for loading large amounts of data.

-   `encoded_bios.pkl` & `encoded_descriptions.pkl`: Used for the main vote prediction task.
-   `encoded_pre_trunc_bios.pkl` & `encoded_pre_full_bios.pkl`: Used for the contrastive pretraining task.

Each `.pkl` file has the following internal structure:

```python
{
    'tokenized_data': {
        'path/to/file1.txt': {
            'input_ids': tensor([...]),
            'attention_mask': tensor([...])
        },
        'path/to/file2.txt': { ... }
    },
    'metadata': {
        'model_name': 'sentence-transformers/all-roberta-large-v1',
        'max_sequence_length': 512,
        'num_tokenized': 8116,
        'tokenization_method': 'transformers_autotokenizer'
    }
}
```

The `pretraining_dataset.json` file has a simpler structure, mapping each justice's file name to their appointment year and nominating president for temporal splitting during training.

```json
{
    "John_Jay.txt": [ "1789", "George Washington" ],
    "John_Roberts.txt": [ "2005", "George W. Bush" ]
}
```

## 🚀 How to Run

The pipeline is designed to be simple to execute from the command line.

**1. Main Encoder (Recommended)**
This is the preferred method, as it intelligently handles all steps.
```bash
# Run the complete, smart tokenization pipeline
python scripts/tokenization/main_encoder.py

# Check status without tokenizing
python scripts/tokenization/main_encoder.py --check

# Tokenize only missing biography files
python scripts/tokenization/main_encoder.py --bios-only

# Force re-tokenization of all required files
python scripts/tokenization/main_encoder.py --force
```

**2. Pretraining Data Encoder**
This script must be run separately to prepare the data for the contrastive learning task.
```bash
python scripts/tokenization/encode_pretraining.py
```

## 📚 References

-   **Hugging Face Transformers**: Wolf, T., et al. (2020). *Transformers: State-of-the-Art Natural Language Processing*. [Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations](https://www.aclweb.org/anthology/2020.emnlp-demos.6/).
-   **Sentence-Transformers**: Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. [arXiv:1908.10084](https://arxiv.org/abs/1908.10084). 
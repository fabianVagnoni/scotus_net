# SCOTUS AI Encoding Configuration System

This directory contains the encoding pipeline for SCOTUS AI with centralized configuration management.

## 📁 Files Overview

- **`config.env`** - Central configuration file with all encoding hyperparameters
- **`config.py`** - Configuration loader module
- **`encode_bios.py`** - Biography encoding script
- **`encode_descriptions.py`** - Case description encoding script  
- **`main_encoder.py`** - Main encoding pipeline orchestrator

## ⚙️ Configuration System

### Configuration File (`config.env`)

All encoding hyperparameters are centralized in `config.env`:

```bash
# Model Configuration
BIO_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
DESCRIPTION_MODEL_NAME=nlpaueb/legal-bert-base-uncased

# Embedding Configuration
EMBEDDING_DIM=384
MAX_SEQUENCE_LENGTH=512

# Batch Processing
BIO_BATCH_SIZE=16
DESCRIPTION_BATCH_SIZE=8

# Device Configuration
DEVICE=auto  # Options: cuda, cpu, auto

# File Paths
BIO_OUTPUT_FILE=data/processed/encoded_bios.pkl
DESCRIPTION_OUTPUT_FILE=data/processed/encoded_descriptions.pkl
DATASET_FILE=data/processed/case_dataset.json
BIO_INPUT_DIR=data/processed/bios
DESCRIPTION_INPUT_DIR=data/processed/case_descriptions

# Processing Settings
MAX_DESCRIPTION_WORDS=10000
SHOW_PROGRESS=true
CLEAR_CACHE_ON_OOM=true
RANDOM_SEED=42
```

### Configuration Loader (`config.py`)

The configuration loader provides:

- **Automatic loading** of `config.env` from the same directory
- **Type conversion** (string, int, float, bool, path)
- **Default values** if config file is missing
- **Convenient properties** for common settings
- **Validation** with error handling

#### Usage Examples:

```python
from config import get_config, get_bio_config, get_description_config

# Get full configuration
config = get_config()
print(f"Bio model: {config.bio_model_name}")
print(f"Batch size: {config.bio_batch_size}")

# Get specific configuration dictionaries
bio_config = get_bio_config()
desc_config = get_description_config()

# Use custom config file
config = get_config("custom_config.env")
```

## 🚀 Usage

### 1. Main Encoding Pipeline

Run the complete encoding pipeline:

```bash
# From project root
python src/models/encoding/main_encoder.py

# Check status only
python src/models/encoding/main_encoder.py --check

# Encode only biographies
python src/models/encoding/main_encoder.py --bios-only

# Encode only descriptions  
python src/models/encoding/main_encoder.py --descriptions-only

# Force re-encoding
python src/models/encoding/main_encoder.py --force

# Use custom config
python src/models/encoding/main_encoder.py --config my_config.env
```

### 2. Individual Encoding Scripts

#### Biography Encoding:

```bash
# Use config defaults
python src/models/encoding/encode_bios.py

# Override specific parameters
python src/models/encoding/encode_bios.py --batch-size 32 --device cuda

# Encode specific files
python src/models/encoding/encode_bios.py --file-list bio_files.txt

# Use custom config
python src/models/encoding/encode_bios.py --config my_config.env
```

#### Description Encoding:

```bash
# Use config defaults
python src/models/encoding/encode_descriptions.py

# Override specific parameters
python src/models/encoding/encode_descriptions.py --batch-size 4 --device cpu

# Encode specific files
python src/models/encoding/encode_descriptions.py --file-list desc_files.txt

# Use custom config
python src/models/encoding/encode_descriptions.py --config my_config.env
```

## 📊 Configuration Parameters

### Model Settings

| Parameter | Description | Default |
|-----------|-------------|---------|
| `BIO_MODEL_NAME` | HuggingFace model for biography encoding | `sentence-transformers/all-MiniLM-L6-v2` |
| `DESCRIPTION_MODEL_NAME` | HuggingFace model for description encoding | `nlpaueb/legal-bert-base-uncased` |
| `EMBEDDING_DIM` | Target embedding dimension | `384` |
| `MAX_SEQUENCE_LENGTH` | Maximum sequence length for tokenization | `512` |

### Batch Processing

| Parameter | Description | Default |
|-----------|-------------|---------|
| `BIO_BATCH_SIZE` | Batch size for biography encoding | `16` |
| `DESCRIPTION_BATCH_SIZE` | Batch size for description encoding | `8` |
| `DEVICE` | Computing device (`cuda`, `cpu`, `auto`) | `auto` |

### File Paths

| Parameter | Description | Default |
|-----------|-------------|---------|
| `DATASET_FILE` | Main dataset JSON file | `data/processed/case_dataset.json` |
| `BIO_INPUT_DIR` | Biography files directory | `data/processed/bios` |
| `DESCRIPTION_INPUT_DIR` | Description files directory | `data/processed/case_descriptions` |
| `BIO_OUTPUT_FILE` | Biography embeddings output | `data/processed/encoded_bios.pkl` |
| `DESCRIPTION_OUTPUT_FILE` | Description embeddings output | `data/processed/encoded_descriptions.pkl` |

### Processing Settings

| Parameter | Description | Default |
|-----------|-------------|---------|
| `MAX_DESCRIPTION_WORDS` | Skip descriptions larger than this | `10000` |
| `SHOW_PROGRESS` | Show progress bars | `true` |
| `CLEAR_CACHE_ON_OOM` | Clear GPU cache on out-of-memory | `true` |
| `RANDOM_SEED` | Random seed for reproducibility | `42` |
| `NUM_WORKERS` | CPU threads for data loading | `4` |

## 🔧 Customization

### Creating Custom Configurations

1. **Copy the default config:**
   ```bash
   cp src/models/encoding/config.env my_custom_config.env
   ```

2. **Edit parameters:**
   ```bash
   # Increase batch sizes for powerful GPUs
   BIO_BATCH_SIZE=32
   DESCRIPTION_BATCH_SIZE=16
   
   # Use different models
   BIO_MODEL_NAME=sentence-transformers/all-mpnet-base-v2
   DESCRIPTION_MODEL_NAME=microsoft/DialoGPT-medium
   
   # Custom paths
   BIO_OUTPUT_FILE=experiments/run1/bios.pkl
   DESCRIPTION_OUTPUT_FILE=experiments/run1/descriptions.pkl
   ```

3. **Use custom config:**
   ```bash
   python src/models/encoding/main_encoder.py --config my_custom_config.env
   ```

### Environment-Specific Configs

Create different configs for different environments:

- `config.env` - Default/development
- `config_production.env` - Production settings
- `config_gpu.env` - High-performance GPU settings
- `config_cpu.env` - CPU-only settings

### Override Priority

Configuration values are resolved in this order (highest to lowest priority):

1. **Command-line arguments** (e.g., `--batch-size 32`)
2. **Custom config file** (e.g., `--config custom.env`)
3. **Default config file** (`config.env`)
4. **Hard-coded defaults** (in `config.py`)

## 🐛 Troubleshooting

### Config File Not Found
```
⚠️  Config file not found: config.env
Using default values...
```
**Solution:** Ensure `config.env` exists in the same directory as the script, or specify a custom config with `--config`.

### Invalid Parameter Values
```
⚠️  Invalid integer value for BIO_BATCH_SIZE: abc, using default: 16
```
**Solution:** Check that numeric parameters contain valid numbers in `config.env`.

### GPU Out of Memory
```
⚠️  GPU memory issue at batch 5. Clearing cache...
```
**Solution:** Reduce batch sizes in `config.env`:
```bash
BIO_BATCH_SIZE=8
DESCRIPTION_BATCH_SIZE=4
```

### Model Loading Errors
```
❌ Error loading model: nlpaueb/legal-bert-base-uncased
```
**Solution:** Verify model names are correct in `config.env`, or use alternative models:
```bash
# Alternative legal models
DESCRIPTION_MODEL_NAME=microsoft/DialoGPT-medium
DESCRIPTION_MODEL_NAME=nlpaueb/legal-bert-small-uncased
```

## 📈 Performance Tuning

### For High-End GPUs (RTX 4090, A100)
```bash
BIO_BATCH_SIZE=64
DESCRIPTION_BATCH_SIZE=32
DEVICE=cuda
```

### For Mid-Range GPUs (RTX 3070, RTX 4070)
```bash
BIO_BATCH_SIZE=32
DESCRIPTION_BATCH_SIZE=16
DEVICE=cuda
```

### For CPU-Only Systems
```bash
BIO_BATCH_SIZE=8
DESCRIPTION_BATCH_SIZE=4
DEVICE=cpu
NUM_WORKERS=8
```

### For Memory-Constrained Systems
```bash
BIO_BATCH_SIZE=4
DESCRIPTION_BATCH_SIZE=2
MAX_DESCRIPTION_WORDS=5000
CLEAR_CACHE_ON_OOM=true
```

## 🔄 Integration

The configuration system is automatically integrated into all encoding scripts. No code changes are needed to use the centralized configuration - just edit `config.env` and run the scripts normally.

All scripts support the `--config` parameter to use custom configuration files, allowing for flexible experimentation and deployment scenarios. 
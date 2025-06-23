# SCOTUS AI Tokenization

The tokenization module handles text encoding and embedding generation for justice biographies and case descriptions. It uses sentence transformer models to create high-quality embeddings for the machine learning pipeline.

## 🎯 Overview

The tokenization system converts raw text files into numerical embeddings that can be used by the SCOTUS voting prediction model:

```
Text Files → Sentence Transformers → Embeddings → ML Model
     ↓               ↓                   ↓          ↓
[Raw Text]    [Pre-trained Models]  [Vectors]  [Predictions]
```

## 🤖 Model Architecture

### Text Encoders

1. **Biography Encoder**: `sentence-transformers/all-MiniLM-L6-v2`
   - General-purpose sentence transformer
   - 384-dimensional embeddings
   - Optimized for biographical text and general content

2. **Case Description Encoder**: `Stern5497/sbert-legal-xlm-roberta-base`
   - Legal domain-specialized model
   - 384-dimensional embeddings  
   - Fine-tuned on legal text and court documents

### Embedding Process

```python
Text → Tokenization → Model Encoding → Embeddings
  ↓         ↓              ↓             ↓
"John..."  [101,1188,...]  RoBERTa     [0.1,0.3,...]
```

## 🏗️ Module Components

### `encode_bios.py`
Justice biography encoding:
- Loads justice biography text files
- Encodes using biography-specific transformer model
- Saves embeddings with metadata mapping
- Handles batch processing for efficiency

### `encode_descriptions.py`
Case description encoding:
- Processes case description files
- Uses legal-specialized transformer model
- Creates embeddings for case content
- Supports dataset-driven encoding

### `main_encoder.py`
Complete tokenization pipeline:
- Orchestrates biography and description encoding
- Manages dependencies and file validation
- Supports incremental and batch processing
- Provides comprehensive progress tracking

### `config.py`
Configuration management:
- Model selection and parameters
- Processing configuration
- Output path management
- Device and resource settings

## 🚀 Usage

### Complete Tokenization Pipeline

```bash
# Run full tokenization pipeline
python scripts/tokenization/main_encoder.py

# Tokenize only biographies
python scripts/tokenization/main_encoder.py --bios-only

# Tokenize only case descriptions
python scripts/tokenization/main_encoder.py --descriptions-only

# Force re-tokenization of existing files
python scripts/tokenization/main_encoder.py --force-retokenization
```

### Individual Encoding

```bash
# Encode justice biographies
python scripts/tokenization/encode_bios.py \
    --input data/processed/bios \
    --output data/processed/encoded_bios.pkl

# Encode case descriptions
python scripts/tokenization/encode_descriptions.py \
    --input data/processed/case_descriptions \
    --output data/processed/encoded_descriptions.pkl
```

### Python API Usage

```python
from scripts.tokenization.encode_bios import encode_biography_files
from scripts.tokenization.encode_descriptions import encode_description_files

# Encode biographies
encode_biography_files(
    bios_dir="data/processed/bios",
    output_file="data/processed/encoded_bios.pkl",
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Encode case descriptions
encode_description_files(
    descriptions_dir="data/processed/case_descriptions", 
    output_file="data/processed/encoded_descriptions.pkl",
    model_name="Stern5497/sbert-legal-xlm-roberta-base"
)
```

## ⚙️ Configuration

### Environment Configuration (`config.env`)

```bash
# Model Configuration
BIO_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
DESCRIPTION_MODEL_NAME=Stern5497/sbert-legal-xlm-roberta-base
EMBEDDING_DIM=384
MAX_SEQUENCE_LENGTH=512

# Processing Configuration
BIO_BATCH_SIZE=32
DESCRIPTION_BATCH_SIZE=16
DEVICE=auto
NUM_WORKERS=0

# Input/Output Paths
BIO_INPUT_DIR=data/processed/bios
DESCRIPTION_INPUT_DIR=data/processed/case_descriptions
BIO_OUTPUT_FILE=data/processed/encoded_bios.pkl
DESCRIPTION_OUTPUT_FILE=data/processed/encoded_descriptions.pkl

# Processing Options
SHOW_PROGRESS=true
CLEAR_CACHE_ON_OOM=true
USE_MODEL_CACHE=true
```

### Advanced Configuration

```python
from scripts.tokenization.config import EncodingConfig

# Load custom configuration
config = EncodingConfig("custom_config.env")

# Access configuration values
print(f"Bio model: {config.bio_model_name}")
print(f"Embedding dimension: {config.embedding_dim}")
print(f"Device: {config.device}")
```

## 📊 Output Format

### Encoded Data Structure

The tokenization pipeline outputs pickle files containing:

```python
# encoded_bios.pkl structure
{
    'embeddings': {
        'path/to/bio1.txt': numpy.array([0.1, 0.3, ...]),  # 384-dim
        'path/to/bio2.txt': numpy.array([0.2, 0.1, ...]),
        # ... more embeddings
    },
    'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
    'embedding_dim': 384,
    'total_files': 116,
    'creation_date': '2024-01-15T10:30:00'
}
```

### Loading Encoded Data

```python
import pickle

# Load biography embeddings
with open('data/processed/encoded_bios.pkl', 'rb') as f:
    bio_data = pickle.load(f)

bio_embeddings = bio_data['embeddings']
model_name = bio_data['model_name']

# Load case description embeddings  
with open('data/processed/encoded_descriptions.pkl', 'rb') as f:
    desc_data = pickle.load(f)

desc_embeddings = desc_data['embeddings']
```

## 🔧 Pipeline Features

### Smart Processing

- **Incremental Encoding**: Only processes new or modified files
- **Resume Capability**: Continues from where processing left off
- **Validation**: Checks file availability before processing
- **Conflict Resolution**: Handles file path mismatches

### Batch Processing

```python
# Efficient batch processing
for batch in batch_iterator(text_files, batch_size=32):
    embeddings = model.encode(batch)
    # Process batch results
```

### Memory Management

- **Automatic Cache Clearing**: Handles GPU memory overflow
- **Batch Size Optimization**: Adjusts batch size based on available memory
- **Progress Tracking**: Real-time progress with memory usage monitoring

### Error Handling

- **File Validation**: Ensures all referenced files exist
- **Model Loading**: Graceful handling of model download failures
- **Encoding Errors**: Continues processing despite individual file failures

## 📈 Performance Optimization

### GPU Acceleration

```bash
# Enable GPU processing
DEVICE=cuda python scripts/tokenization/main_encoder.py

# Automatic device selection
DEVICE=auto python scripts/tokenization/main_encoder.py
```

### Batch Size Tuning

```python
# Optimize batch size for your hardware
BIO_BATCH_SIZE=64      # Larger batch for biographies
DESCRIPTION_BATCH_SIZE=16  # Smaller batch for longer case descriptions
```

### Model Caching

```python
# Enable model caching to avoid re-downloading
USE_MODEL_CACHE=true

# Cache location (default: ~/.cache/huggingface)
export TRANSFORMERS_CACHE=/path/to/cache
```

## 🧪 Testing and Validation

### Tokenization Status Check

```bash
# Check tokenization status without processing
python scripts/tokenization/main_encoder.py --check-status
```

### Validation Tests

```python
from scripts.tokenization.main_encoder import validate_final_tokenizations

# Validate tokenization completeness
validation_results = validate_final_tokenizations(
    dataset_file="data/processed/case_dataset.json",
    bio_tokenized_file="data/processed/encoded_bios.pkl",
    description_tokenized_file="data/processed/encoded_descriptions.pkl"
)

print(f"Bio coverage: {validation_results['bio_coverage']}")
print(f"Description coverage: {validation_results['description_coverage']}")
```

### Performance Benchmarks

| Component | Files | Avg Time | GPU Memory |
|-----------|-------|----------|------------|
| Biography Encoding | 116 files | ~2 minutes | ~1GB |
| Case Description Encoding | ~8,000 files | ~15 minutes | ~2GB |
| Complete Pipeline | All files | ~20 minutes | ~2GB |

## 📁 Output Structure

### Generated Files

```
data/processed/
├── encoded_bios.pkl           # Biography embeddings
├── encoded_descriptions.pkl   # Case description embeddings
└── tokenization_log.json     # Processing metadata
```

### Embedding Statistics

```python
# Get embedding statistics
from scripts.tokenization.encode_bios import load_tokenized_bios

embeddings, metadata = load_tokenized_bios("data/processed/encoded_bios.pkl")

print(f"Total embeddings: {len(embeddings)}")
print(f"Embedding dimension: {metadata['embedding_dim']}")
print(f"Model used: {metadata['model_name']}")
```

## 🚨 Important Considerations

### Model Requirements

- **Internet Connection**: Required for initial model download
- **Storage Space**: ~2GB for model caching
- **Memory**: 8GB+ RAM recommended for batch processing
- **GPU**: Optional but recommended for faster processing

### Data Preprocessing

- **Text Cleaning**: Automatic handling of encoding issues
- **Length Limits**: Respects model maximum sequence lengths
- **Format Validation**: Ensures proper text file formatting

### Compatibility

- **Model Versions**: Specific transformer model versions for reproducibility
- **Python Requirements**: Compatible with PyTorch and transformers library
- **Platform Support**: Cross-platform compatibility (Windows, macOS, Linux)

## 🛠️ Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| CUDA out of memory | Batch size too large | Reduce batch size in config |
| Model download fails | Network/proxy issues | Check internet connection |
| File not found | Missing input files | Run data pipeline first |
| Encoding errors | Text format issues | Check file encoding (UTF-8) |

### Debug Mode

```bash
# Enable detailed logging
python scripts/tokenization/main_encoder.py --verbose

# Test model loading
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### Memory Optimization

```python
# Clear cache manually
import torch
torch.cuda.empty_cache()

# Reduce batch size
BIO_BATCH_SIZE=16
DESCRIPTION_BATCH_SIZE=8
```

## 📞 Support

For tokenization issues:
- Verify input files exist and are readable
- Check model download and caching
- Monitor GPU memory usage during processing
- Review configuration parameters in `config.env`

---

**Ready to encode your text data?** Start with `python scripts/tokenization/main_encoder.py` 🔤 
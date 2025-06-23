# SCOTUS AI Configuration

This directory contains configuration files for the SCOTUS AI system. The configuration system uses YAML files for structured settings and environment variables for sensitive or deployment-specific values.

## 📁 Configuration Files

### `base_config.yaml`
Main configuration file containing all system settings:
- Data paths and directories
- Model architecture parameters
- Training hyperparameters
- Pipeline configuration
- Logging settings

## 🔧 Configuration Structure

### Data Configuration
```yaml
data:
  raw_dir: "data/raw"                    # Raw scraped data
  processed_dir: "data/processed"        # Processed datasets
  external_dir: "data/external"          # External data sources
  output_dir: "models_output"            # Model outputs
```

### Model Configuration
```yaml
model:
  # Text encoding models
  bio_model_name: "sentence-transformers/all-MiniLM-L6-v2"
  description_model_name: "Stern5497/sbert-legal-xlm-roberta-base"
  
  # Architecture parameters
  embedding_dim: 384
  hidden_dim: 512
  dropout_rate: 0.1
  max_justices: 15
  
  # Attention mechanism
  use_justice_attention: true
  num_attention_heads: 4
```

### Training Configuration
```yaml
training:
  # Optimization
  learning_rate: 0.0001
  weight_decay: 0.01
  batch_size: 4
  num_epochs: 10
  
  # Regularization
  dropout_rate: 0.1
  max_grad_norm: 1.0
  
  # Scheduling
  patience: 5
  lr_scheduler_factor: 0.5
  lr_scheduler_patience: 3
  
  # Loss function
  loss_function: "kl_div"
  kl_reduction: "batchmean"
```

### Pipeline Configuration
```yaml
pipeline:
  # Processing steps
  steps:
    - scrape_justices
    - scrape_bios
    - download_scdb
    - process_cases
    - scrape_descriptions
    - process_bios
    - create_metadata
    - create_descriptions
    - build_dataset
  
  # Processing options
  interactive_mode: true
  resume_enabled: true
  verbose_output: true
```

## 🌍 Environment Variables

### API Configuration
```bash
# Required for case description filtering
GEMMA_KEY=your_gemini_api_key_here

# Optional: Additional API keys
OPENAI_API_KEY=your_openai_key_here
```

### Model Configuration Overrides
```bash
# Override model settings
SCOTUS_MODEL_EMBEDDING_DIM=512
SCOTUS_MODEL_HIDDEN_DIM=1024
SCOTUS_MODEL_BATCH_SIZE=8

# Device configuration
SCOTUS_DEVICE=cuda
SCOTUS_NUM_WORKERS=4
```

### Path Overrides
```bash
# Override data paths
SCOTUS_DATA_RAW_DIR=/data/scotus/raw
SCOTUS_DATA_PROCESSED_DIR=/data/scotus/processed
SCOTUS_OUTPUT_DIR=/models/scotus
```

## 🚀 Usage Examples

### Loading Configuration
```python
from scripts.utils.config import Config

# Load default configuration
config = Config()

# Load custom configuration
config = Config("configs/custom_config.yaml")

# Access configuration values
embedding_dim = config.get('model.embedding_dim')
batch_size = config.get('training.batch_size', 4)  # with default
```

### Environment Integration
```python
import os

# Set environment variables
os.environ['SCOTUS_MODEL_BATCH_SIZE'] = '8'
os.environ['SCOTUS_TRAINING_LEARNING_RATE'] = '0.0002'

# Configuration automatically uses environment values
config = Config()
batch_size = config.get('training.batch_size')  # Returns 8
```

## 🔧 Customization

### Creating Custom Configurations

1. **Copy base configuration:**
   ```bash
   cp configs/base_config.yaml configs/my_config.yaml
   ```

2. **Modify settings:**
   ```yaml
   # Custom model architecture
   model:
     embedding_dim: 512
     hidden_dim: 1024
     num_attention_heads: 8
   
   # Aggressive training
   training:
     learning_rate: 0.001
     batch_size: 8
     num_epochs: 20
   ```

3. **Use custom configuration:**
   ```bash
   python scripts/models/model_trainer.py --config configs/my_config.yaml
   ```

### Environment-Specific Configurations

#### Development Configuration (`configs/dev_config.yaml`)
```yaml
training:
  batch_size: 2          # Small batch for fast iteration
  num_epochs: 5          # Quick training
  learning_rate: 0.001   # Higher LR for faster convergence

pipeline:
  verbose_output: true   # Detailed logging
  quick_mode: true       # Limited data processing
```

#### Production Configuration (`configs/prod_config.yaml`)
```yaml
training:
  batch_size: 8          # Larger batch for efficiency
  num_epochs: 50         # Thorough training
  learning_rate: 0.0001  # Conservative LR

pipeline:
  verbose_output: false  # Minimal logging
  interactive_mode: false # No user prompts
```

#### GPU Configuration (`configs/gpu_config.yaml`)
```yaml
model:
  device: "cuda"

training:
  batch_size: 16         # Large batch for GPU
  num_workers: 8         # Parallel data loading
  use_mixed_precision: true

tokenization:
  bio_batch_size: 64
  description_batch_size: 32
```

## 📊 Configuration Validation

### Required Settings
The system validates these required configuration keys:
- `data.raw_dir`
- `data.processed_dir`
- `model.embedding_dim`
- `model.bio_model_name`
- `model.description_model_name`
- `training.learning_rate`
- `training.batch_size`

### Validation Script
```python
from scripts.utils.config import Config

def validate_config(config_file=None):
    config = Config(config_file)
    
    required_keys = [
        'data.raw_dir',
        'model.embedding_dim',
        'training.learning_rate'
    ]
    
    for key in required_keys:
        value = config.get(key)
        if value is None:
            raise ValueError(f"Missing required configuration: {key}")
        print(f"✓ {key}: {value}")
    
    print("Configuration validation passed!")

# Validate default config
validate_config()
```

## 🔒 Security Considerations

### Sensitive Data
Never commit sensitive information to configuration files:
- API keys and tokens
- Database passwords
- Private URLs or endpoints

Use environment variables instead:
```bash
# .env file (not committed to git)
GEMMA_KEY=your_actual_api_key
DATABASE_PASSWORD=your_actual_password
```

### File Permissions
Ensure configuration files have appropriate permissions:
```bash
# Restrict access to configuration files
chmod 600 configs/*.yaml
```

## 📝 Configuration Reference

### Complete Configuration Schema

```yaml
# Data paths
data:
  raw_dir: string                    # Raw data directory
  processed_dir: string              # Processed data directory
  external_dir: string               # External data directory
  output_dir: string                 # Model output directory

# Model architecture
model:
  bio_model_name: string             # Biography encoder model
  description_model_name: string     # Case description encoder model
  embedding_dim: integer             # Embedding dimension
  hidden_dim: integer                # Hidden layer dimension
  dropout_rate: float                # Dropout rate
  max_justices: integer              # Maximum justices per case
  use_justice_attention: boolean     # Enable attention mechanism
  num_attention_heads: integer       # Number of attention heads
  device: string                     # Computing device (auto/cuda/cpu)

# Training parameters
training:
  learning_rate: float               # Optimizer learning rate
  weight_decay: float                # L2 regularization
  batch_size: integer                # Training batch size
  num_epochs: integer                # Number of training epochs
  patience: integer                  # Early stopping patience
  lr_scheduler_factor: float         # LR reduction factor
  lr_scheduler_patience: integer     # LR scheduler patience
  max_grad_norm: float              # Gradient clipping threshold
  loss_function: string             # Loss function type
  validation_frequency: integer      # Validation check frequency

# Pipeline configuration
pipeline:
  steps: list                       # Processing steps to execute
  interactive_mode: boolean         # Enable user interaction
  resume_enabled: boolean           # Enable resume functionality
  verbose_output: boolean           # Enable detailed output
  quick_mode: boolean              # Enable quick test mode

# Tokenization settings
tokenization:
  bio_batch_size: integer           # Biography encoding batch size
  description_batch_size: integer   # Description encoding batch size
  max_sequence_length: integer      # Maximum token sequence length
  show_progress: boolean            # Show progress bars
  clear_cache_on_oom: boolean      # Clear GPU cache on OOM

# Logging configuration
logging:
  level: string                     # Log level (DEBUG/INFO/WARNING/ERROR)
  file: string                      # Log file path
  max_size: string                  # Maximum log file size
  backup_count: integer             # Number of backup log files
```

## 🛠️ Troubleshooting

### Configuration Loading Issues
```python
# Debug configuration loading
from scripts.utils.config import Config

try:
    config = Config("configs/my_config.yaml")
    print("Configuration loaded successfully")
except Exception as e:
    print(f"Configuration loading failed: {e}")
```

### Environment Variable Issues
```python
import os

# Check environment variables
env_vars = {k: v for k, v in os.environ.items() if k.startswith('SCOTUS_')}
print("SCOTUS environment variables:")
for k, v in env_vars.items():
    print(f"  {k}: {v}")
```

### Configuration Validation
```bash
# Validate configuration syntax
python -c "
import yaml
with open('configs/base_config.yaml') as f:
    config = yaml.safe_load(f)
print('Configuration syntax is valid')
"
```

## 📞 Support

For configuration issues:
- Check YAML syntax and indentation
- Verify file paths and permissions
- Validate environment variable names
- Review configuration schema requirements

---

**Flexible configuration for optimal SCOTUS AI performance** ⚙️ 
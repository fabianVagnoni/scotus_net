# SCOTUS AI Utilities

The utilities module provides essential support components for the SCOTUS AI system, including configuration management, logging, progress tracking, and test set management.

## 🛠️ Module Components

### `config.py`
Central configuration management system:
- YAML configuration file loading
- Environment variable integration
- Hierarchical configuration access
- Type-safe configuration retrieval

### `logger.py`
Comprehensive logging system:
- Structured logging with multiple handlers
- File and console output
- Configurable log levels
- Timestamp and context information

### `progress.py`
Progress tracking and display:
- Cross-platform progress bars
- Fallback for environments without tqdm
- Consistent progress reporting
- Memory-efficient progress tracking

### `holdout_test_set.py`
Test set management for evaluation:
- Holdout test set creation and management
- Temporal-based case selection
- Data integrity validation
- Test set isolation for unbiased evaluation

## 🔧 Configuration Management

### YAML Configuration (`config.py`)

The configuration system supports hierarchical YAML configuration files:

```yaml
# configs/base_config.yaml
data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  external_dir: "data/external"

model:
  embedding_dim: 384
  hidden_dim: 512
  max_justices: 15

training:
  learning_rate: 0.0001
  batch_size: 4
  num_epochs: 10

pipeline:
  steps:
    - scrape_justices
    - scrape_bios
    - process_cases
```

### Usage Examples

```python
from scripts.utils.config import Config

# Load default configuration
config = Config()

# Access nested configuration
embedding_dim = config.get('model.embedding_dim', 384)
data_dir = config.get('data.processed_dir', 'data/processed')

# Update configuration
config.update('model.learning_rate', 0.0002)

# Save modified configuration
config.save('configs/updated_config.yaml')

# Load custom configuration file
custom_config = Config('configs/custom_config.yaml')
```

### Environment Variable Integration

```python
# Environment variables override YAML values
import os
os.environ['SCOTUS_MODEL_EMBEDDING_DIM'] = '512'

config = Config()
# Will use 512 from environment instead of YAML value
embedding_dim = config.get('model.embedding_dim')
```

## 📋 Logging System

### Logger Configuration (`logger.py`)

The logging system provides structured, multi-level logging:

```python
from scripts.utils.logger import get_logger

# Get logger for specific module
logger = get_logger('data_pipeline')

# Log different levels
logger.info("Starting data processing")
logger.warning("API quota approaching limit")
logger.error("Failed to process case", extra={'case_id': '123'})
logger.debug("Detailed processing information")
```

### Log Configuration

```python
from scripts.utils.logger import Logger

# Custom logger configuration
logger_config = Logger(
    log_file="logs/custom.log",
    log_level="DEBUG"
)

logger = logger_config.get_logger()
```

### Log Output Format

```
2024-01-15 10:30:45,123 - data_pipeline - INFO - Starting justice metadata scraping
2024-01-15 10:30:46,456 - data_pipeline - WARNING - Rate limit approaching: 90% used
2024-01-15 10:30:47,789 - data_pipeline - ERROR - Failed to process case_id: 2021_123
```

## 📊 Progress Tracking

### Progress Bar System (`progress.py`)

The progress system provides consistent progress tracking across all modules:

```python
from scripts.utils.progress import tqdm, get_progress_bar

# Standard usage
for item in tqdm(items, desc="Processing cases"):
    process_item(item)

# Custom progress bar
pbar = get_progress_bar(
    total=len(items),
    desc="Encoding biographies",
    unit="bio"
)

for item in items:
    process_item(item)
    pbar.update(1)
pbar.close()
```

### Fallback Support

The progress system gracefully handles environments without tqdm:

```python
# Automatically detects tqdm availability
from scripts.utils.progress import HAS_TQDM

if HAS_TQDM:
    print("Using enhanced progress bars")
else:
    print("Using basic progress indicators")
```

### Progress Configuration

```python
# Disable progress bars for quiet mode
progress_bar = get_progress_bar(
    items, 
    desc="Processing",
    disable=quiet_mode
)
```

## 🧪 Test Set Management

### Holdout Test Set (`holdout_test_set.py`)

The holdout test set manager ensures unbiased model evaluation:

```python
from scripts.utils.holdout_test_set import HoldoutTestSetManager

# Initialize manager
manager = HoldoutTestSetManager(
    dataset_file="data/processed/case_dataset.json",
    holdout_file="data/processed/holdout_test_set.json"
)

# Create holdout test set (15% of most recent cases)
holdout_case_ids = manager.create_holdout_test_set(percentage=0.15)

# Filter training dataset to exclude holdout cases
training_dataset = manager.filter_dataset_exclude_holdout(full_dataset)

# Get holdout dataset for evaluation
test_dataset = manager.get_holdout_dataset(full_dataset)
```

### Temporal Selection Strategy

The holdout system uses temporal-based selection to ensure realistic evaluation:

1. **Filter Complete Cases**: Only cases with descriptions and biographies
2. **Extract Case Years**: Determine case year from case ID
3. **Select Recent Cases**: Choose most recent 15% of cases
4. **Maintain Isolation**: Ensure test cases never appear in training

### Test Set Validation

```python
# Validate test set integrity
holdout_cases = manager.get_holdout_case_ids()
training_cases = set(training_dataset.keys())

# Ensure no overlap
assert len(holdout_cases.intersection(training_cases)) == 0

# Check temporal distribution
print(f"Holdout cases span: {min(case_years)} - {max(case_years)}")
```

## ⚙️ Configuration Examples

### Base Configuration (`configs/base_config.yaml`)

```yaml
# Data paths
data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  external_dir: "data/external"

# Model configuration
model:
  bio_model_name: "sentence-transformers/all-MiniLM-L6-v2"
  description_model_name: "Stern5497/sbert-legal-xlm-roberta-base"
  embedding_dim: 384
  hidden_dim: 512
  dropout_rate: 0.1
  max_justices: 15

# Training parameters
training:
  learning_rate: 0.0001
  batch_size: 4
  num_epochs: 10
  weight_decay: 0.01
  patience: 5

# Pipeline configuration
pipeline:
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

# Logging configuration
logging:
  level: "INFO"
  file: "logs/scotus_ai.log"
  max_size: "10MB"
  backup_count: 5
```

### Environment-Specific Overrides

```bash
# Environment variables for production
export SCOTUS_DATA_RAW_DIR="/data/production/raw"
export SCOTUS_MODEL_BATCH_SIZE="8"
export SCOTUS_LOGGING_LEVEL="WARNING"
```

## 🚀 Usage Patterns

### Configuration in Data Pipeline

```python
from scripts.utils.config import Config

config = Config()

# Get data paths
raw_dir = config.get('data.raw_dir')
processed_dir = config.get('data.processed_dir')

# Get pipeline settings
batch_size = config.get('training.batch_size', 4)
learning_rate = config.get('training.learning_rate', 0.0001)
```

### Logging in Modules

```python
from scripts.utils.logger import get_logger

logger = get_logger(__name__)

def process_cases():
    logger.info("Starting case processing")
    try:
        # Processing logic
        logger.info(f"Processed {count} cases successfully")
    except Exception as e:
        logger.error(f"Case processing failed: {e}")
        raise
```

### Progress Tracking in Loops

```python
from scripts.utils.progress import tqdm

def scrape_biographies(justices):
    logger.info(f"Scraping {len(justices)} justice biographies")
    
    for justice in tqdm(justices, desc="Scraping biographies"):
        try:
            biography = scrape_biography(justice)
            save_biography(biography)
        except Exception as e:
            logger.warning(f"Failed to scrape {justice['name']}: {e}")
```

### Test Set Management in Training

```python
from scripts.utils.holdout_test_set import HoldoutTestSetManager

# Setup test set manager
test_manager = HoldoutTestSetManager()

# Create or load holdout test set
holdout_cases = test_manager.create_holdout_test_set(percentage=0.15)

# Filter training data
training_data = test_manager.filter_dataset_exclude_holdout(full_dataset)

# Train model on filtered data
model = train_model(training_data)

# Evaluate on holdout set
test_data = test_manager.get_holdout_dataset(full_dataset)
evaluation_results = evaluate_model(model, test_data)
```

## 📁 Directory Structure

```
scripts/utils/
├── __init__.py                 # Module initialization
├── config.py                   # Configuration management
├── logger.py                   # Logging system
├── progress.py                 # Progress tracking
└── holdout_test_set.py        # Test set management
```

## 🔧 Advanced Features

### Configuration Validation

```python
from scripts.utils.config import Config

config = Config()

# Validate required configuration
required_keys = [
    'data.raw_dir',
    'model.embedding_dim',
    'training.learning_rate'
]

for key in required_keys:
    if config.get(key) is None:
        raise ValueError(f"Required configuration missing: {key}")
```

### Dynamic Configuration Updates

```python
# Update configuration at runtime
config.update('model.batch_size', 8)
config.update('training.learning_rate', 0.0002)

# Save updated configuration
config.save('configs/runtime_config.yaml')
```

### Structured Logging

```python
from scripts.utils.logger import get_logger

logger = get_logger('model_training')

# Structured logging with context
logger.info(
    "Model training completed",
    extra={
        'epoch': 10,
        'train_loss': 0.342,
        'val_loss': 0.298,
        'duration': 1800  # seconds
    }
)
```

## 🚨 Important Considerations

### Configuration Security

- **Sensitive Data**: Never commit API keys or credentials to YAML files
- **Environment Variables**: Use environment variables for sensitive configuration
- **File Permissions**: Ensure configuration files have appropriate permissions

### Logging Best Practices

- **Log Levels**: Use appropriate log levels (DEBUG, INFO, WARNING, ERROR)
- **Context**: Include relevant context in log messages
- **Performance**: Avoid excessive logging in tight loops
- **Sensitive Data**: Never log sensitive information

### Test Set Integrity

- **Temporal Validation**: Ensure test cases are chronologically separated
- **No Data Leakage**: Validate no test cases appear in training data
- **Reproducibility**: Use consistent random seeds for test set creation

## 🛠️ Troubleshooting

### Configuration Issues

```python
# Debug configuration loading
config = Config()
config.print_config()  # Print all configuration values

# Check specific values
print(f"Data dir: {config.get('data.raw_dir')}")
print(f"Model dim: {config.get('model.embedding_dim')}")
```

### Logging Issues

```python
# Test logger functionality
from scripts.utils.logger import get_logger

logger = get_logger('test')
logger.info("Test log message")

# Check log file permissions
import os
log_file = "logs/scotus_ai.log"
if os.path.exists(log_file):
    print(f"Log file exists: {log_file}")
else:
    print(f"Log file missing: {log_file}")
```

### Progress Bar Issues

```python
# Test progress bar functionality
from scripts.utils.progress import tqdm, HAS_TQDM

print(f"tqdm available: {HAS_TQDM}")

# Test basic progress bar
for i in tqdm(range(10), desc="Testing"):
    time.sleep(0.1)
```

## 📞 Support

For utilities issues:
- Check configuration file syntax and paths
- Verify log file permissions and disk space
- Test progress bar functionality in your environment
- Validate test set temporal distribution

---

**Essential utilities for robust SCOTUS AI development** 🔧 
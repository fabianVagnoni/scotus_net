# Contrastive Justice Pretraining

This module implements contrastive learning for pretraining justice biography encoders. The goal is to train the truncated biography encoder to produce similar embeddings to the full biography encoder for the same justice, while producing different embeddings for different justices.

## 🎯 Overview

The contrastive justice pretraining uses a teacher-student architecture where:
- **Teacher Model**: Full biography encoder (frozen)
- **Student Model**: Truncated biography encoder (trainable)
- **Objective**: Learn similar embeddings for the same justice across truncated and full biographies
- **Temporal Splits**: Uses appointment dates to create time-based train/validation splits

## 🏗️ Architecture

### Model Components

1. **ContrastiveJustice**: Simple model that produces embeddings from truncated and full biographies
2. **ContrastiveLoss**: Handles contrastive learning logic (NT-Xent + MSE)
3. **ContrastiveJusticeTrainer**: Manages training loop and data processing

### Data Flow

```
Truncated Bio → Truncated Encoder → Embeddings
                                    ↓
Full Bio → Full Encoder → Embeddings → Contrastive Loss
```

## 📁 File Structure

```
scripts/pretraining/
├── constrastive_justice.py          # Main model implementation
├── contrastive_trainer.py           # Training logic
├── loss.py                          # Contrastive loss implementation
├── config.py                        # Configuration management
├── config.env                       # Configuration file
├── run_training.py                  # Example training script
└── README.md                        # This documentation
```

## ⚙️ Configuration

The system uses a `config.env` file for all hyperparameters. You can modify this file to customize the training:

```bash
# Model Configuration
MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
DROPOUT_RATE=0.1

# Training Configuration
BATCH_SIZE=8
LEARNING_RATE=1e-5
NUM_EPOCHS=10
WEIGHT_DECAY=0.01

# Loss Configuration
TEMPERATURE=0.1
ALPHA=0.5

# Data Configuration
VAL_SPLIT=0.2
JUSTICES_FILE=data/raw/justices.json
TRUNC_BIO_TOKENIZED_FILE=data/processed/encoded_bios.pkl
FULL_BIO_TOKENIZED_FILE=data/raw/encoded_bios.pkl
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace model for encoding |
| `BATCH_SIZE` | `8` | Training batch size |
| `LEARNING_RATE` | `1e-5` | Learning rate for optimization |
| `NUM_EPOCHS` | `10` | Number of training epochs |
| `TEMPERATURE` | `0.1` | Temperature for contrastive loss |
| `ALPHA` | `0.5` | Weight for NT-Xent vs MSE loss |
| `JUSTICES_FILE` | `data/raw/justices.json` | Justice information with appointment dates |

## 🚀 Usage

### Basic Training

```bash
# Run training with default configuration
python scripts/pretraining/run_training.py
```

### Custom Configuration

1. **Modify config.env**:
```bash
# Edit the configuration file
BATCH_SIZE=16
LEARNING_RATE=2e-5
NUM_EPOCHS=20
```

2. **Use custom config file**:
```python
from config import ContrastiveJusticeConfig
from contrastive_trainer import ContrastiveJusticeTrainer

# Load custom configuration
config = ContrastiveJusticeConfig("path/to/custom_config.env")
trainer = ContrastiveJusticeTrainer(config)
trained_model = trainer.train_model(config.justices_file)
```

### Programmatic Usage

```python
from config import ContrastiveJusticeConfig
from contrastive_trainer import ContrastiveJusticeTrainer

# Create configuration
config = ContrastiveJusticeConfig()

# Customize specific parameters
config.batch_size = 16
config.learning_rate = 2e-5

# Create trainer and run training
trainer = ContrastiveJusticeTrainer(config)
trained_model = trainer.train_model(config.justices_file)
```

## 📊 Data Requirements

### Input Data Structure

The training script expects:

1. **Justices Data**: `data/raw/justices.json` (justice information with appointment dates)
2. **Truncated Biographies**: `data/processed/encoded_bios.pkl` (tokenized)
3. **Full Biographies**: `data/raw/encoded_bios.pkl` (tokenized)

### Data Format

```
data/
├── raw/
│   ├── justices.json                # Justice information with appointment dates
│   └── encoded_bios.pkl             # Tokenized full biographies
└── processed/
    └── encoded_bios.pkl             # Tokenized truncated biographies
```

### Justice Data Format

The `justices.json` file contains justice information with appointment dates:

```json
{
  "John Jay": {
    "name": "John Jay",
    "appointment_date": "September 26, 1789 ( Acclamation )",
    "tenure_start": "October 19, 1789",
    "tenure_end": "June 29, 1795",
    ...
  }
}
```

## 🔧 Model Implementation

### ContrastiveJustice Class

```python
class ContrastiveJustice(nn.Module):
    def __init__(self, trunc_bio_tokenized_file, full_bio_tokenized_file, 
                 model_name, dropout_rate=0.1):
        # Initialize teacher and student models
        # Load tokenized data
```

### Key Methods

- `forward()`: Generate embeddings for truncated and full biographies
- `get_tokenized_stats()`: Get statistics about loaded data

### ContrastiveLoss Class

```python
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5, alpha=0.5):
        # Initialize NT-Xent and MSE loss components
```

The loss combines:
- **NT-Xent Loss**: Contrastive learning for similarity
- **MSE Loss**: Direct similarity between embeddings
- **Combined**: `α * NT-Xent + (1-α) * MSE`

## 📈 Training Process

### Temporal Splits

The training uses **temporal splits** based on appointment dates:
- **Train Set**: Earlier appointed justices (e.g., 1789-1950)
- **Validation Set**: Later appointed justices (e.g., 1950-present)
- **Rationale**: Simulates real-world scenario where we train on historical data and validate on newer justices

### Training Loop

1. **Data Loading**: Load justices data and encoded biographies
2. **Temporal Splitting**: Split justices by appointment date
3. **Forward Pass**: Generate embeddings for both truncated and full bios
4. **Loss Computation**: Calculate contrastive loss
5. **Backward Pass**: Update truncated bio encoder parameters
6. **Validation**: Monitor loss on validation set

### Monitoring

The training provides:
- Real-time loss tracking with progress bars
- Validation loss monitoring
- Learning rate scheduling
- Early stopping
- Checkpoint saving

## 💾 Model Outputs

### Saved Files

```
models/contrastive_justice/
├── best_model.pth                   # Best model based on validation loss
├── final_model.pth                  # Final trained model
└── training.log                     # Training log file
```

### Model Checkpoint Contents

```python
checkpoint = {
    'epoch': 5,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': 0.123,
    'val_loss': 0.098,
    'train_losses': [0.5, 0.3, 0.2, 0.15, 0.123],
    'val_losses': [0.4, 0.25, 0.18, 0.12, 0.098],
    'config': {...}
}
```

## 🔄 Integration with Main Model

### Loading Pretrained Encoder

After training, you can load the pretrained truncated bio encoder into the main SCOTUS voting model:

```python
# Load pretrained contrastive model
checkpoint = torch.load('models/contrastive_justice/best_model.pth')
contrastive_model = ContrastiveJustice(...)
contrastive_model.load_state_dict(checkpoint['model_state_dict'])

# Extract truncated bio encoder for main model
pretrained_trunc_encoder = contrastive_model.truncated_bio_model
```

### Fine-tuning Strategy

1. **Pretrain**: Train contrastive model on biography pairs
2. **Transfer**: Load pretrained encoder into main model
3. **Fine-tune**: Continue training on voting prediction task

## 🧪 Evaluation

### Metrics

- **Contrastive Loss**: Measures similarity learning quality
- **Embedding Similarity**: Cosine similarity between truncated and full embeddings
- **Justice Discrimination**: Ability to distinguish between different justices

### Validation

The model is evaluated on a held-out validation set to ensure:
- Good generalization to unseen justices
- Proper similarity learning without overfitting
- Consistent embedding quality

## 🐛 Troubleshooting

### Common Issues

1. **Missing Config File**: System will use defaults if `config.env` not found
2. **Invalid Config Values**: Invalid values will be replaced with defaults
3. **Missing Tokenized Files**: Run tokenization first
4. **Memory Issues**: Reduce batch size in config

### Debugging Tips

- Check configuration with `config.print_config()`
- Use smaller batch sizes for debugging
- Monitor GPU memory usage
- Check data loading with validation split

## 📚 References

- **InfoNCE Loss**: [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748)
- **Contrastive Learning**: [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
- **Teacher-Student Learning**: [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) 
# SCOTUS AI Models

The models module contains the machine learning components for predicting Supreme Court case outcomes. It includes neural network architectures, training pipelines, hyperparameter optimization, and evaluation tools.

## 🧠 Model Architecture

### SCOTUS Voting Model

The core model (`scotus_voting_model.py`) is a neural network that predicts voting probability distributions for Supreme Court cases:

```
Justice Biographies + Case Description → Voting Probabilities
        ↓                  ↓                      ↓
   [Bio Encoder]      [Case Encoder]         [Output Layer]
        ↓                  ↓                      ↓
   [Embeddings]       [Embeddings]           [Probabilities]
        ↓                  ↓                      ↓
        └──────→ [Cross-Attention] ←──────────────┘
                        ↓
                [Prediction Head]
                        ↓
              [in_favor, against, absent, other]
```

### Key Components

1. **Text Encoders**:
   - **Biography Encoder**: `sentence-transformers/all-MiniLM-L6-v2`
   - **Case Encoder**: `Stern5497/sbert-legal-xlm-roberta-base` (legal-specialized)

2. **Cross-Attention Mechanism** (`justice_cross_attention.py`):
   - Multi-head attention between justice biographies and case descriptions
   - Captures justice-case interactions and dependencies
   - Configurable attention heads and dimensions

3. **Prediction Head**:
   - Multi-layer neural network
   - Dropout and regularization
   - Outputs 4-dimensional probability distribution

## 🏗️ Model Components

### `scotus_voting_model.py`
Main model class with end-to-end prediction capability:
- Pre-tokenized data loading
- Text encoding with sentence transformers
- Cross-attention computation
- Voting prediction output

### `justice_cross_attention.py`
Attention mechanism implementation:
- Multi-head cross-attention
- Justice-case interaction modeling
- Configurable attention parameters

### `model_trainer.py`
Training pipeline with:
- Dataset loading and splitting
- Training loop with validation
- Early stopping and learning rate scheduling
- Model evaluation and saving

### `hyperparameter_optimization.py`
Automated hyperparameter tuning:
- Optuna-based optimization
- Comprehensive parameter search
- Multi-objective optimization
- Best configuration saving

## 🚀 Usage

### Training a Model

```bash
# Basic training with default configuration
python scripts/models/model_trainer.py

# Training with custom dataset
python scripts/models/model_trainer.py --dataset data/processed/case_dataset.json
```

### Hyperparameter Optimization

```bash
# Run optimization with 50 trials
python scripts/models/hyperparameter_optimization.py --n-trials 50

# Distributed optimization with storage
python scripts/models/hyperparameter_optimization.py \
    --n-trials 100 \
    --storage sqlite:///optuna_study.db \
    --study-name scotus_optimization
```

### Model Prediction

```python
from scripts.models.scotus_voting_model import SCOTUSVotingModel

# Load trained model
model = SCOTUSVotingModel.load_model(
    "models_output/best_model.pth",
    bio_tokenized_file="data/processed/encoded_bios.pkl",
    description_tokenized_file="data/processed/encoded_descriptions.pkl",
    bio_model_name="sentence-transformers/all-MiniLM-L6-v2",
    description_model_name="Stern5497/sbert-legal-xlm-roberta-base"
)

# Make prediction
prediction = model.predict_from_files(
    case_description_path="data/processed/case_descriptions/case_123.txt",
    justice_bio_paths=[
        "data/processed/bios/John_Roberts.txt",
        "data/processed/bios/Clarence_Thomas.txt"
    ]
)

print(f"Voting probabilities: {prediction}")
# Output: [0.65, 0.25, 0.10, 0.00]  # [in_favor, against, absent, other]
```

## ⚙️ Configuration

### Model Configuration (`config.py`)

```python
# Model Architecture
BIO_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
DESCRIPTION_MODEL_NAME = 'Stern5497/sbert-legal-xlm-roberta-base'
EMBEDDING_DIM = 384
HIDDEN_DIM = 512
MAX_JUSTICES = 15

# Attention Mechanism
USE_JUSTICE_ATTENTION = True
NUM_ATTENTION_HEADS = 4

# Training Parameters
LEARNING_RATE = 0.0001
NUM_EPOCHS = 10
BATCH_SIZE = 4
DROPOUT_RATE = 0.1
WEIGHT_DECAY = 0.01
```

### Environment Configuration (`config.env`)

```bash
# Model Configuration
BIO_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
DESCRIPTION_MODEL_NAME=Stern5497/sbert-legal-xlm-roberta-base
EMBEDDING_DIM=384
HIDDEN_DIM=512

# Training Configuration
LEARNING_RATE=0.0001
BATCH_SIZE=4
NUM_EPOCHS=10
DEVICE=auto

# Data Paths
DATASET_FILE=data/processed/case_dataset.json
BIO_TOKENIZED_FILE=data/processed/encoded_bios.pkl
DESCRIPTION_TOKENIZED_FILE=data/processed/encoded_descriptions.pkl
```

## 📊 Model Performance

### Loss Functions

The model supports multiple loss functions:

1. **KL Divergence** (default):
   - Measures difference between predicted and actual probability distributions
   - Best for probability distribution outputs
   - Handles uncertainty in voting patterns

2. **Mean Squared Error**:
   - Regression-style loss for continuous prediction
   - Good for numerical voting percentages

3. **Cross Entropy**:
   - Classification loss for discrete outcomes
   - Suitable for majority vote prediction

### Evaluation Metrics

- **KL Divergence Loss**: Primary training and validation metric
- **MSE**: Alternative regression metric
- **Accuracy**: Classification accuracy for majority predictions
- **F1 Score**: Balanced precision/recall for voting outcomes

### Performance Benchmarks

```python
# Example evaluation results
{
    'kl_divergence': 0.342,
    'mse': 0.089,
    'accuracy': 0.724,
    'f1_score': 0.691
}
```

## 🔧 Advanced Features

### Cross-Attention Mechanism

The justice cross-attention mechanism captures interactions between:
- Justice biographical information
- Case-specific legal content
- Historical voting patterns
- Justice ideological positions

```python
# Attention computation
attention_scores = model.justice_attention(
    case_embedding,      # Case description encoding
    justice_embeddings,  # Multiple justice biography encodings
    justice_mask        # Mask for variable number of justices
)
```

### Batch Prediction

```python
# Predict on multiple cases
dataset_entries = [
    (justice_bio_paths, case_description_path, target_votes),
    # ... more cases
]

predictions = model.predict_batch_from_dataset(
    dataset_entries,
    return_probabilities=True
)
```

### Model Introspection

```python
# Get model statistics
stats = model.get_tokenized_stats()
print(f"Encoded biographies: {stats['bio_count']}")
print(f"Encoded descriptions: {stats['description_count']}")

# Get available data paths
paths = model.get_available_paths()
print(f"Available bios: {len(paths['bio_paths'])}")
print(f"Available descriptions: {len(paths['description_paths'])}")
```

## 🎯 Hyperparameter Optimization

### Optimization Parameters

The hyperparameter optimization searches over:

- **Learning Rate**: 1e-5 to 1e-2
- **Hidden Dimensions**: 256 to 1024
- **Attention Heads**: 2 to 8
- **Dropout Rate**: 0.0 to 0.5
- **Batch Size**: 2 to 16
- **Weight Decay**: 1e-5 to 1e-1

### Optimization Strategy

```python
# Multi-objective optimization
def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    hidden_dim = trial.suggest_int('hidden_dim', 256, 1024, step=128)
    
    # Train model with suggested parameters
    model = train_model_with_params(lr, hidden_dim, ...)
    
    # Return validation loss
    return validation_loss
```

### Best Configuration Export

```python
# Save best hyperparameters
best_params = study.best_params
save_best_config(study, "best_config.env")
```

## 🧪 Testing and Evaluation

### Holdout Test Set

```python
from scripts.utils.holdout_test_set import HoldoutTestSetManager

# Create holdout test set (15% of most recent cases)
manager = HoldoutTestSetManager()
holdout_cases = manager.create_holdout_test_set(percentage=0.15)

# Evaluate on holdout set
trainer = SCOTUSModelTrainer()
results = trainer.evaluate_on_holdout_test_set("models_output/best_model.pth")
```

### Cross-Validation

```python
# Split dataset for training/validation
train_data, val_data = trainer.split_dataset(
    dataset, 
    train_ratio=0.85, 
    val_ratio=0.15
)
```

### Model Evaluation

```python
# Comprehensive evaluation
evaluation_results = trainer.evaluate_model(
    model, 
    test_dataloader, 
    criterion
)

print(f"Test Loss: {evaluation_results['loss']:.4f}")
print(f"Test Accuracy: {evaluation_results['accuracy']:.4f}")
```

## 📁 Model Outputs

### Saved Models

```
models_output/
├── best_model.pth              # Best model checkpoint
├── model_config.json           # Model configuration
├── training_history.json       # Training metrics history
├── hyperparameter_study.db     # Optuna optimization results
└── evaluation_results.json     # Test set evaluation
```

### Model Checkpoints

```python
# Model checkpoint structure
{
    'model_state_dict': model.state_dict(),
    'config': model_config,
    'epoch': epoch,
    'train_loss': train_loss,
    'val_loss': val_loss,
    'hyperparameters': hyperparams
}
```

## 🚨 Important Considerations

### Data Leakage Prevention

- **Pre-decision Content**: Only uses information available before the court decision
- **Temporal Validation**: Validates chronological data splits
- **Content Filtering**: AI-filtered case descriptions exclude outcomes

### Model Interpretability

- **Attention Weights**: Visualize justice-case attention patterns
- **Feature Importance**: Analyze which biographical factors matter most
- **Prediction Confidence**: Output probability distributions for uncertainty

### Computational Requirements

- **GPU Recommended**: CUDA-compatible GPU for efficient training
- **Memory**: 16GB+ RAM for full dataset processing
- **Storage**: 5GB+ for model checkpoints and embeddings

## 🛠️ Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| CUDA out of memory | Batch size too large | Reduce `BATCH_SIZE` in config |
| Model not converging | Learning rate too high | Lower `LEARNING_RATE` |
| Poor validation performance | Overfitting | Increase `DROPOUT_RATE` |
| Slow training | CPU-only training | Enable GPU with `DEVICE=cuda` |

### Debug Mode

```bash
# Enable verbose training logs
python scripts/models/model_trainer.py --verbose

# Test model loading
python -c "from scripts.models.scotus_voting_model import SCOTUSVotingModel; print('Model loads successfully')"
```

### Memory Optimization

```python
# Clear CUDA cache on out-of-memory
import torch
torch.cuda.empty_cache()

# Use gradient checkpointing for large models
model.gradient_checkpointing_enable()
```

## 📞 Support

For model-related issues:
- Check CUDA compatibility and GPU availability
- Review tokenization requirements (run tokenization pipeline first)
- Validate configuration parameters in `config.env`
- Monitor training logs for convergence issues

---

**Ready to train your SCOTUS prediction model?** Start with `python scripts/models/model_trainer.py` 🤖 
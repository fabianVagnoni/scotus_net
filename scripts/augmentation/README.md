# SCOTUS AI Augmentation System

This module provides text augmentation capabilities for the SCOTUS AI dataset. It creates augmented versions of justice biographies and case descriptions to expand the training dataset.

## Overview

The augmentation system takes the processed dataset from `data/processed/` and creates an augmented dataset in `data/augmented/` with multiple versions of each text (original + augmented versions).

## Directory Structure

```
augmentation/
├── __init__.py                           # Module initialization
├── main.py                              # Main augmentation pipeline orchestrator
├── justice_bios_augmentation.py         # Justice biographies augmentation
├── case_descriptions_augmentation.py    # Case descriptions augmentation
└── README.md                            # This file

data/
├── processed/                           # Original processed data
│   ├── case_dataset.json               # Original case dataset
│   ├── bios/                           # Original justice bios
│   └── case_descriptions/              # Original case descriptions
└── augmented/                          # Augmented data (created by this system)
    ├── case_dataset.json               # Augmented case dataset
    ├── bios/                           # Augmented justice bios
    └── case_descriptions/              # Augmented case descriptions
```

## Pipeline Steps

1. **Load Original Dataset**: Load `data/processed/case_dataset.json`
2. **Augment Justice Biographies**: Create multiple versions of each justice bio
3. **Augment Case Descriptions**: Create multiple versions of each case description
4. **Build Augmented Dataset**: Create new dataset with file paths to augmented versions
5. **Validate**: Ensure all files exist and dataset is consistent

## Usage

### Command Line Interface

```bash
# Run full augmentation pipeline
python augmentation/main.py

# Run only justice biographies
python augmentation/main.py --bios-only

# Run only case descriptions  
python augmentation/main.py --descriptions-only

# Copy originals only (no text augmentation)
python augmentation/main.py --no-augmentation

# Custom augmentation settings
python augmentation/main.py --augmentation-iterations 5 --augmentations word_embedding_augmentation synonym_augmentation
```

### Docker Interface

```bash
# Run full augmentation pipeline
./docker-run.sh augmentation

# Run only justice bios
./docker-run.sh augmentation --bios-only

# Run only case descriptions
./docker-run.sh augmentation --descriptions-only

# Run without text augmentation (copy originals only)
./docker-run.sh augmentation --no-augmentation

# Custom settings
./docker-run.sh augmentation --augmentation-iterations 5
```

## Augmentation Techniques

The system supports the following text augmentation techniques:

- **word_embedding_augmentation**: Replace words with similar embeddings
- **synonym_augmentation**: Replace words with synonyms
- **back_translation**: Translate to another language and back
- **summarization**: Create summary versions

## Random Selection Feature

To introduce more diversity and uncertainty in the augmented versions, the system uses a **random selection mechanism**:

- **Per-iteration selection**: In each iteration, each augmentation technique has a configurable probability of being selected
- **Random order**: Selected augmentations are applied in random order
- **Cascading effect**: Each augmentation is applied to the result of the previous one
- **Configurable probability**: Default is 50% (0.5), can be adjusted from 0.0 to 1.0

**Example**: With 4 techniques and 50% selection probability:
- Iteration 1: Might select `[word_embedding_augmentation, synonym_augmentation]`
- Iteration 2: Might select `[back_translation]` only
- Iteration 3: Might select `[word_embedding_augmentation, synonym_augmentation, summarization]`

This creates more diverse and unpredictable augmented versions compared to applying all techniques every time.

## Configuration

### Default Settings

- **Augmentation iterations**: 3 versions per text
- **Random seed**: 42 (for reproducibility)
- **Random selection probability**: 0.5 (50% chance of selecting each augmentation per iteration)
- **Techniques**: word_embedding_augmentation
- **Input**: `data/processed/case_dataset.json`
- **Output**: `data/augmented/case_dataset.json`

### Custom Configuration

You can customize the augmentation settings via command line arguments:

```bash
python augmentation/main.py \
    --augmentation-iterations 5 \
    --augmentations word_embedding_augmentation synonym_augmentation \
    --augmentation-seed 123 \
    --random-selection-prob 0.7 \
    --input-dataset data/processed/my_dataset.json \
    --output-dataset data/augmented/my_augmented_dataset.json
```

## Output Format

### Augmented Files

Each original file gets multiple versions:
- `justice_name_v0.txt` - Original version
- `justice_name_v1.txt` - First augmented version
- `justice_name_v2.txt` - Second augmented version
- etc.

### Augmented Dataset

The augmented dataset follows the same format as the original:
```json
{
  "case_id": [
    ["data/augmented/bios/justice_v0.txt", "data/augmented/bios/justice_v1.txt", ...],
    "data/augmented/case_descriptions/case_v0.txt",
    [pct_in_favor, pct_against, pct_absent]
  ]
}
```

## Integration with Training Pipeline

After creating the augmented dataset, you can use it with the existing training pipeline:

### For Tokenization

```bash
# Use augmented dataset for tokenization
python scripts/tokenization/main_encoder.py --dataset data/augmented/case_dataset.json
```

### For Training

```bash
# Set environment variable to use augmented dataset
export DATASET_FILE=data/augmented/case_dataset.json

# Run training
python scripts/models/run_training.py --experiment-name augmented_training
```

### For Hyperparameter Optimization

```bash
# Use augmented dataset for hyperparameter tuning
export DATASET_FILE=data/augmented/case_dataset.json

python scripts/models/hyperparameter_optimization.py --experiment-name augmented_tuning
```

## File Naming Convention

### Justice Biographies
- Original: `Justice_Name.txt`
- Augmented: `Justice_Name_v0.txt`, `Justice_Name_v1.txt`, etc.

### Case Descriptions  
- Original: `329_U.S._1.txt`
- Augmented: `329_U.S._1_v0.txt`, `329_U.S._1_v1.txt`, etc.

## Requirements

- Python 3.8+
- Required packages: `nlpaug`, `transformers`, `sentence-transformers`
- Original processed dataset must exist in `data/processed/`

## Error Handling

The system includes robust error handling:
- Graceful failure if augmentation techniques are unavailable
- Continues processing even if individual files fail
- Detailed logging and progress tracking
- Validation of final output

## Performance Considerations

- **Memory**: Large datasets may require significant memory
- **Time**: Text augmentation can be computationally expensive
- **Storage**: Augmented datasets can be 3-5x larger than originals
- **Resume**: The system can resume from interruptions

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure `scripts/data_pipeline/augmenter.py` exists
2. **Memory issues**: Reduce augmentation iterations or use smaller batches
3. **File permissions**: Ensure write access to `data/augmented/` directory
4. **Missing dependencies**: Install required packages with `pip install nlpaug transformers sentence-transformers`

### Debug Mode

Enable verbose output for debugging:
```bash
python augmentation/main.py --verbose
```

## Examples

### Quick Test (No Augmentation)

```bash
# Copy originals only to test the pipeline
python augmentation/main.py --no-augmentation --verbose
```

### Full Augmentation

```bash
# Run complete augmentation with default settings
python augmentation/main.py --verbose
```

### Custom Augmentation

```bash
# Use multiple techniques with more iterations and higher selection probability
python augmentation/main.py \
    --augmentations word_embedding_augmentation synonym_augmentation \
    --augmentation-iterations 5 \
    --random-selection-prob 0.7 \
    --verbose
```

## Contributing

To add new augmentation techniques:

1. Extend the `Augmenter` class in `scripts/data_pipeline/augmenter.py`
2. Add the technique to the choices in the argument parser
3. Update this README with the new technique description
4. Test with a small dataset first 
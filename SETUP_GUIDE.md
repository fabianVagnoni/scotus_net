# SCOTUS AI Project Setup Guide

This guide will help you set up the SCOTUS AI project for predicting Supreme Court case outcomes.

## Prerequisites

- Python 3.8 or higher
- Git (for version control)
- At least 4GB of free disk space

## Project Structure

```
scotus_ai/
├── data/                     # Data storage
│   ├── raw/                 # Raw scraped data
│   ├── processed/           # Cleaned and processed data
│   └── external/            # External datasets
├── src/                     # Source code
│   ├── data_pipeline/       # Data extraction and processing
│   │   ├── scraper.py      # Web scraper for SCOTUS data
│   │   └── processor.py    # Data cleaning and feature extraction
│   ├── models/              # ML models and training
│   │   └── model_trainer.py # Model training pipeline
│   └── utils/               # Utility functions
│       ├── config.py       # Configuration management
│       └── logger.py       # Logging utilities
├── configs/                 # Configuration files
│   └── base_config.yaml    # Base configuration
├── notebooks/               # Jupyter notebooks for exploration
├── logs/                    # Log files
├── models_output/           # Trained model artifacts
├── requirements.txt         # Python dependencies
├── setup.py                # Package setup
└── README.md               # Project documentation
```

## Manual Setup Instructions

### Step 1: Install Python

1. Download Python 3.8+ from [python.org](https://python.org)
2. During installation, make sure to check "Add Python to PATH"
3. Verify installation by opening Command Prompt/PowerShell and running:
   ```
   python --version
   ```

### Step 2: Clone and Navigate to Project

```bash
git clone <your-repo-url>
cd scotus_ai
```

### Step 3: Create Virtual Environment

**Windows (Command Prompt/PowerShell):**
```cmd
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Install spaCy English model
python -m spacy download en_core_web_sm
```

### Step 5: Configure Environment

1. Copy `env.example` to `.env`:
   ```bash
   cp env.example .env
   ```

2. Edit `.env` file with your configuration:
   - Add API keys if you have them
   - Adjust scraping settings
   - Configure database URLs if needed

### Step 6: Test Installation

Run a simple test to verify everything is working:

```python
python -c "
import sys
print('Python version:', sys.version)
import torch
print('PyTorch version:', torch.__version__)
import transformers
print('Transformers version:', transformers.__version__)
print('✅ All dependencies installed successfully!')
"
```

## Usage

### 1. Data Collection

Scrape SCOTUS case data:

```bash
python src/data_pipeline/scraper.py
```

This will:
- Scrape cases from Justia and CourtListener
- Save raw data to `data/raw/`
- Take several hours for full dataset

### 2. Data Processing

Process and clean the scraped data:

```bash
python src/data_pipeline/processor.py
```

This will:
- Clean and preprocess text
- Extract features
- Create train/validation/test splits
- Save processed data to `data/processed/`

### 3. Model Training

Train the prediction model:

```bash
python src/models/model_trainer.py
```

This will:
- Load processed data
- Fine-tune a transformer model (DistilBERT by default)
- Save trained model to `models_output/`

### 4. Explore with Jupyter

Start Jupyter notebook for data exploration:

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

## Configuration

The project uses YAML configuration files in the `configs/` directory. Key settings:

- **Model settings**: Model type, hyperparameters
- **Data settings**: Paths, split ratios
- **Training settings**: Batch size, epochs, learning rate
- **Pipeline settings**: Scraping delays, preprocessing options

## Troubleshooting

### Common Issues

1. **Python not found**: Make sure Python is in your PATH
2. **Permission errors**: Run as administrator (Windows) or use `sudo` (Linux/Mac)
3. **Memory errors**: Reduce batch size in config
4. **Network timeouts**: Increase scraping delays

### Getting Help

1. Check the logs in `logs/` directory
2. Review configuration in `configs/base_config.yaml`
3. Ensure all dependencies are installed correctly

## Development

### Adding New Features

1. Create feature branch: `git checkout -b feature-name`
2. Add your code following the existing structure
3. Update tests and documentation
4. Submit pull request

### Code Style

- Follow PEP 8 guidelines
- Use type hints where possible
- Add docstrings to functions and classes
- Run `black` for code formatting

## Next Steps

1. **Scale up data collection**: Collect more historical cases
2. **Improve features**: Add more sophisticated legal text analysis
3. **Model optimization**: Experiment with different architectures
4. **Deployment**: Create web API for predictions
5. **Evaluation**: Add comprehensive model evaluation metrics

## Resources

- [Supreme Court Database](http://scdb.wustl.edu/)
- [CourtListener API](https://www.courtlistener.com/api/)
- [Justia Supreme Court Cases](https://supreme.justia.com/cases/federal/us/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)

## License

MIT License - see LICENSE file for details. 
# SCOTUS AI Prediction Project

An AI-powered system to predict Supreme Court of the United States (SCOTUS) case outcomes using machine learning and natural language processing.

## Project Overview

This project aims to:
- Extract and process SCOTUS case data from various sources
- Fine-tune language models to understand legal text and case contexts
- Predict whether cases will be approved or denied by the Court
- Provide insights into factors influencing Court decisions

## Project Structure

```
scotus_ai/
├── data/                     # Data storage
│   ├── raw/                 # Raw scraped data
│   ├── processed/           # Cleaned and processed data
│   └── external/            # External datasets
├── src/                     # Source code
│   ├── data_pipeline/       # Data extraction and processing
│   ├── models/              # ML models and training
│   ├── features/            # Feature engineering
│   └── utils/               # Utility functions
├── notebooks/               # Jupyter notebooks for exploration
├── configs/                 # Configuration files
├── tests/                   # Unit tests
├── scripts/                 # Standalone scripts
├── logs/                    # Log files
└── models_output/           # Trained model artifacts
```

## Setup Instructions

### 1. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Language Models

```bash
# Download spaCy English model
python -m spacy download en_core_web_sm
```

### 4. Environment Configuration

Copy `.env.example` to `.env` and fill in your configuration:

```bash
cp .env.example .env
```

## Usage

### Data Pipeline

```bash
# Run data extraction
python src/data_pipeline/scraper.py

# Process raw data
python src/data_pipeline/processor.py
```

### Model Training

```bash
# Train the model
python src/models/train.py --config configs/base_config.yaml
```

## Contributing

1. Follow PEP 8 style guidelines
2. Write tests for new functionality
3. Update documentation as needed

## License

MIT License - see LICENSE file for details 
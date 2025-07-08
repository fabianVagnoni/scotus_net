# SCOTUS AI: Supreme Court Case Outcome Prediction

A comprehensive machine learning system for predicting Supreme Court of the United States (SCOTUS) case outcomes using justice biographies, case descriptions, and historical voting patterns.

## 🎯 Project Overview

SCOTUS AI combines natural language processing, attention mechanisms, and historical data to predict how Supreme Court justices will vote on cases. The system processes justice biographies, case descriptions, and voting patterns to create predictive models with cross-attention mechanisms.

### Key Features

- **📊 Complete Data Pipeline**: Automated scraping and processing of Supreme Court data
- **🤖 Advanced ML Models**: Neural networks with justice-case cross-attention mechanisms  
- **🔄 End-to-End Automation**: From data collection to model training and prediction
- **📈 Hyperparameter Optimization**: Automated model tuning with Optuna
- **🐳 Docker Support**: Containerized deployment and execution
- **📋 Comprehensive Logging**: Detailed progress tracking and error handling

## 🏗️ Architecture

```
SCOTUS AI
├── Data Pipeline          # Automated data collection and processing
│   ├── Justice Scraping   # Wikipedia biographies and metadata
│   ├── Case Scraping      # Justia case descriptions with AI filtering
│   ├── SCDB Integration   # Supreme Court Database voting records
│   └── Data Processing    # Cleaning, enrichment, and dataset creation
├── ML Models             # Neural networks and training
│   ├── Voting Prediction # Main SCOTUS voting outcome model
│   ├── Cross-Attention   # Justice-case attention mechanisms
│   ├── Model Training    # Training pipeline with evaluation
│   └── Hyperparameter    # Automated optimization with Optuna
├── Tokenization          # Text encoding and embedding
│   ├── Biography Encoding # Justice biography embeddings
│   ├── Case Encoding     # Case description embeddings
│   └── Model Management  # Sentence transformer models
└── Utilities            # Configuration, logging, and tools
    ├── Configuration     # YAML and environment config
    ├── Logging          # Comprehensive logging system
    ├── Progress Tracking # tqdm-based progress bars
    └── Test Management  # Holdout test set management
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM
- 10GB+ disk space

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/scotus_ai.git
cd scotus_ai
```

2. **Set up environment:**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

3. **Configure environment:**
```bash
# Copy and edit environment configuration
cp env.example .env
# Edit .env with your API keys (GEMMA_KEY for Gemini API)
```

### Running the Complete Pipeline

```bash
# Run the complete data pipeline from scratch
python scripts/data_pipeline/main.py

# Or run specific components
python scripts/data_pipeline/main.py --step scrape-justices
python scripts/data_pipeline/main.py --step scrape-cases
python scripts/data_pipeline/main.py --step dataset
```

### Training Models

```bash
# Tokenize text data
python scripts/tokenization/main_encoder.py

# Train the voting prediction model
python scripts/models/model_trainer.py

# Run hyperparameter optimization
python scripts/models/hyperparameter_optimization.py
```

## 📊 Data Sources

### Primary Data Sources
- **[Supreme Court Database (SCDB)](http://scdb.wustl.edu/)**: Comprehensive voting records
- **[Wikipedia](https://en.wikipedia.org/)**: Justice biographies and metadata
- **[Justia](https://supreme.justia.com/)**: Case descriptions and legal documents

### Data Processing Pipeline
1. **Justice Metadata**: Scrapes Wikipedia for justice information
2. **Biography Processing**: Extracts pre-SCOTUS career information
3. **Case Descriptions**: AI-filtered to remove post-decision content
4. **Voting Records**: Processes SCDB justice-centered vote data
5. **Dataset Creation**: Creates ML-ready JSON dataset

## 🤖 Machine Learning Models

### SCOTUS Voting Model
- **Architecture**: Multi-layer neural network with cross-attention
- **Input**: Justice biographies + case descriptions
- **Output**: Voting probability distribution (in favor, against, absent, other)
- **Key Features**:
  - Justice-case cross-attention mechanism
  - Sentence transformer embeddings
  - Configurable architecture and hyperparameters

### Model Components
- **Biography Encoder**: `sentence-transformers/all-MiniLM-L6-v2`
- **Case Encoder**: `Stern5497/sbert-legal-xlm-roberta-base`
- **Attention Mechanism**: Multi-head cross-attention
- **Loss Function**: KL Divergence (configurable)

## 🔧 Configuration

### Environment Configuration
```bash
# API Keys
GEMMA_KEY=your_gemini_api_key_here

# Model Configuration
BIO_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
DESCRIPTION_MODEL_NAME=Stern5497/sbert-legal-xlm-roberta-base
EMBEDDING_DIM=384
HIDDEN_DIM=512

# Training Configuration
LEARNING_RATE=0.0001
BATCH_SIZE=4
NUM_EPOCHS=10
```

### YAML Configuration
Main configuration in `configs/base_config.yaml`:
```yaml
data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  
model:
  embedding_dim: 384
  hidden_dim: 512
  
training:
  learning_rate: 0.0001
  batch_size: 4
```

## 📁 Project Structure

```
scotus_ai/
├── configs/                    # Configuration files
│   └── base_config.yaml       # Main configuration
├── data/                      # Data storage
│   ├── raw/                   # Raw scraped data
│   ├── processed/             # Processed datasets
│   └── external/              # External data sources
├── scripts/                   # Main codebase
│   ├── data_pipeline/         # Data collection and processing
│   ├── models/               # ML models and training
│   ├── tokenization/         # Text encoding
│   └── utils/                # Utilities and helpers
├── logs/                     # Application logs
├── venv/                     # Virtual environment
├── docker-compose.yml        # Docker orchestration
├── Dockerfile               # Container definition
└── requirements.txt         # Python dependencies
```

## 🐳 Docker Deployment

### Using Docker Compose
```bash
# Build and run the complete system
docker-compose up --build

# Run specific services
docker-compose run scotus-ai python scripts/data_pipeline/main.py
docker-compose run scotus-ai python scripts/models/model_trainer.py

# Run hyperparameter optimization
docker-compose run scotus-ai hyperparameter-tuning --experiment-name my_experiment --n-trials 50
```

### Using Docker Run Script
```bash
# Build the image
./docker-run.sh build

# Run data pipeline
./docker-run.sh data-pipeline

# Train model
./docker-run.sh train

# Run hyperparameter tuning
./docker-run.sh tune --experiment-name architecture_test --n-trials 100
./docker-run.sh hyperparameter-tuning --experiment-name lr_test --n-trials 50

# Open interactive shell
./docker-run.sh shell
```

### Manual Docker Build
```bash
# Build the image
docker build -t scotus-ai .

# Run the container
docker run -it --gpus all -v $(pwd):/app scotus-ai

# Run hyperparameter tuning directly
docker run -it --gpus all -v $(pwd):/app scotus-ai hyperparameter-tuning --experiment-name test
```

## 📈 Usage Examples

### Data Pipeline
```python
# Run complete pipeline
from scripts.data_pipeline.main import run_full_pipeline
run_full_pipeline()

# Process specific components
from scripts.data_pipeline.scraper_justices import main as scrape_justices
scrape_justices("data/raw/justices.json")
```

### Model Training
```python
from scripts.models.model_trainer import SCOTUSModelTrainer

trainer = SCOTUSModelTrainer()
trainer.train_model("data/processed/case_dataset.json")
```

### Prediction
```python
from scripts.models.scotus_voting_model import SCOTUSVotingModel

model = SCOTUSVotingModel.load_model("models_output/best_model.pth")
prediction = model.predict_from_files(
    case_description_path="data/processed/case_descriptions/case_123.txt",
    justice_bio_paths=["data/processed/bios/John_Roberts.txt"]
)
```

## 🧪 Testing and Evaluation

### Holdout Test Set
```python
from scripts.utils.holdout_test_set import HoldoutTestSetManager

manager = HoldoutTestSetManager()
holdout_cases = manager.create_holdout_test_set(percentage=0.15)
```

### Model Evaluation
```python
from scripts.models.model_trainer import SCOTUSModelTrainer

trainer = SCOTUSModelTrainer()
results = trainer.evaluate_on_holdout_test_set("models_output/best_model.pth")
```

## 📊 Performance Metrics

The model is evaluated using:
- **KL Divergence Loss**: Measures prediction distribution accuracy
- **Mean Squared Error**: Alternative regression-style loss
- **Accuracy**: Classification accuracy for voting outcomes
- **F1 Score**: Balanced precision and recall

## 🔍 Hyperparameter Optimization

```bash
# Run Optuna optimization (local)
python scripts/models/hyperparameter_optimization.py --n-trials 100

# Run with Docker
./docker-run.sh tune --experiment-name architecture_study --n-trials 100
docker-compose run scotus-ai hyperparameter-tuning --experiment-name test --n-trials 50
```

Key hyperparameters optimized:
- Learning rate
- Hidden dimensions
- Dropout rates
- Attention heads
- Batch size
- Fine-tuning strategy

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📋 Requirements

### Python Dependencies
- `torch>=1.9.0`
- `transformers>=4.20.0`
- `sentence-transformers>=2.2.0`
- `pandas>=1.5.0`
- `numpy>=1.21.0`
- `beautifulsoup4>=4.11.0`
- `requests>=2.28.0`
- `tqdm>=4.64.0`
- `optuna>=3.0.0`
- `google-generativeai>=0.3.0`

### System Requirements
- CUDA-compatible GPU (recommended)
- 16GB+ RAM for full pipeline
- 10GB+ disk space for complete datasets

## 🚨 Important Notes

### Data Collection Ethics
- Respects robots.txt and rate limits
- Uses appropriate delays between requests
- Filters content to avoid data leakage
- Cites all data sources appropriately

### API Requirements
- **Gemini API**: Required for case description filtering
- **Rate Limits**: Automatic handling with resume capability
- **Quota Management**: Stops on API limit exceeded

### Legal Disclaimer
This project is for educational and research purposes only. Predictions should not be used for legal advice or decision-making. Always consult qualified legal professionals for legal matters.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Supreme Court Database (SCDB) for comprehensive voting data
- Wikipedia for justice biographical information
- Justia for case descriptions and legal documents
- Hugging Face for pre-trained language models
- The open-source community for various tools and libraries

## 📚 References

Justia, "Supreme Court," Justia Supreme Court Center. [Online]. Available: https://supreme.justia.com/.

H. J. Spaeth et al., "Supreme Court Database," The Supreme Court Database, Washington University in St. Louis. [Online]. Available: http://scdb.wustl.edu/data.php.

Wikipedia, "List of justices of the Supreme Court of the United States," Wikipedia, Jun. 23, 2025. [Online]. Available: https://en.wikipedia.org/wiki/List_of_justices_of_the_Supreme_Court_of_the_United_States.

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the documentation in individual module READMEs
- Review the configuration files for customization options

---

**Built with ❤️ for AI research and education** 
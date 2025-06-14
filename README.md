# SCOTUS AI Prediction Project

An AI-powered system to predict Supreme Court of the United States (SCOTUS) case outcomes using machine learning and natural language processing. This project creates a comprehensive dataset combining case metadata, justice biographies, and AI-filtered case descriptions for training predictive models.

## 🎯 Project Overview

This project aims to:
- **Extract comprehensive SCOTUS data** from multiple authoritative sources
- **Create AI-filtered case descriptions** that exclude post-decision information (preventing data leakage)
- **Build a complete ML dataset** with case metadata, justice biographies, and voting patterns
- **Enable outcome prediction** by training models on pre-decision information only

## 🏗️ System Architecture

The project uses a **9-step pipeline** that processes data from scratch to final ML-ready dataset:

### **Data Collection (Steps 1-5)**
1. **Justice Metadata** - Scrape Supreme Court justice information from Wikipedia
2. **Justice Biographies** - Download detailed biographies for each justice
3. **SCDB Data** - Download official Supreme Court Database voting records
4. **Case Processing** - Process raw SCDB data into structured metadata
5. **Case Descriptions** - Scrape and AI-filter case descriptions from Justia

### **Data Processing (Steps 6-9)**
6. **Biography Processing** - Enrich biographies with metadata and remove SCOTUS content
7. **Case Metadata** - Create natural language case descriptions from structured data
8. **Complete Descriptions** - Combine metadata with AI-filtered case descriptions
9. **Final Dataset** - Build JSON dataset mapping cases to file paths and voting data

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd scotus_ai

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp env.example .env

# Edit .env file with your API keys
# GEMMA_KEY=your_gemini_api_key_here
```

### 3. Run the Complete Pipeline

```bash
# Run the full pipeline (interactive mode)
python src/data_pipeline/main.py

# Or run in non-interactive mode
python src/data_pipeline/main.py --non-interactive
```

## 📊 Pipeline Features

### **Smart Resume Functionality**
- **Automatic detection** of existing processed files
- **Resume from any step** - no need to restart from beginning
- **Progress preservation** - continues exactly where it left off
- **API quota handling** - gracefully stops and resumes when limits exceeded

### **AI-Powered Content Filtering**
- **Gemini 2.0 Flash** integration for intelligent content processing
- **Pre-decision filtering** - removes post-decision information to prevent data leakage
- **Batch processing** - efficient handling of large case datasets
- **Quality control** - ensures filtered content meets ML training standards

### **Comprehensive Data Sources**
- **Wikipedia** - Justice metadata and biographies
- **Supreme Court Database (SCDB)** - Official voting records and case metadata
- **Justia** - Detailed case descriptions and legal content
- **AI Enhancement** - Intelligent content filtering and enrichment

## 🛠️ Usage Examples

### **Full Pipeline Execution**
```bash
# Interactive mode (recommended)
python src/data_pipeline/main.py

# Start from specific step
python src/data_pipeline/main.py --from-step 5

# Quick mode (limited processing for testing)
python src/data_pipeline/main.py --quick
```

### **Individual Step Execution**
```bash
# Scrape justice metadata
python src/data_pipeline/main.py --step scrape-justices

# Scrape case descriptions with AI filtering
python src/data_pipeline/main.py --step scrape-cases

# Build final dataset
python src/data_pipeline/main.py --step dataset
```

### **Data Status Check**
```bash
# Check what data already exists
python src/data_pipeline/main.py --check
```

## 📁 Project Structure

```
scotus_ai/
├── data/                          # Data storage
│   ├── raw/                      # Raw scraped data
│   │   ├── justices.json         # Justice metadata
│   │   ├── bios/                 # Raw justice biographies
│   │   ├── SCDB_2024_01_justiceCentered_Vote.csv  # Official voting data
│   │   └── case_descriptions_ai_filtered/  # AI-filtered case descriptions
│   └── processed/                # Cleaned and processed data
│       ├── cases_metadata.csv    # Processed case metadata with voting stats
│       ├── bios/                 # Processed justice biographies
│       ├── case_metadata/        # Natural language case descriptions
│       ├── case_descriptions/    # Complete case descriptions
│       └── case_dataset.json     # Final ML dataset
├── src/
│   ├── data_pipeline/            # Main pipeline scripts
│   │   ├── main.py              # Pipeline orchestrator
│   │   ├── scraper_justices.py  # Justice metadata scraper
│   │   ├── scraper_bios.py      # Biography scraper
│   │   ├── scraper_scdb.py      # SCDB data downloader
│   │   ├── scraper_case_descriptions.py  # Case description scraper with AI
│   │   ├── process_cases_metadata.py     # Case metadata processor
│   │   ├── process_bios.py      # Biography processor
│   │   ├── case_metadata_creation.py     # Case metadata creator
│   │   ├── case_descriptions_creation.py # Complete descriptions creator
│   │   └── build_case_dataset.py # Final dataset builder
│   └── utils/                    # Utility functions
│       ├── progress.py          # Progress bar utilities
│       └── logger.py            # Logging utilities
├── requirements.txt              # Python dependencies
├── env.example                   # Environment configuration template
└── README.md                     # This file
```

## 🔧 Key Components

### **Pipeline Orchestrator (`main.py`)**
- **Smart step detection** - automatically determines optimal starting point
- **Interactive mode** - guides users through pipeline execution
- **Error handling** - graceful handling of API limits and failures
- **Progress tracking** - real-time progress bars and status updates

### **AI Content Filtering (`scraper_case_descriptions.py`)**
- **Gemini 2.0 Flash integration** - intelligent content processing
- **Pre-decision filtering** - removes post-decision information
- **Resume functionality** - continues from where it left off
- **Batch processing** - efficient handling of large datasets

### **Data Processing Pipeline**
- **Metadata enrichment** - combines multiple data sources
- **Quality control** - ensures data integrity and completeness
- **Format standardization** - creates consistent data structures
- **ML-ready output** - produces training-ready datasets

## 📈 Dataset Statistics

The pipeline processes:
- **8,823 unique Supreme Court cases** (from SCDB)
- **116 justice biographies** (from Wikipedia)
- **AI-filtered case descriptions** (from Justia + Gemini)
- **Voting patterns** and outcome data for each case
- **Complete metadata** including parties, origins, and issue areas

## 🔑 Environment Variables

Required in `.env` file:
```bash
# AI API Key (for content filtering)
GEMMA_KEY=your_gemini_api_key_here

# Optional: Additional configuration
SCRAPER_DELAY=1.0
MAX_RETRIES=3
LOG_LEVEL=INFO
```

## 🎯 ML Dataset Output

The final dataset (`data/processed/case_dataset.json`) contains:
- **Case IDs** mapped to file paths
- **Justice biography paths** for each case
- **Case description paths** (AI-filtered)
- **Voting percentages** (in favor, against, absent)
- **Complete metadata** for training

## 🚨 Important Notes

### **API Usage**
- **Gemini API** is used for content filtering (requires API key)
- **Rate limiting** is built-in to respect API quotas
- **Resume functionality** prevents data loss during interruptions

### **Data Quality**
- **Pre-decision filtering** ensures no data leakage in ML training
- **Multiple validation steps** ensure data integrity
- **Comprehensive error handling** maintains pipeline reliability

### **Performance**
- **Batch processing** optimizes API usage
- **Progress tracking** provides real-time feedback
- **Resume capability** handles interruptions gracefully

## 🤝 Contributing

1. Follow the existing code structure and patterns
2. Add appropriate error handling and logging
3. Update documentation for new features
4. Test with the existing pipeline

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Troubleshooting

### **Common Issues**
- **API quota exceeded**: Pipeline will automatically stop and can be resumed
- **Missing dependencies**: Run `pip install -r requirements.txt`
- **Permission errors**: Ensure write access to data directories

### **Getting Help**
- Check the logs in console output
- Verify environment variables in `.env` file
- Ensure all required directories exist

---

**Ready to predict Supreme Court outcomes?** Run `python src/data_pipeline/main.py` to start building your dataset! 🚀 
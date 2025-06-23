# SCOTUS AI Data Pipeline

The data pipeline module handles automated collection, processing, and preparation of Supreme Court case data from multiple sources. It creates a comprehensive ML-ready dataset by scraping, filtering, and enriching legal data.

## 🏗️ Pipeline Overview

The data pipeline consists of 9 sequential steps that transform raw data sources into a machine learning-ready dataset:

```
Raw Data Sources → Processing Pipeline → ML Dataset
     ↓                      ↓              ↓
┌─────────────┐     ┌──────────────┐   ┌────────────┐
│ Wikipedia   │────→│   9-Step     │──→│ JSON       │
│ SCDB        │     │   Pipeline   │   │ Dataset    │
│ Justia      │     │              │   │            │
└─────────────┘     └──────────────┘   └────────────┘
```

## 📊 Pipeline Steps

### Data Collection (Steps 1-5)

1. **Justice Metadata Scraping** (`scraper_justices.py`)
   - Scrapes Supreme Court justice information from Wikipedia
   - Extracts biographical data, appointment details, and service periods
   - Output: `data/raw/justices.json`

2. **Justice Biography Scraping** (`scraper_bios.py`) 
   - Downloads complete Wikipedia biographies for each justice
   - Validates URLs and handles name correction mapping
   - Output: `data/raw/bios/*.txt`

3. **SCDB Data Download** (`scraper_scdb.py`)
   - Downloads official Supreme Court Database voting records
   - Extracts justice-centered vote data from Washington University
   - Output: `data/raw/SCDB_2024_01_justiceCentered_Vote.csv`

4. **Cases Metadata Processing** (`process_cases_metadata.py`)
   - Processes raw SCDB data into structured format
   - Computes voting percentages and statistics
   - Output: `data/processed/cases_metadata.csv`

5. **Case Description Scraping** (`scraper_case_descriptions.py`)
   - Scrapes case descriptions from Justia Supreme Court database
   - **AI-powered filtering** using Gemini 2.0 Flash to remove post-decision content
   - Resume-enabled processing with quota management
   - Output: `data/raw/case_descriptions_ai_filtered/*.txt`

### Data Processing (Steps 6-9)

6. **Biography Processing** (`process_bios.py`)
   - Enriches biographies with metadata from justice information
   - Removes SCOTUS-related content to prevent data leakage
   - Adds structured metadata headers
   - Output: `data/processed/bios/*.txt`

7. **Case Metadata Creation** (`case_metadata_creation.py`)
   - Creates natural language descriptions from structured case data
   - Decodes SCDB numeric codes into readable descriptions
   - Output: `data/processed/case_metadata/*.txt`

8. **Complete Case Descriptions** (`case_descriptions_creation.py`)
   - Combines metadata with AI-filtered case descriptions
   - Creates comprehensive case summaries for ML training
   - Output: `data/processed/case_descriptions/*.txt`

9. **Final Dataset Building** (`build_case_dataset.py`)
   - Creates JSON dataset mapping cases to file paths
   - Links justice biographies, case descriptions, and voting data
   - Output: `data/processed/case_dataset.json`

## 🚀 Usage

### Run Complete Pipeline

```bash
# Interactive mode (recommended)
python scripts/data_pipeline/main.py

# Non-interactive mode
python scripts/data_pipeline/main.py --non-interactive

# Start from specific step
python scripts/data_pipeline/main.py --from-step 5

# Quick test mode (limited data)
python scripts/data_pipeline/main.py --quick
```

### Run Individual Steps

```bash
# Scrape justice metadata
python scripts/data_pipeline/main.py --step scrape-justices

# Scrape case descriptions with AI filtering
python scripts/data_pipeline/main.py --step scrape-cases

# Build final dataset
python scripts/data_pipeline/main.py --step dataset
```

### Check Data Status

```bash
# Check existing data and pipeline status
python scripts/data_pipeline/main.py --check
```

## 🤖 AI-Powered Features

### Gemini 2.0 Flash Integration

The pipeline uses Google's Gemini 2.0 Flash model for intelligent content filtering:

- **Pre-decision Filtering**: Removes court decisions, outcomes, and rulings
- **Content Preservation**: Keeps case facts, legal issues, and procedural history  
- **Data Leakage Prevention**: Ensures ML training data doesn't contain outcomes
- **Quality Control**: Validates filtered content meets training standards

### Resume Capability

- **Automatic Resume**: Continues from where processing left off
- **API Quota Handling**: Gracefully stops when API limits exceeded
- **Progress Preservation**: Never loses processed data during interruptions
- **Smart Detection**: Automatically determines optimal starting point

## 📁 Data Sources

### Primary Sources

| Source | Purpose | Data Type |
|--------|---------|-----------|
| [Wikipedia](https://en.wikipedia.org/) | Justice biographies and metadata | Biographical text |
| [SCDB](http://scdb.wustl.edu/) | Official voting records | Structured data |
| [Justia](https://supreme.justia.com/) | Case descriptions | Legal text |

### Data Flow

```
Wikipedia → Justice Metadata + Biographies
    ↓
SCDB → Voting Records + Case Metadata  
    ↓
Justia → Case Descriptions
    ↓
Gemini AI → Filtered Case Content
    ↓
Processing → ML-Ready Dataset
```

## ⚙️ Configuration

### Environment Variables

```bash
# Required API key for AI filtering
GEMMA_KEY=your_gemini_api_key_here

# Optional configuration
SCRAPER_DELAY=1.0
MAX_RETRIES=3
LOG_LEVEL=INFO
```

### Pipeline Parameters

```python
# In main.py
run_full_pipeline(
    from_step=1,        # Starting step (1-9)
    quick_mode=False,   # Limit processing for testing
    interactive=True    # Prompt user for choices
)
```

## 📊 Output Structure

### Final Dataset Format

```json
{
  "case_id": [
    ["path/to/justice1_bio.txt", "path/to/justice2_bio.txt"],
    "path/to/case_description.txt", 
    [0.67, 0.22, 0.11, 0.0]  // [in_favor, against, absent, other]
  ]
}
```

### Directory Structure

```
data/
├── raw/                              # Raw scraped data
│   ├── justices.json                 # Justice metadata
│   ├── bios/                         # Raw biographies
│   ├── SCDB_2024_01_justiceCentered_Vote.csv
│   └── case_descriptions_ai_filtered/ # AI-filtered cases
├── processed/                        # Processed data
│   ├── cases_metadata.csv            # Processed case metadata
│   ├── bios/                         # Processed biographies
│   ├── case_metadata/                # Case metadata descriptions
│   ├── case_descriptions/            # Complete case descriptions
│   └── case_dataset.json             # Final ML dataset
```

## 🔧 Module Components

### Core Scripts

- **`main.py`**: Pipeline orchestrator with smart resume and step management
- **`scraper_*.py`**: Data collection scripts for different sources
- **`process_*.py`**: Data processing and cleaning scripts
- **`*_creation.py`**: Dataset creation and enrichment scripts
- **`build_case_dataset.py`**: Final dataset assembly

### Key Features

- **Progress Tracking**: Real-time progress bars with tqdm
- **Error Handling**: Comprehensive error recovery and logging
- **Data Validation**: Multiple validation checkpoints
- **Resume Support**: Automatic continuation from interruptions
- **Interactive Mode**: User guidance through pipeline execution

## 🚨 Important Notes

### Data Quality Assurance

- **Pre-decision Content**: AI filtering ensures no data leakage
- **Multiple Validation**: Each step validates data integrity
- **Quality Control**: Comprehensive error checking and handling
- **Source Attribution**: All data sources properly cited

### Ethical Considerations

- **Rate Limiting**: Respects API quotas and website policies
- **robots.txt Compliance**: Follows web scraping best practices
- **Attribution**: Proper citation of all data sources
- **Fair Use**: Educational and research use only

### Performance Optimization

- **Batch Processing**: Efficient handling of large datasets
- **Memory Management**: Optimized for large-scale processing
- **Progress Feedback**: Real-time status and completion estimates
- **Resource Management**: Automatic cleanup and memory optimization

## 🧪 Testing and Validation

### Pipeline Testing

```bash
# Run quick test with limited data
python scripts/data_pipeline/main.py --quick

# Test specific components
python scripts/data_pipeline/scraper_justices.py --output test_justices.json
```

### Data Validation

- **File Existence**: Validates all referenced files exist
- **Content Quality**: Checks for minimum content length and quality
- **Format Validation**: Ensures proper JSON and CSV formatting
- **Cross-References**: Validates relationships between datasets

## 📈 Performance Metrics

### Pipeline Statistics

- **Cases Processed**: ~8,823 unique Supreme Court cases
- **Justices Covered**: 116 Supreme Court justices (all-time)
- **AI Filtering**: ~95% content successfully filtered
- **Processing Time**: ~2-4 hours for complete pipeline
- **Data Volume**: ~500MB raw data, ~200MB processed

### Quality Metrics

- **Coverage Rate**: Percentage of cases with complete data
- **Filtering Accuracy**: AI content filtering success rate
- **Data Integrity**: Cross-validation between data sources
- **Completeness**: Percentage of complete case records

## 🛠️ Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| API quota exceeded | Gemini API rate limits | Wait and resume with same command |
| Missing dependencies | Package installation | Run `pip install -r requirements.txt` |
| Network timeouts | Connection issues | Retry with automatic resume |
| File permissions | Write access | Ensure data/ directory is writable |

### Debug Mode

```bash
# Enable verbose logging
python scripts/data_pipeline/main.py --verbose

# Check specific step status
python scripts/data_pipeline/main.py --step scrape-cases --verbose
```

## 📞 Support

For data pipeline issues:
- Check pipeline logs for specific error messages
- Verify API keys in `.env` file
- Ensure sufficient disk space (>10GB recommended)
- Review data source availability and access

---

**Ready to build your SCOTUS dataset?** Start with `python scripts/data_pipeline/main.py` 🚀 
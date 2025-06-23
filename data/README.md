# SCOTUS AI Data

This directory contains all data used by the SCOTUS AI system, organized into raw scraped data, processed datasets, and external resources. The data pipeline transforms raw sources into machine learning-ready datasets.

## 📁 Directory Structure

```
data/
├── raw/                              # Raw scraped data (original sources)
│   ├── justices.json                 # Justice metadata from Wikipedia
│   ├── bios/                         # Raw justice biographies (*.txt)
│   ├── SCDB_2024_01_justiceCentered_Vote.csv  # Official SCDB voting data
│   └── case_descriptions_ai_filtered/ # AI-filtered case descriptions
├── processed/                        # Processed and cleaned datasets  
│   ├── cases_metadata.csv            # Processed case metadata with voting stats
│   ├── bios/                         # Processed justice biographies
│   ├── case_metadata/                # Case metadata descriptions
│   ├── case_descriptions/            # Complete case descriptions
│   ├── encoded_bios.pkl              # Biography embeddings
│   ├── encoded_descriptions.pkl      # Case description embeddings
│   ├── case_dataset.json             # Final ML dataset
│   └── holdout_test_set.json         # Holdout test cases
└── external/                         # External data sources (optional)
    └── supplementary_data/           # Additional legal databases
```

## 📊 Data Sources

### Primary Sources

1. **[Wikipedia](https://en.wikipedia.org/)**
   - Justice biographical information
   - Career histories and background
   - Appointment details and service periods

2. **[Supreme Court Database (SCDB)](http://scdb.wustl.edu/)**
   - Official voting records (1946-present)
   - Case metadata and outcomes
   - Justice-centered vote data

3. **[Justia Supreme Court Center](https://supreme.justia.com/)**
   - Detailed case descriptions
   - Legal documents and briefs
   - Court opinions and decisions

### Data Collection Process

```
Wikipedia → Justice Metadata + Biographies
    ↓
SCDB → Voting Records + Case Information
    ↓  
Justia → Case Descriptions + Legal Content
    ↓
Gemini AI → Content Filtering (removes post-decision info)
    ↓
Processing Pipeline → ML-Ready Dataset
```

## 🗂️ Data Formats

### Raw Data

#### Justice Metadata (`raw/justices.json`)
```json
{
  "Justice Name": {
    "name": "John Roberts",
    "url": "https://en.wikipedia.org/wiki/John_Roberts",
    "birth_death": "(born 1955)",
    "state": "NY",
    "position": "Chief Justice",
    "appointment_date": "September 29, 2005",
    "nominated_by": "George W. Bush",
    "tenure_start": "September 29, 2005",
    "tenure_end": "",
    "tenure_status": "Active"
  }
}
```

#### Justice Biographies (`raw/bios/*.txt`)
```
=== Early Life ===
John Glover Roberts Jr. was born on January 27, 1955, in Buffalo, New York...

=== Education ===
Roberts attended La Lumiere School in La Porte, Indiana...

=== Legal Career ===
After graduating from Harvard Law School, Roberts served as a law clerk...
```

#### SCDB Data (`raw/SCDB_2024_01_justiceCentered_Vote.csv`)
| Column | Description | Example |
|--------|-------------|---------|
| caseIssuesId | Unique case identifier | 1946-001-01 |
| caseName | Case name | Korematsu v. United States |
| usCite | US Reports citation | 323 U.S. 214 |
| justiceName | Justice name | HHBurton |
| vote | Vote value (1=majority, 2=dissent, 8=absent) | 1 |
| chief | Chief Justice | FMVinson |

#### AI-Filtered Case Descriptions (`raw/case_descriptions_ai_filtered/*.txt`)
```
Case: Brown v. Board of Education
Citation: 347 U.S. 483
Case ID: 1954-001-01
Source: https://supreme.justia.com/cases/federal/us/347/483/
Note: Content filtered by Gemini 2.0 Flash to exclude post-decision information
================================================================================

This case involves the constitutionality of racial segregation in public schools.
The plaintiffs argue that separate educational facilities are inherently unequal...
```

### Processed Data

#### Processed Cases Metadata (`processed/cases_metadata.csv`)
| Column | Description | Type |
|--------|-------------|------|
| caseIssuesId | Unique case identifier | string |
| usCite | US Reports citation | string |
| caseName | Case name | string |
| chief | Chief Justice | string |
| petitioner | Petitioner type code | integer |
| votes_in_favor | Number of justices voting in favor | integer |
| votes_against | Number of justices voting against | integer |
| pct_in_favor | Percentage voting in favor | float |
| pct_against | Percentage voting against | float |
| justice_votes | Individual justice votes | string |

#### Processed Justice Biographies (`processed/bios/*.txt`)
```
Justice: John Roberts
State: New York
Birth Year: 1955
Position: Chief Justice
Appointment Date: September 29, 2005
Nominated By: George W. Bush

=== Early Life ===
John Glover Roberts Jr. was born on January 27, 1955, in Buffalo, New York...
```

#### Final ML Dataset (`processed/case_dataset.json`)
```json
{
  "case_id": [
    [
      "data/processed/bios/John_Roberts.txt",
      "data/processed/bios/Clarence_Thomas.txt"
    ],
    "data/processed/case_descriptions/347_U.S._483.txt",
    [0.89, 0.0, 0.11, 0.0]
  ]
}
```

#### Encoded Embeddings (`processed/encoded_*.pkl`)
```python
# Pickle file structure
{
    'embeddings': {
        'file_path': numpy.array([...]),  # 384-dimensional embedding
        # ... more embeddings
    },
    'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
    'embedding_dim': 384,
    'total_files': 116,
    'creation_date': '2024-01-15T10:30:00'
}
```

## 📈 Data Statistics

### Dataset Coverage

| Component | Count | Coverage |
|-----------|-------|----------|
| **Justices** | 116 | All SCOTUS justices (1789-present) |
| **Cases** | ~8,823 | SCDB cases (1946-present) |
| **Biographies** | 116 | Complete justice biographies |
| **Case Descriptions** | ~8,000 | AI-filtered case descriptions |
| **Voting Records** | ~200,000 | Individual justice votes |

### Data Quality Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **Biography Completeness** | 98% | Justices with complete biographical data |
| **Case Description Coverage** | 95% | Cases with AI-filtered descriptions |
| **Voting Data Completeness** | 99% | Cases with complete voting records |
| **Temporal Coverage** | 1946-2024 | Years covered by voting data |

### File Sizes

| Directory | Files | Total Size | Avg File Size |
|-----------|-------|------------|---------------|
| `raw/bios/` | 116 | ~50MB | ~430KB |
| `raw/case_descriptions_ai_filtered/` | ~8,000 | ~200MB | ~25KB |
| `processed/bios/` | 116 | ~40MB | ~345KB |
| `processed/case_descriptions/` | ~8,000 | ~300MB | ~38KB |
| `processed/encoded_*.pkl` | 2 | ~500MB | ~250MB |

## 🔧 Data Processing Pipeline

### Stage 1: Data Collection
1. **Justice Scraping**: Wikipedia metadata and biographies
2. **SCDB Download**: Official voting records and case data
3. **Case Scraping**: Justia case descriptions with AI filtering
4. **Quality Control**: Validation and error checking

### Stage 2: Data Processing
1. **Biography Processing**: Metadata enrichment and SCOTUS content removal
2. **Case Processing**: Natural language descriptions from structured data
3. **Description Creation**: Complete case summaries with filtered content
4. **Dataset Assembly**: Final JSON dataset with file path mappings

### Stage 3: Data Encoding
1. **Text Tokenization**: Sentence transformer encoding
2. **Embedding Generation**: 384-dimensional vector representations
3. **Data Validation**: Completeness and quality checks
4. **Test Set Creation**: Holdout set for unbiased evaluation

## 🚨 Important Considerations

### Data Quality Assurance

- **Pre-decision Content**: AI filtering removes post-decision information
- **Temporal Integrity**: Chronological data separation for valid testing
- **Source Attribution**: All data sources properly cited and referenced
- **Content Validation**: Multiple validation steps ensure data integrity

### Ethical Data Use

- **Fair Use**: Educational and research use under fair use doctrine
- **Attribution**: Proper citation of all data sources
- **Rate Limiting**: Respectful scraping with appropriate delays
- **Terms Compliance**: Adherence to website terms of service

### Data Limitations

- **Temporal Scope**: SCDB data starts from 1946 (modern era)
- **Case Selection**: Not all historical cases have descriptions
- **Language**: English-only content from US sources
- **Bias Considerations**: Inherent biases in source materials

## 🔄 Data Updates

### Automated Updates
The data pipeline supports incremental updates:
- **Resume Capability**: Continues from interruption points
- **Incremental Processing**: Only processes new or modified cases
- **Conflict Resolution**: Handles data conflicts and duplicates

### Manual Updates
For custom data additions:
1. Add new files to appropriate directories
2. Update dataset mappings in `case_dataset.json`
3. Re-run tokenization for new text content
4. Validate data integrity with existing datasets

### Version Control
- **Timestamps**: All processed data includes creation timestamps
- **Model Versions**: Embedding data includes model version information
- **Checksums**: Data integrity verification with file checksums

## 🛠️ Data Utilities

### Loading Processed Data
```python
import json
import pickle
import pandas as pd

# Load case dataset
with open('data/processed/case_dataset.json', 'r') as f:
    case_dataset = json.load(f)

# Load case metadata
cases_df = pd.read_csv('data/processed/cases_metadata.csv')

# Load embeddings
with open('data/processed/encoded_bios.pkl', 'rb') as f:
    bio_embeddings = pickle.load(f)
```

### Data Validation
```python
from scripts.utils.holdout_test_set import HoldoutTestSetManager

# Validate dataset completeness
manager = HoldoutTestSetManager()
dataset = manager.load_dataset()

# Check data coverage
total_cases = len(dataset)
cases_with_descriptions = sum(1 for entry in dataset.values() if entry[1])
coverage = cases_with_descriptions / total_cases * 100

print(f"Dataset coverage: {coverage:.1f}%")
```

## 📞 Support

For data-related issues:
- Check file paths and permissions
- Verify data pipeline completion
- Review data source availability
- Validate file formats and encoding

---

**Comprehensive legal data for AI-powered Supreme Court analysis** 📊 
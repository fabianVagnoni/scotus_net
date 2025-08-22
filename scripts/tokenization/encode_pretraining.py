import os
import logging
import sys
import re
import json

# Add the current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from config import get_config, get_bio_config, get_pretraining_config
except ImportError:
    # Fallback for when running as module from root
    try:
        from scripts.tokenization.config import get_config, get_bio_config, get_pretraining_config
    except ImportError:
        # Final fallback - try relative import
        from .config import get_config, get_bio_config, get_pretraining_config

from tqdm import tqdm
import pickle
from transformers import AutoTokenizer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def encode_pretraining_dataset(pretraining_full_bio_file: str = None, pretraining_trunc_bio_file: str = None, 
                             trunc_bios_dir: str = None, full_bios_dir: str = None, 
                             pretraining_dataset_file: str = None):
    """
    Encode pretraining dataset by tokenizing truncated and full biographies,
    and extract temporal information (appointment year and nominating president).
    
    Args:
        pretraining_full_bio_file: Path to save full bio tokenized data
        pretraining_trunc_bio_file: Path to save truncated bio tokenized data
        trunc_bios_dir: Directory containing truncated biography files
        full_bios_dir: Directory containing full biography files
        pretraining_dataset_file: Path to save pretraining dataset with temporal info
    """
    logger.info("🚀 Starting pretraining dataset encoding and temporal extraction")
    
    # Load configuration and set defaults
    config = get_config()
    bio_config = get_bio_config()
    pretraining_config = get_pretraining_config()
    
    logger.info(f"🤖 Loading tokenizer: {pretraining_config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(pretraining_config['model_name'])
    logger.info("✅ Tokenizer loaded successfully")
    
    # Get the project root directory (two levels up from scripts/tokenization/)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Use config values as defaults if not provided, and make them absolute paths
    if pretraining_full_bio_file is None:
        pretraining_full_bio_file = pretraining_config['pretraining_full_bio_file']
    if pretraining_trunc_bio_file is None:
        pretraining_trunc_bio_file = pretraining_config['pretraining_trunc_bio_file']
    if trunc_bios_dir is None:
        trunc_bios_dir = pretraining_config['trunc_bios_dir']
    if full_bios_dir is None:
        full_bios_dir = pretraining_config['full_bios_dir']
    if pretraining_dataset_file is None:
        pretraining_dataset_file = pretraining_config['pretraining_dataset_file']
    
    # Convert relative paths to absolute paths relative to project root
    pretraining_full_bio_file = os.path.join(project_root, pretraining_full_bio_file)
    pretraining_trunc_bio_file = os.path.join(project_root, pretraining_trunc_bio_file)
    trunc_bios_dir = os.path.join(project_root, trunc_bios_dir)
    full_bios_dir = os.path.join(project_root, full_bios_dir)
    pretraining_dataset_file = os.path.join(project_root, pretraining_dataset_file)
    
    logger.info(f"📁 Truncated bios directory: {trunc_bios_dir}")
    logger.info(f"📁 Full bios directory: {full_bios_dir}")
    logger.info(f"💾 Full bio output file: {pretraining_full_bio_file}")
    logger.info(f"💾 Truncated bio output file: {pretraining_trunc_bio_file}")
    logger.info(f"💾 Pretraining dataset file: {pretraining_dataset_file}")

    # Check if the directories exist
    if not os.path.exists(trunc_bios_dir):
        logger.error(f"❌ Truncated bios directory does not exist: {trunc_bios_dir}")
        raise FileNotFoundError(f"Truncated bios directory does not exist: {trunc_bios_dir}")
    if not os.path.exists(full_bios_dir):
        logger.error(f"❌ Full bios directory does not exist: {full_bios_dir}")
        raise FileNotFoundError(f"Full bios directory does not exist: {full_bios_dir}")

    # Get list of truncated and full biographies    
    trunc_justices = [f for f in os.listdir(trunc_bios_dir) if os.path.isfile(os.path.join(trunc_bios_dir, f)) and f.endswith('.txt')]
    full_justices = [f for f in os.listdir(full_bios_dir) if os.path.isfile(os.path.join(full_bios_dir, f)) and f.endswith('.txt')]
    
    logger.info(f"📚 Found {len(trunc_justices)} truncated biography files")
    logger.info(f"📚 Found {len(full_justices)} full biography files")
    
    # Build mapping from truncated filename -> corresponding full filename
    # Compatible with both unaugmented (exact match) and augmented (suffix like _v0) cases
    def canonicalize_base_name(filename: str) -> str:
        name = filename[:-4] if filename.lower().endswith('.txt') else filename
        # strip simple version suffixes like _v0, _v1 or -v2 at the end
        return re.sub(r'[_-]v\d+$', '', name)

    full_names_set = set(full_justices)
    trunc_to_full: dict[str, str] = {}

    for trunc in trunc_justices:
        # Try exact match first
        if trunc in full_names_set:
            trunc_to_full[trunc] = trunc
            continue
        # Fallback: map versioned/truncated variants (e.g., Abe_Fortas_v1.txt -> Abe_Fortas.txt)
        base = canonicalize_base_name(trunc)
        candidate_full = f"{base}.txt"
        if candidate_full in full_names_set:
            trunc_to_full[trunc] = candidate_full

    if len(trunc_to_full) == 0:
        logger.error("❌ No matching truncated→full biography pairs found between directories")
        raise ValueError("No matching truncated→full biography pairs found between directories")

    logger.info(f"✅ Found {len(trunc_to_full)} truncated→full biography pairs to process")
    
    # Initialize datasets
    pretraining_full_bio_dataset = {"tokenized_data": {}, "metadata": {}}
    pretraining_trunc_bio_dataset = {"tokenized_data": {}, "metadata": {}}
    pretraining_dataset = {}  # For temporal information
    
    # Load justices metadata
    justices_metadata_file = os.path.join(project_root, "data/raw/justices.json")
    logger.info(f"📖 Loading justices metadata from: {justices_metadata_file}")
    
    with open(justices_metadata_file, 'r', encoding='utf-8') as f:
        justices_metadata = json.load(f)
    
    logger.info(f"📚 Loaded metadata for {len(justices_metadata)} justices")
    
    logger.info(f"🧠 Starting tokenization and temporal extraction for {len(trunc_to_full)} truncated biographies")
    successful_encodings = 0
    successful_extractions = 0
    failed_encodings = 0
    failed_extractions = 0
    
    for trunc_name in tqdm(sorted(trunc_to_full.keys()), desc="Processing biographies"):
        try:
            trunc_bio_path = os.path.join(trunc_bios_dir, trunc_name)
            full_bio_filename = trunc_to_full[trunc_name]
            full_bio_path = os.path.join(full_bios_dir, full_bio_filename)
            
            logger.debug(f"Processing truncated bio: {trunc_name} → full bio: {full_bio_filename}")
            
            # Read biography texts
            with open(trunc_bio_path, 'r', encoding='utf-8') as f:
                trunc_bio = f.read()
            with open(full_bio_path, 'r', encoding='utf-8') as f:
                full_bio = f.read()
            
            # Tokenize full biography
            full_bio_encoded = tokenizer(full_bio, 
                                         add_special_tokens=True, 
                                         padding=True, 
                                         truncation=True, 
                                         max_length=pretraining_config['max_sequence_length'], 
                                         return_tensors='pt',
                                         return_token_type_ids=False,
                                         return_attention_mask=True)
            
            # Tokenize truncated biography
            trunc_bio_encoded = tokenizer(trunc_bio, 
                                          add_special_tokens=True, 
                                          padding=True, 
                                          truncation=True, 
                                          max_length=pretraining_config['max_sequence_length'], 
                                          return_tensors='pt',
                                          return_token_type_ids=False,
                                          return_attention_mask=True)
            
            # Store tokenized data
            pretraining_full_bio_dataset["tokenized_data"][trunc_name] = {
                "input_ids": full_bio_encoded["input_ids"].squeeze(0),
                "attention_mask": full_bio_encoded["attention_mask"].squeeze(0)
            }
            pretraining_trunc_bio_dataset["tokenized_data"][trunc_name] = {
                "input_ids": trunc_bio_encoded["input_ids"].squeeze(0),
                "attention_mask": trunc_bio_encoded["attention_mask"].squeeze(0)
            }
            
            successful_encodings += 1
            
            # Extract temporal information from justices metadata
            try:
                # Derive canonical justice name for metadata lookup (strip version suffix, replace underscores)
                canonical_base = canonicalize_base_name(trunc_name)
                justice_name = canonical_base.replace('_', ' ')
                
                # Look up justice in metadata
                if justice_name in justices_metadata:
                    justice_info = justices_metadata[justice_name]
                    
                    # Extract appointment year from appointment_date
                    appointment_date = justice_info.get('appointment_date', '')
                    appointment_year = None
                    if appointment_date:
                        # Extract year from date string (e.g., "September 26, 1789 ( Acclamation )" -> "1789")
                        year_match = re.search(r'(\d{4})', appointment_date)
                        if year_match:
                            appointment_year = year_match.group(1)
                    
                    # Get nominating president
                    nominating_president = justice_info.get('nominated_by', '')
                    
                    # Store extracted information in pretraining dataset
                    pretraining_dataset[trunc_name] = [appointment_year, nominating_president]
                    
                    # Log extraction results
                    if appointment_year:
                        successful_extractions += 1
                        if nominating_president:
                            logger.debug(f"Successfully extracted: Year={appointment_year}, President={nominating_president}")
                        else:
                            logger.debug(f"Successfully extracted: Year={appointment_year}, President=Unknown")
                    else:
                        failed_extractions += 1
                        logger.warning(f"Failed to extract appointment year for {justice_name}")
                        
                else:
                    failed_extractions += 1
                    logger.warning(f"Justice {justice_name} not found in metadata")
                    pretraining_dataset[trunc_name] = [None, None]
                    
            except Exception as e:
                failed_extractions += 1
                logger.error(f"Error extracting temporal information for {trunc_name}: {e}")
                # Store None values for failed extractions
                pretraining_dataset[trunc_name] = [None, None]
            
        except Exception as e:
            logger.error(f"❌ Error processing {trunc_name}: {e}")
            failed_encodings += 1
            continue
    
    # Log final statistics
    logger.info(f"✅ Successfully encoded {successful_encodings} biographies")
    logger.info(f"✅ Successfully extracted temporal info for {successful_extractions} justices")
    if failed_encodings > 0:
        logger.warning(f"⚠️  Failed to encode {failed_encodings} biographies")
    if failed_extractions > 0:
        logger.warning(f"⚠️  Failed to extract temporal info for {failed_extractions} justices")
    
    # Set metadata for tokenized datasets
    pretraining_full_bio_dataset["metadata"] = {
        'model_name': pretraining_config['model_name'],
        'max_sequence_length': pretraining_config['max_sequence_length'],
        'num_tokenized': len(pretraining_full_bio_dataset["tokenized_data"]),
        'failed_files': [],
        'device_used': pretraining_config['device'],
        'tokenization_method': 'transformers_autotokenizer'}
    
    pretraining_trunc_bio_dataset["metadata"] = {
        'model_name': pretraining_config['model_name'],
        'max_sequence_length': pretraining_config['max_sequence_length'],
        'num_tokenized': len(pretraining_trunc_bio_dataset["tokenized_data"]),
        'failed_files': [],
        'device_used': pretraining_config['device'],
        'tokenization_method': 'transformers_autotokenizer'}
    
    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(pretraining_full_bio_file), exist_ok=True)
    os.makedirs(os.path.dirname(pretraining_trunc_bio_file), exist_ok=True)
    os.makedirs(os.path.dirname(pretraining_dataset_file), exist_ok=True)
    
    # Save tokenized datasets
    logger.info(f"💾 Saving full bio dataset to: {pretraining_full_bio_file}")
    with open(pretraining_full_bio_file, 'wb') as f:
        pickle.dump(pretraining_full_bio_dataset, f)
    
    logger.info(f"💾 Saving truncated bio dataset to: {pretraining_trunc_bio_file}")
    with open(pretraining_trunc_bio_file, 'wb') as f:
        pickle.dump(pretraining_trunc_bio_dataset, f)
    
    # Save pretraining dataset with temporal information
    logger.info(f"💾 Saving pretraining dataset to: {pretraining_dataset_file}")
    with open(pretraining_dataset_file, 'w', encoding='utf-8') as f:
        json.dump(pretraining_dataset, f, indent=4)
    
    logger.info("🎉 Pretraining dataset encoding and temporal extraction completed successfully!")
    logger.info(f"📊 Final Summary:")
    logger.info(f"   - Total truncated biographies processed: {len(trunc_to_full)}")
    logger.info(f"   - Successful tokenizations: {successful_encodings}")
    logger.info(f"   - Successful temporal extractions: {successful_extractions}")
    logger.info(f"   - Tokenization success rate: {successful_encodings/len(trunc_to_full)*100:.1f}%")
    logger.info(f"   - Temporal extraction success rate: {successful_extractions/len(trunc_to_full)*100:.1f}%")


if __name__ == "__main__":
    encode_pretraining_dataset()
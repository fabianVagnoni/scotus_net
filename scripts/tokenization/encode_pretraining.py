import os
import logging
from scripts.tokenization.config import get_config, get_bio_config, get_pretraining_config
from tqdm import tqdm
import pickle
from transformers import AutoTokenizer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def encode_pretraining_dataset(pretraining_full_bio_file: str = None, pretraining_trunc_bio_file: str = None, trunc_bios_dir: str = None, full_bios_dir: str = None ):
    """
    Encode pretraining dataset by tokenizing truncated and full biographies.
    
    Args:
        pretraining_full_bio_file: Path to save full bio tokenized data
        pretraining_trunc_bio_file: Path to save truncated bio tokenized data
        trunc_bios_dir: Directory containing truncated biography files
        full_bios_dir: Directory containing full biography files
    """
    logger.info("🚀 Starting pretraining dataset encoding")
    
    # Load configuration and set defaults
    config = get_config()
    bio_config = get_bio_config()
    pretraining_config = get_pretraining_config()
    
    logger.info(f"🤖 Loading tokenizer: {pretraining_config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(pretraining_config['model_name'])
    logger.info("✅ Tokenizer loaded successfully")
    
    # Use config values as defaults if not provided
    if pretraining_full_bio_file is None:
        pretraining_full_bio_file = pretraining_config['pretraining_full_bio_file']
    if pretraining_trunc_bio_file is None:
        pretraining_trunc_bio_file = pretraining_config['pretraining_trunc_bio_file']
    if trunc_bios_dir is None:
        trunc_bios_dir = pretraining_config['trunc_bios_dir']
    if full_bios_dir is None:
        full_bios_dir = pretraining_config['full_bios_dir']
    
    logger.info(f"📁 Truncated bios directory: {trunc_bios_dir}")
    logger.info(f"📁 Full bios directory: {full_bios_dir}")
    logger.info(f"💾 Full bio output file: {pretraining_full_bio_file}")
    logger.info(f"💾 Truncated bio output file: {pretraining_trunc_bio_file}")
    
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
    
    if len(trunc_justices) != len(full_justices):
        logger.error(f"❌ Number of truncated ({len(trunc_justices)}) and full ({len(full_justices)}) justices do not match")
        raise ValueError("Number of truncated and full justices do not match")
    
    logger.info("✅ File count validation passed")
    
    pretraining_full_bio_dataset = {"tokenized_data": {}, "metadata": {}}
    pretraining_trunc_bio_dataset = {"tokenized_data": {}, "metadata": {}}
    
    logger.info(f"🧠 Starting tokenization of {len(trunc_justices)} justice biographies")
    successful_encodings = 0
    failed_encodings = 0
    
    for justice in tqdm(trunc_justices, desc="Encoding biographies"):
        try:
            trunc_bio_path = os.path.join(trunc_bios_dir, justice)
            full_bio_path = os.path.join(full_bios_dir, justice)
            
            logger.debug(f"Processing justice: {justice}")
            
            trunc_bio = open(trunc_bio_path, 'r').read()
            full_bio = open(full_bio_path, 'r').read()
            
            full_bio_encoded = tokenizer(full_bio, 
                                         add_special_tokens=True, 
                                         padding=True, 
                                         truncation=True, 
                                         max_length=pretraining_config['max_sequence_length'], 
                                         return_tensors='pt',
                                         return_token_type_ids=False,
                                         return_attention_mask=True)
            
            trunc_bio_encoded = tokenizer(trunc_bio, 
                                          add_special_tokens=True, 
                                          padding=True, 
                                          truncation=True, 
                                          max_length=pretraining_config['max_sequence_length'], 
                                          return_tensors='pt',
                                          return_token_type_ids=False,
                                          return_attention_mask=True)
            
            pretraining_full_bio_dataset["tokenized_data"][justice] = {
                "input_ids": full_bio_encoded["input_ids"].squeeze(0),
                "attention_mask": full_bio_encoded["attention_mask"].squeeze(0)
            }
            pretraining_trunc_bio_dataset["tokenized_data"][justice] = {
                "input_ids": trunc_bio_encoded["input_ids"].squeeze(0),
                "attention_mask": trunc_bio_encoded["attention_mask"].squeeze(0)
            }
            
            successful_encodings += 1
            
        except Exception as e:
            logger.error(f"❌ Error processing {justice}: {e}")
            failed_encodings += 1
            continue
    
    logger.info(f"✅ Successfully encoded {successful_encodings} biographies")
    if failed_encodings > 0:
        logger.warning(f"⚠️  Failed to encode {failed_encodings} biographies")
    
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
    
    logger.info(f"💾 Saving full bio dataset to: {pretraining_full_bio_file}")
    with open(pretraining_full_bio_file, 'wb') as f:
        pickle.dump(pretraining_full_bio_dataset, f)
    
    logger.info(f"💾 Saving truncated bio dataset to: {pretraining_trunc_bio_file}")
    with open(pretraining_trunc_bio_file, 'wb') as f:
        pickle.dump(pretraining_trunc_bio_dataset, f)
    
    logger.info("🎉 Pretraining dataset encoding completed successfully!")


if __name__ == "__main__":
    encode_pretraining_dataset()
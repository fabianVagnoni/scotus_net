import pickle
import re
import json
import logging
from scripts.pretraining.config import ContrastiveJusticeConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_pretraining_dataset(config):
    """
    Create a dataset for contrastive learning by extracting appointment dates and nominating presidents.
    
    Args:
        config: Configuration object containing file paths
        
    Returns:
        Dictionary mapping justice paths to [appointment_year, nominating_president]
    """
    logger.info("Starting pretraining dataset creation...")
    
    # Load tokenized biography data
    tokenized_files = "data/processed/encoded_bios.pkl" #config.tokenized_files
    logger.info(f"Loading tokenized data from: {tokenized_files}")
    
    with open(tokenized_files, 'rb') as f:
        tokenized_data = pickle.load(f)
    
    # Get list of justices from tokenized data
    justices = list(tokenized_data["tokenized_data"].keys())
    logger.info(f"Found {len(justices)} justices in tokenized data")
    
    # Regex patterns to extract appointment year and nominating president
    match_president = r'Nominated By:\s*(.+)'
    match_year = r'Appointment Date.*?(\d{4})'
    
    # Initialize dataset dictionary
    dataset = {}
    successful_extractions = 0
    failed_extractions = 0
    
    # Process each justice
    for i, justice in enumerate(justices, 1):
        logger.info(f"Processing justice {i}/{len(justices)}: {justice}")
        
        # Get justice data from tokenized data
        justice_data = tokenized_data["tokenized_data"][justice]
        
        # Read the raw biography file to extract temporal information
        try:
            with open(justice, "r", encoding="utf-8") as f:
                justice_text = f.read()
            
            # Extract appointment year and nominating president using regex
            justice_year = re.search(match_year, justice_text)
            justice_president = re.search(match_president, justice_text)
            
            # Store extracted information in dataset
            dataset[justice] = [
                justice_year.group(1) if justice_year is not None else None, 
                justice_president.group(1) if justice_president is not None else None
            ]
            
            # Log extraction results
            if justice_year is not None and justice_president is not None:
                successful_extractions += 1
                logger.debug(f"Successfully extracted: Year={justice_year.group(1)}, President={justice_president.group(1)}")
            else:
                failed_extractions += 1
                logger.warning(f"Failed to extract complete information for {justice}")
                
        except Exception as e:
            failed_extractions += 1
            logger.error(f"Error processing {justice}: {e}")
            # Continue with next justice
            continue
    
    # Log final statistics
    logger.info(f"Dataset creation completed:")
    logger.info(f"  - Total justices processed: {len(justices)}")
    logger.info(f"  - Successful extractions: {successful_extractions}")
    logger.info(f"  - Failed extractions: {failed_extractions}")
    logger.info(f"  - Success rate: {successful_extractions/len(justices)*100:.1f}%")
    
    return dataset

def main():
    """
    Main function to create and save the pretraining dataset.
    """
    logger.info("🚀 Starting pretraining dataset creation script")
    
    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = ContrastiveJusticeConfig()
        logger.info("Configuration loaded successfully")
        
        # Create the dataset
        logger.info("Creating pretraining dataset...")
        dataset = create_pretraining_dataset(config)
        
        # Save dataset to JSON file
        save_path = "data/processed/pretraining_dataset.json"
        logger.info(f"Saving dataset to: {save_path}")
        
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=4)
        
        logger.info(f"✅ Successfully saved dataset with {len(dataset)} entries")
        logger.info("🎉 Pretraining dataset creation completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Dataset creation failed: {e}")
        raise

if __name__ == "__main__":
    main()
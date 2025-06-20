#!/usr/bin/env python3
"""
SCOTUS AI Main Encoder Pipeline
==============================

Intelligent encoding pipeline that pre-encodes all texts referenced in the case dataset
for fast model training. Uses the same smart resumption pattern as the main data pipeline.

Features:
- Smart resumption: automatically detects what's already encoded
- Progress tracking with tqdm
- Graceful failure handling
- Updates dataset with encoded file locations
- Supports both biography and case description encoding
- Optimized batch processing

Pipeline Steps:
1. Load case dataset and analyze encoding requirements
2. Encode justice biographies (if needed)
3. Encode case descriptions (if needed)  
4. Update dataset with encoded file locations
5. Validate and report final status

Usage:
    python main_encoder.py                    # Run full encoding pipeline
    python main_encoder.py --bios-only       # Encode only biographies
    python main_encoder.py --descriptions-only # Encode only case descriptions
    python main_encoder.py --check           # Check encoding status
    python main_encoder.py --force           # Force re-encoding (overwrite existing)
"""

import os
import sys
import json
import argparse
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from tqdm import tqdm
import pickle

# Import configuration
from config import get_config, get_bio_config, get_description_config

def print_header(title: str, char: str = "="):
    """Print a formatted header."""
    width = 80
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}")

def print_step(step_num: int, total_steps: int, description: str):
    """Print a step header."""
    print(f"\n{'📍' if step_num <= total_steps else '✅'} STEP {step_num}/{total_steps}: {description}")
    print("-" * 60)

def run_script(script_path: str, args: List[str] = None, description: str = ""):
    """Run a Python script with optional arguments - allows real-time progress bar display."""
    if args is None:
        args = []
    
    cmd = [sys.executable, script_path] + args
    print(f"🚀 Running: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        # Don't capture output so tqdm progress bars can display in real-time
        result = subprocess.run(cmd, check=False)
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ Completed in {elapsed:.1f}s")
            return True
        else:
            print(f"❌ Failed after {elapsed:.1f}s")
            print(f"Return code: {result.returncode}")
            return False
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Unexpected error after {elapsed:.1f}s: {e}")
        return False

def load_case_dataset(dataset_file: str) -> Dict:
    """Load the case dataset and return it."""
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
    
    print(f"📥 Loading case dataset from: {dataset_file}")
    
    with open(dataset_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"✅ Loaded dataset with {len(dataset)} cases")
    return dataset

def analyze_encoding_requirements(dataset: Dict) -> Tuple[Set[str], Set[str], Dict]:
    """
    Analyze the dataset to determine what needs to be encoded.
    
    Returns:
        (unique_bio_paths, unique_case_paths, stats)
    """
    print("\n🔍 Analyzing encoding requirements...")
    
    unique_bio_paths = set()
    unique_case_paths = set()
    total_bios = 0
    total_cases = 0
    missing_bio_files = []
    missing_case_files = []
    
    for case_id, case_data in tqdm(dataset.items(), desc="Analyzing dataset"):
        # Handle both old format (3 elements) and new format (4 elements with encoded_locations)
        if len(case_data) == 3:
            justice_bio_paths, case_description_path, voting_percentages = case_data
        elif len(case_data) == 4:
            justice_bio_paths, case_description_path, voting_percentages, encoded_locations = case_data
        else:
            print(f"⚠️  Unexpected case data format for case {case_id}: {len(case_data)} elements")
            continue
        
        # Collect biography paths
        for bio_path in justice_bio_paths:
            if bio_path and bio_path.strip():
                abs_bio_path = os.path.abspath(bio_path)
                unique_bio_paths.add(abs_bio_path)
                total_bios += 1
                
                # Check if file exists
                if not os.path.exists(abs_bio_path):
                    missing_bio_files.append(abs_bio_path)
        
        # Collect case description paths
        if case_description_path and case_description_path.strip():
            abs_case_path = os.path.abspath(case_description_path)
            unique_case_paths.add(abs_case_path)
            total_cases += 1
            
            # Check if file exists
            if not os.path.exists(abs_case_path):
                missing_case_files.append(abs_case_path)
    
    stats = {
        'total_cases': len(dataset),
        'unique_bio_files': len(unique_bio_paths),
        'unique_case_files': len(unique_case_paths),
        'total_bio_references': total_bios,
        'total_case_references': total_cases,
        'missing_bio_files': len(missing_bio_files),
        'missing_case_files': len(missing_case_files)
    }
    
    print(f"\n📊 ENCODING REQUIREMENTS ANALYSIS:")
    print(f"   📚 Cases in dataset: {stats['total_cases']}")
    print(f"   👥 Unique biography files: {stats['unique_bio_files']}")
    print(f"   📖 Unique case description files: {stats['unique_case_files']}")
    print(f"   🔗 Total biography references: {stats['total_bio_references']}")
    print(f"   🔗 Total case description references: {stats['total_case_references']}")
    
    if missing_bio_files:
        print(f"   ⚠️  Missing biography files: {len(missing_bio_files)}")
    if missing_case_files:
        print(f"   ⚠️  Missing case description files: {len(missing_case_files)}")
    
    return unique_bio_paths, unique_case_paths, stats

def check_existing_encodings(bio_embedding_file: str, description_embedding_file: str, 
                           unique_bio_paths: Set[str], unique_case_paths: Set[str]) -> Dict:
    """Check what encodings already exist."""
    print("\n🔍 Checking existing encodings...")
    
    status = {
        'bio_file_exists': False,
        'description_file_exists': False,
        'bio_metadata_exists': False,
        'description_metadata_exists': False,
        'encoded_bios': set(),
        'encoded_descriptions': set(),
        'bio_coverage': 0.0,
        'description_coverage': 0.0
    }
    
    # Check biography encodings
    if os.path.exists(bio_embedding_file):
        status['bio_file_exists'] = True
        
        try:
            # Load existing bio encodings to check coverage
            with open(bio_embedding_file, 'rb') as f:
                data = pickle.load(f)
            
            bio_embeddings = data['embeddings']
            bio_metadata = data['metadata']
            status['bio_metadata_exists'] = True
            
            encoded_bio_paths = set(os.path.abspath(path) for path in bio_embeddings.keys())
            status['encoded_bios'] = encoded_bio_paths
            
            # Calculate coverage
            if unique_bio_paths:
                covered_bios = encoded_bio_paths.intersection(unique_bio_paths)
                status['bio_coverage'] = len(covered_bios) / len(unique_bio_paths)
            
            print(f"✅ Found biography encodings: {len(encoded_bio_paths)} files")
            print(f"   📊 Coverage: {status['bio_coverage']:.1%} of required files")
            
        except Exception as e:
            print(f"⚠️  Error reading biography encodings: {e}")
    else:
        print(f"⚪ No biography encodings found: {bio_embedding_file}")
    
    # Check description encodings
    if os.path.exists(description_embedding_file):
        status['description_file_exists'] = True
        
        try:
            # Load existing description encodings to check coverage
            with open(description_embedding_file, 'rb') as f:
                data = pickle.load(f)
            
            description_embeddings = data['embeddings']
            description_metadata = data['metadata']
            status['description_metadata_exists'] = True
            
            encoded_desc_paths = set(os.path.abspath(path) for path in description_embeddings.keys())
            status['encoded_descriptions'] = encoded_desc_paths
            
            # Calculate coverage
            if unique_case_paths:
                covered_descriptions = encoded_desc_paths.intersection(unique_case_paths)
                status['description_coverage'] = len(covered_descriptions) / len(unique_case_paths)
            
            print(f"✅ Found description encodings: {len(encoded_desc_paths)} files")
            print(f"   📊 Coverage: {status['description_coverage']:.1%} of required files")
            
        except Exception as e:
            print(f"⚠️  Error reading description encodings: {e}")
    else:
        print(f"⚪ No description encodings found: {description_embedding_file}")
    
    return status

def determine_encoding_plan(unique_bio_paths: Set[str], unique_case_paths: Set[str], 
                          encoding_status: Dict, force_reencoding: bool = False) -> Dict:
    """Determine what encoding steps need to be performed."""
    print("\n🎯 Determining encoding plan...")
    
    plan = {
        'encode_bios': False,
        'encode_descriptions': False,
        'missing_bio_files': [],
        'missing_description_files': [],
        'bios_to_encode': set(),
        'descriptions_to_encode': set()
    }
    
    if force_reencoding:
        print("🔄 Force re-encoding enabled - will re-encode all files")
        plan['encode_bios'] = len(unique_bio_paths) > 0
        plan['encode_descriptions'] = len(unique_case_paths) > 0
        plan['bios_to_encode'] = unique_bio_paths.copy()
        plan['descriptions_to_encode'] = unique_case_paths.copy()
    else:
        # Check what's missing for biographies
        if unique_bio_paths:
            missing_bios = unique_bio_paths - encoding_status['encoded_bios']
            if missing_bios:
                plan['encode_bios'] = True
                plan['bios_to_encode'] = missing_bios
                print(f"📝 Need to encode {len(missing_bios)} biography files")
            else:
                print(f"✅ All biography files already encoded")
        
        # Check what's missing for descriptions
        if unique_case_paths:
            missing_descriptions = unique_case_paths - encoding_status['encoded_descriptions']
            if missing_descriptions:
                plan['encode_descriptions'] = True
                plan['descriptions_to_encode'] = missing_descriptions
                print(f"📝 Need to encode {len(missing_descriptions)} description files")
            else:
                print(f"✅ All description files already encoded")
    
    # Filter out missing files
    existing_bios = {path for path in plan['bios_to_encode'] if os.path.exists(path)}
    missing_bios = plan['bios_to_encode'] - existing_bios
    plan['bios_to_encode'] = existing_bios
    plan['missing_bio_files'] = list(missing_bios)
    
    existing_descriptions = {path for path in plan['descriptions_to_encode'] if os.path.exists(path)}
    missing_descriptions = plan['descriptions_to_encode'] - existing_descriptions
    plan['descriptions_to_encode'] = existing_descriptions
    plan['missing_description_files'] = list(missing_descriptions)
    
    if missing_bios:
        print(f"⚠️  {len(missing_bios)} biography files are missing and will be skipped")
    if missing_descriptions:
        print(f"⚠️  {len(missing_descriptions)} description files are missing and will be skipped")
    
    return plan

def create_file_list_for_encoding(file_paths: Set[str], temp_dir: str, list_name: str) -> str:
    """Create a temporary file list for encoding specific files."""
    os.makedirs(temp_dir, exist_ok=True)
    list_file = os.path.join(temp_dir, f"{list_name}_files.txt")
    
    with open(list_file, 'w', encoding='utf-8') as f:
        for file_path in sorted(file_paths):
            f.write(f"{file_path}\n")
    
    print(f"📝 Created file list: {list_file} ({len(file_paths)} files)")
    return list_file

def encode_biographies(bios_to_encode: Set[str], output_file: str, 
                      existing_encodings: Set[str] = None) -> bool:
    """Encode biography files."""
    if not bios_to_encode:
        print("✅ No biographies need encoding")
        return True
    
    print_step(1, 2, f"Encoding {len(bios_to_encode)} Biography Files")
    
    # If we have existing encodings, we need to merge them
    if existing_encodings and os.path.exists(output_file):
        print(f"🔄 Merging with {len(existing_encodings)} existing encodings...")
        # Create temporary directory for the new encodings
        temp_output = output_file.replace('.pkl', '_temp.pkl')
    else:
        temp_output = output_file
    
    # Create temporary directory for file list
    temp_dir = "temp_encoding"
    bio_list_file = create_file_list_for_encoding(bios_to_encode, temp_dir, "bios")
    
    try:
        # Run biography encoding
        bio_config = get_bio_config()
        args = [
            "--file-list", bio_list_file,
            "--output", temp_output,
            "--model-name", bio_config['model_name'],
            "--embedding-dim", str(bio_config['embedding_dim']),
            "--batch-size", str(bio_config['batch_size'])
        ]
        
        success = run_script("src/models/encoding/encode_bios.py", args, 
                           f"Encoding {len(bios_to_encode)} biography files")
        
        # If we were doing a merge, combine the files
        if temp_output != output_file and success:
            success = merge_pickle_files(output_file, temp_output, output_file)
            if os.path.exists(temp_output):
                os.remove(temp_output)
        
        return success
        
    finally:
        # Cleanup temporary files
        if os.path.exists(bio_list_file):
            os.remove(bio_list_file)
        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
            os.rmdir(temp_dir)

def encode_descriptions(descriptions_to_encode: Set[str], output_file: str,
                       existing_encodings: Set[str] = None) -> bool:
    """Encode case description files."""
    if not descriptions_to_encode:
        print("✅ No descriptions need encoding")
        return True
    
    print_step(2, 2, f"Encoding {len(descriptions_to_encode)} Case Description Files")
    
    # If we have existing encodings, we need to merge them
    if existing_encodings and os.path.exists(output_file):
        print(f"🔄 Merging with {len(existing_encodings)} existing encodings...")
        temp_output = output_file.replace('.pkl', '_temp.pkl')
    else:
        temp_output = output_file
    
    # Create temporary directory for file list
    temp_dir = "temp_encoding"
    desc_list_file = create_file_list_for_encoding(descriptions_to_encode, temp_dir, "descriptions")
    
    try:
        # Run description encoding
        desc_config = get_description_config()
        args = [
            "--file-list", desc_list_file,
            "--output", temp_output,
            "--model-name", desc_config['model_name'],
            "--embedding-dim", str(desc_config['embedding_dim']),
            "--batch-size", str(desc_config['batch_size'])
        ]
        
        success = run_script("src/models/encoding/encode_descriptions.py", args,
                           f"Encoding {len(descriptions_to_encode)} case description files")
        
        # If we were doing a merge, combine the files
        if temp_output != output_file and success:
            success = merge_pickle_files(output_file, temp_output, output_file)
            if os.path.exists(temp_output):
                os.remove(temp_output)
        
        return success
        
    finally:
        # Cleanup temporary files
        if os.path.exists(desc_list_file):
            os.remove(desc_list_file)
        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
            os.rmdir(temp_dir)

def merge_pickle_files(existing_file: str, new_file: str, output_file: str) -> bool:
    """Merge two pickle files, removing duplicates."""
    try:
        print(f"🔄 Merging encodings...")
        
        # Load both files
        with open(existing_file, 'rb') as f:
            existing_data = pickle.load(f)
        with open(new_file, 'rb') as f:
            new_data = pickle.load(f)
        
        existing_embeddings = existing_data['embeddings']
        new_embeddings = new_data['embeddings']
        
        print(f"   📊 Existing: {len(existing_embeddings)} encodings")
        print(f"   📊 New: {len(new_embeddings)} encodings")
        
        # Combine embeddings, with new ones overwriting existing ones for same paths
        combined_embeddings = existing_embeddings.copy()
        combined_embeddings.update(new_embeddings)  # New embeddings take precedence
        
        print(f"   📊 Combined: {len(combined_embeddings)} encodings")
        
        # Merge metadata
        combined_metadata = existing_data['metadata'].copy()
        combined_metadata['num_embeddings'] = len(combined_embeddings)
        combined_metadata['merged_from'] = [existing_file, new_file]
        combined_metadata['merge_timestamp'] = time.time()
        
        # Save merged file
        merged_data = {
            'embeddings': combined_embeddings,
            'metadata': combined_metadata
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(merged_data, f)
        
        print(f"✅ Successfully merged encodings")
        return True
        
    except Exception as e:
        print(f"❌ Error merging files: {e}")
        return False

def update_dataset_with_encodings(dataset_file: str, bio_embedding_file: str, 
                                description_embedding_file: str) -> bool:
    """Update the dataset with encoded file locations."""
    print("\n🔄 Updating dataset with encoded file locations...")
    
    try:
        # Load the dataset
        with open(dataset_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        # Create backup
        backup_file = dataset_file.replace('.json', '_backup.json')
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2)
        print(f"📋 Created backup: {backup_file}")
        
        # Update each case entry
        updated_cases = 0
        
        for case_id, case_data in tqdm(dataset.items(), desc="Updating dataset"):
            # Handle both old and new formats
            if len(case_data) == 3:
                justice_bio_paths, case_description_path, voting_percentages = case_data
            elif len(case_data) == 4:
                justice_bio_paths, case_description_path, voting_percentages, _ = case_data
            else:
                print(f"⚠️  Unexpected case data format for case {case_id}: {len(case_data)} elements")
                continue
            
            # New format: [justice_bio_paths, case_description_path, voting_percentages, encoded_locations]
            encoded_locations = {
                'bio_embeddings_file': bio_embedding_file if os.path.exists(bio_embedding_file) else None,
                'description_embeddings_file': description_embedding_file if os.path.exists(description_embedding_file) else None
            }
            
            # Update the case data
            dataset[case_id] = [justice_bio_paths, case_description_path, voting_percentages, encoded_locations]
            updated_cases += 1
        
        # Save updated dataset
        with open(dataset_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"✅ Updated {updated_cases} cases with encoding locations")
        return True
        
    except Exception as e:
        print(f"❌ Error updating dataset: {e}")
        return False

def validate_final_encodings(dataset_file: str, bio_embedding_file: str, 
                           description_embedding_file: str) -> Dict:
    """Validate that all required files are encoded."""
    print("\n🔍 Validating final encodings...")
    
    # Load dataset and analyze requirements
    dataset = load_case_dataset(dataset_file)
    unique_bio_paths, unique_case_paths, stats = analyze_encoding_requirements(dataset)
    
    # Check final encoding status
    final_status = check_existing_encodings(bio_embedding_file, description_embedding_file,
                                          unique_bio_paths, unique_case_paths)
    
    validation_results = {
        'total_required_bios': len(unique_bio_paths),
        'total_required_descriptions': len(unique_case_paths),
        'encoded_bios': len(final_status['encoded_bios']),
        'encoded_descriptions': len(final_status['encoded_descriptions']),
        'bio_coverage': final_status['bio_coverage'],
        'description_coverage': final_status['description_coverage'],
        'missing_bio_encodings': unique_bio_paths - final_status['encoded_bios'],
        'missing_description_encodings': unique_case_paths - final_status['encoded_descriptions']
    }
    
    print(f"\n📊 FINAL VALIDATION RESULTS:")
    print(f"   👥 Biography coverage: {validation_results['bio_coverage']:.1%} "
          f"({validation_results['encoded_bios']}/{validation_results['total_required_bios']})")
    print(f"   📖 Description coverage: {validation_results['description_coverage']:.1%} "
          f"({validation_results['encoded_descriptions']}/{validation_results['total_required_descriptions']})")
    
    if validation_results['missing_bio_encodings']:
        print(f"   ⚠️  Missing {len(validation_results['missing_bio_encodings'])} biography encodings")
    if validation_results['missing_description_encodings']:
        print(f"   ⚠️  Missing {len(validation_results['missing_description_encodings'])} description encodings")
    
    return validation_results

def run_encoding_pipeline(dataset_file: str = None,
                         bio_embedding_file: str = None,
                         description_embedding_file: str = None,
                         bios_only: bool = False,
                         descriptions_only: bool = False,
                         force_reencoding: bool = False) -> bool:
    """Run the complete encoding pipeline."""
    
    # Load configuration and set defaults
    config = get_config()
    bio_config = get_bio_config()
    desc_config = get_description_config()
    
    if dataset_file is None:
        dataset_file = config.dataset_file
    if bio_embedding_file is None:
        bio_embedding_file = bio_config['output_file']
    if description_embedding_file is None:
        description_embedding_file = desc_config['output_file']
    
    print_header("🚀 SCOTUS AI ENCODING PIPELINE")
    start_time = time.time()
    
    try:
        # Step 1: Load and analyze dataset
        dataset = load_case_dataset(dataset_file)
        unique_bio_paths, unique_case_paths, stats = analyze_encoding_requirements(dataset)
        
        if not unique_bio_paths and not unique_case_paths:
            print("❌ No files found to encode in the dataset")
            return False
        
        # Step 2: Check existing encodings
        encoding_status = check_existing_encodings(bio_embedding_file, description_embedding_file,
                                                 unique_bio_paths, unique_case_paths)
        
        # Step 3: Determine encoding plan
        encoding_plan = determine_encoding_plan(unique_bio_paths, unique_case_paths, 
                                              encoding_status, force_reencoding)
        
        # Apply filters for bios_only or descriptions_only
        if bios_only:
            encoding_plan['encode_descriptions'] = False
            encoding_plan['descriptions_to_encode'] = set()
        elif descriptions_only:
            encoding_plan['encode_bios'] = False
            encoding_plan['bios_to_encode'] = set()
        
        # Check if there's anything to do
        if not encoding_plan['encode_bios'] and not encoding_plan['encode_descriptions']:
            print("\n✅ All required files are already encoded!")
            
            # Still update dataset with encoding locations
            update_success = update_dataset_with_encodings(dataset_file, bio_embedding_file, 
                                                         description_embedding_file)
            if update_success:
                validate_final_encodings(dataset_file, bio_embedding_file, description_embedding_file)
            
            return update_success
        
        print(f"\n🎯 ENCODING PLAN:")
        if encoding_plan['encode_bios']:
            print(f"   👥 Encode {len(encoding_plan['bios_to_encode'])} biography files")
        if encoding_plan['encode_descriptions']:
            print(f"   📖 Encode {len(encoding_plan['descriptions_to_encode'])} description files")
        
        # Step 4: Create output directories
        os.makedirs(os.path.dirname(bio_embedding_file), exist_ok=True)
        os.makedirs(os.path.dirname(description_embedding_file), exist_ok=True)
        
        # Step 5: Run encodings
        success = True
        
        if encoding_plan['encode_bios']:
            bio_success = encode_biographies(encoding_plan['bios_to_encode'], bio_embedding_file,
                                           encoding_status['encoded_bios'])
            success = success and bio_success
        
        if encoding_plan['encode_descriptions']:
            desc_success = encode_descriptions(encoding_plan['descriptions_to_encode'], 
                                             description_embedding_file,
                                             encoding_status['encoded_descriptions'])
            success = success and desc_success
        
        if not success:
            print("❌ Encoding pipeline failed")
            return False
        
        # Step 6: Update dataset with encoding locations
        update_success = update_dataset_with_encodings(dataset_file, bio_embedding_file, 
                                                     description_embedding_file)
        
        if not update_success:
            print("⚠️  Encodings completed but failed to update dataset")
            return False
        
        # Step 7: Final validation
        validation_results = validate_final_encodings(dataset_file, bio_embedding_file, 
                                                    description_embedding_file)
        
        # Pipeline completion
        total_time = time.time() - start_time
        
        print_header("🎉 ENCODING PIPELINE COMPLETED!", "=")
        print(f"⏱️  Total execution time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"📊 Final coverage:")
        print(f"   👥 Biographies: {validation_results['bio_coverage']:.1%}")
        print(f"   📖 Descriptions: {validation_results['description_coverage']:.1%}")
        print(f"💾 Output files:")
        print(f"   👥 Biography embeddings: {bio_embedding_file}")
        print(f"   📖 Description embeddings: {description_embedding_file}")
        print(f"   📋 Updated dataset: {dataset_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Encoding pipeline failed with error: {e}")
        return False

def check_encoding_status(dataset_file: str = None,
                         bio_embedding_file: str = None,
                         description_embedding_file: str = None):
    """Check and report encoding status without running the pipeline."""
    print_header("🔍 ENCODING STATUS CHECK")
    
    # Load configuration and set defaults
    config = get_config()
    bio_config = get_bio_config()
    desc_config = get_description_config()
    
    if dataset_file is None:
        dataset_file = config.dataset_file
    if bio_embedding_file is None:
        bio_embedding_file = bio_config['output_file']
    if description_embedding_file is None:
        description_embedding_file = desc_config['output_file']
    
    try:
        # Load and analyze dataset
        dataset = load_case_dataset(dataset_file)
        unique_bio_paths, unique_case_paths, stats = analyze_encoding_requirements(dataset)
        
        # Check existing encodings
        encoding_status = check_existing_encodings(bio_embedding_file, description_embedding_file,
                                                 unique_bio_paths, unique_case_paths)
        
        # Print comprehensive status
        print(f"\n📊 COMPREHENSIVE STATUS REPORT:")
        print(f"   📋 Dataset: {dataset_file}")
        print(f"   📚 Cases: {len(dataset)}")
        print(f"   👥 Unique biography files needed: {len(unique_bio_paths)}")
        print(f"   📖 Unique description files needed: {len(unique_case_paths)}")
        print(f"")
        print(f"   👥 Biography encodings:")
        print(f"      📁 File: {bio_embedding_file}")
        print(f"      ✅ Exists: {encoding_status['bio_file_exists']}")
        if encoding_status['bio_file_exists']:
            print(f"      📊 Coverage: {encoding_status['bio_coverage']:.1%}")
            print(f"      🔢 Encoded files: {len(encoding_status['encoded_bios'])}")
        print(f"")
        print(f"   📖 Description encodings:")
        print(f"      📁 File: {description_embedding_file}")
        print(f"      ✅ Exists: {encoding_status['description_file_exists']}")
        if encoding_status['description_file_exists']:
            print(f"      📊 Coverage: {encoding_status['description_coverage']:.1%}")
            print(f"      🔢 Encoded files: {len(encoding_status['encoded_descriptions'])}")
        
        # Determine what needs to be done
        encoding_plan = determine_encoding_plan(unique_bio_paths, unique_case_paths, encoding_status)
        
        if not encoding_plan['encode_bios'] and not encoding_plan['encode_descriptions']:
            print(f"\n✅ All required files are already encoded!")
        else:
            print(f"\n📝 ACTIONS NEEDED:")
            if encoding_plan['encode_bios']:
                print(f"   👥 Encode {len(encoding_plan['bios_to_encode'])} biography files")
            if encoding_plan['encode_descriptions']:
                print(f"   📖 Encode {len(encoding_plan['descriptions_to_encode'])} description files")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking encoding status: {e}")
        return False

def main():
    # Load configuration for defaults
    config = get_config()
    bio_config = get_bio_config()
    desc_config = get_description_config()
    
    parser = argparse.ArgumentParser(
        description="SCOTUS AI Main Encoder Pipeline - Pre-encode all texts for fast training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_encoder.py                    # Run full encoding pipeline
  python main_encoder.py --check           # Check encoding status only
  python main_encoder.py --bios-only       # Encode only biography files
  python main_encoder.py --descriptions-only # Encode only case description files
  python main_encoder.py --force           # Force re-encoding (overwrite existing)
  
  # Custom file locations
  python main_encoder.py --dataset data/my_dataset.json --bio-output my_bios.pkl
  
Pipeline Steps:
  1. Load case dataset and analyze requirements
  2. Check existing encodings for resumption
  3. Encode missing biography files (sentence transformer)
  4. Encode missing case description files (legal BERT)
  5. Update dataset with encoded file locations
  6. Validate final coverage

Features:
  - Smart resumption (only encodes missing files)
  - Progress tracking with tqdm
  - Graceful failure handling
  - Automatic dataset updating
  - Coverage validation
  - Centralized configuration via config.env
"""
    )
    
    parser.add_argument(
        "--dataset",
        default=config.dataset_file,
        help="Path to case dataset JSON file"
    )
    
    parser.add_argument(
        "--bio-output",
        default=bio_config['output_file'],
        help="Output file for biography embeddings"
    )
    
    parser.add_argument(
        "--description-output", 
        default=desc_config['output_file'],
        help="Output file for case description embeddings"
    )
    
    parser.add_argument(
        "--config",
        help="Path to configuration file (default: config.env in same directory)"
    )
    
    parser.add_argument(
        "--bios-only",
        action="store_true",
        help="Encode only biography files"
    )
    
    parser.add_argument(
        "--descriptions-only",
        action="store_true",
        help="Encode only case description files"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-encoding of all files (overwrite existing)"
    )
    
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check encoding status without running pipeline"
    )
    
    args = parser.parse_args()
    
    # Load custom config if provided
    if args.config:
        config = get_config(args.config)
        print(f"📋 Using custom config: {args.config}")
        config.print_config()
    
    # Validate arguments
    if args.bios_only and args.descriptions_only:
        print("❌ Cannot specify both --bios-only and --descriptions-only")
        sys.exit(1)
    
    # Handle check mode
    if args.check:
        success = check_encoding_status(args.dataset, args.bio_output, args.description_output)
        sys.exit(0 if success else 1)
    
    # Run encoding pipeline
    success = run_encoding_pipeline(
        dataset_file=args.dataset,
        bio_embedding_file=args.bio_output,
        description_embedding_file=args.description_output,
        bios_only=args.bios_only,
        descriptions_only=args.descriptions_only,
        force_reencoding=args.force
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 
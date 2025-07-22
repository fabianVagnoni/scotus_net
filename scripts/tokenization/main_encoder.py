#!/usr/bin/env python3
"""
SCOTUS AI Main Tokenization Pipeline
====================================

Intelligent tokenization pipeline that pre-tokenizes all texts referenced in the case dataset
for efficient model training. Uses the same smart resumption pattern as the main data pipeline.

Features:
- Smart resumption: automatically detects what's already tokenized
- Progress tracking with tqdm
- Graceful failure handling
- Updates dataset with tokenized file locations
- Supports both biography and case description tokenization
- Optimized batch processing

Pipeline Steps:
1. Load case dataset and analyze tokenization requirements
2. Tokenize justice biographies (if needed)
3. Tokenize case descriptions (if needed)  
4. Update dataset with tokenized file locations
5. Validate and report final status

Usage:
    python main_encoder.py                    # Run full tokenization pipeline
    python main_encoder.py --bios-only       # Tokenize only biographies
    python main_encoder.py --descriptions-only # Tokenize only case descriptions
    python main_encoder.py --check           # Check tokenization status
    python main_encoder.py --force           # Force re-tokenization (overwrite existing)
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
import os
import sys

# Add the current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from config import get_config, get_bio_config, get_description_config
except ImportError:
    # Fallback for when running as module from root
    try:
        from scripts.tokenization.config import get_config, get_bio_config, get_description_config
    except ImportError:
        # Final fallback - try relative import
        from .config import get_config, get_bio_config, get_description_config

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

def analyze_tokenization_requirements(dataset: Dict) -> Tuple[Set[str], Set[str], Dict]:
    """
    Analyze the dataset to determine what needs to be tokenized.
    
    Returns:
        (unique_bio_paths, unique_case_paths, stats)
    """
    print("\n🔍 Analyzing tokenization requirements...")
    print(f"🔍 Current working directory: {os.getcwd()}")
    
    unique_bio_paths = set()
    unique_case_paths = set()
    total_bios = 0
    total_cases = 0
    missing_bio_files = []
    missing_case_files = []
    
    # Debug: track first few paths to understand the issue
    debug_bio_paths = []
    debug_case_paths = []
    
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
                # Normalize path separators for cross-platform compatibility
                normalized_path = bio_path.strip().replace('\\', '/')
                unique_bio_paths.add(normalized_path)
                total_bios += 1
                
                # Debug: collect first few paths
                if len(debug_bio_paths) < 5:
                    debug_bio_paths.append(normalized_path)
                
                # Check if file exists
                if not os.path.exists(normalized_path):
                    missing_bio_files.append(normalized_path)
        
        # Collect case description paths
        if case_description_path and case_description_path.strip():
            # Normalize path separators for cross-platform compatibility
            normalized_path = case_description_path.strip().replace('\\', '/')
            unique_case_paths.add(normalized_path)
            total_cases += 1
            
            # Debug: collect first few paths
            if len(debug_case_paths) < 5:
                debug_case_paths.append(normalized_path)
            
            # Check if file exists
            if not os.path.exists(normalized_path):
                missing_case_files.append(normalized_path)
    
    stats = {
        'total_cases': len(dataset),
        'unique_bio_files': len(unique_bio_paths),
        'unique_case_files': len(unique_case_paths),
        'total_bio_references': total_bios,
        'total_case_references': total_cases,
        'missing_bio_files': len(missing_bio_files),
        'missing_case_files': len(missing_case_files)
    }
    
    # Debug output
    print(f"\n🔍 DEBUG: Sample paths found in dataset:")
    for i, path in enumerate(debug_bio_paths):
        exists = os.path.exists(path)
        print(f"   👥 Bio {i+1}: {path} (exists: {exists})")
    for i, path in enumerate(debug_case_paths):
        exists = os.path.exists(path)
        print(f"   📖 Case {i+1}: {path} (exists: {exists})")
    
    print(f"\n📊 TOKENIZATION REQUIREMENTS ANALYSIS:")
    print(f"   📚 Cases in dataset: {stats['total_cases']}")
    print(f"   👥 Unique biography files: {stats['unique_bio_files']}")
    print(f"   📖 Unique case description files: {stats['unique_case_files']}")
    print(f"   🔗 Total biography references: {stats['total_bio_references']}")
    print(f"   🔗 Total case description references: {stats['total_case_references']}")
    print(f"   💡 Note: Only tokenizing files referenced in dataset (more efficient)")
    
    if missing_bio_files:
        print(f"   ⚠️  Missing biography files: {len(missing_bio_files)}")
        # Show first few missing files for debugging
        for i, path in enumerate(missing_bio_files[:3]):
            print(f"      📋 Missing bio {i+1}: {path}")
    if missing_case_files:
        print(f"   ⚠️  Missing case description files: {len(missing_case_files)}")
        # Show first few missing files for debugging
        for i, path in enumerate(missing_case_files[:3]):
            print(f"      📋 Missing case {i+1}: {path}")
    
    return unique_bio_paths, unique_case_paths, stats

def check_existing_tokenizations(bio_tokenized_file: str, description_tokenized_file: str, 
                               unique_bio_paths: Set[str], unique_case_paths: Set[str]) -> Dict:
    """Check what tokenizations already exist."""
    print("\n🔍 Checking existing tokenizations...")
    
    status = {
        'bio_file_exists': False,
        'description_file_exists': False,
        'bio_metadata_exists': False,
        'description_metadata_exists': False,
        'tokenized_bios': set(),
        'tokenized_descriptions': set(),
        'bio_coverage': 0.0,
        'description_coverage': 0.0
    }
    
    # Check biography tokenizations
    if os.path.exists(bio_tokenized_file):
        status['bio_file_exists'] = True
        
        try:
            # Load existing bio tokenizations to check coverage
            with open(bio_tokenized_file, 'rb') as f:
                data = pickle.load(f)
            
            bio_tokenized_data = data['tokenized_data']
            bio_metadata = data['metadata']
            status['bio_metadata_exists'] = True
            
            tokenized_bio_paths = set(path.replace('\\', '/') for path in bio_tokenized_data.keys())
            status['tokenized_bios'] = tokenized_bio_paths
            
            # Calculate coverage
            if unique_bio_paths:
                covered_bios = tokenized_bio_paths.intersection(unique_bio_paths)
                status['bio_coverage'] = len(covered_bios) / len(unique_bio_paths)
            
            print(f"✅ Found biography tokenizations: {len(tokenized_bio_paths)} files")
            print(f"   📊 Coverage: {status['bio_coverage']:.1%} of required files")
            
        except Exception as e:
            print(f"⚠️  Error reading biography tokenizations: {e}")
    else:
        print(f"⚪ No biography tokenizations found: {bio_tokenized_file}")
    
    # Check description tokenizations
    if os.path.exists(description_tokenized_file):
        status['description_file_exists'] = True
        
        try:
            # Load existing description tokenizations to check coverage
            with open(description_tokenized_file, 'rb') as f:
                data = pickle.load(f)
            
            description_tokenized_data = data['tokenized_data']
            description_metadata = data['metadata']
            status['description_metadata_exists'] = True
            
            tokenized_desc_paths = set(path.replace('\\', '/') for path in description_tokenized_data.keys())
            status['tokenized_descriptions'] = tokenized_desc_paths
            
            # Calculate coverage
            if unique_case_paths:
                covered_descriptions = tokenized_desc_paths.intersection(unique_case_paths)
                status['description_coverage'] = len(covered_descriptions) / len(unique_case_paths)
            
            print(f"✅ Found description tokenizations: {len(tokenized_desc_paths)} files")
            print(f"   📊 Coverage: {status['description_coverage']:.1%} of required files")
            
        except Exception as e:
            print(f"⚠️  Error reading description tokenizations: {e}")
    else:
        print(f"⚪ No description tokenizations found: {description_tokenized_file}")
    
    return status

def determine_tokenization_plan(unique_bio_paths: Set[str], unique_case_paths: Set[str], 
                              tokenization_status: Dict, force_retokenization: bool = False) -> Dict:
    """Determine what tokenization steps need to be performed."""
    print("\n🎯 Determining tokenization plan...")
    
    plan = {
        'tokenize_bios': False,
        'tokenize_descriptions': False,
        'missing_bio_files': [],
        'missing_description_files': [],
        'bios_to_tokenize': set(),
        'descriptions_to_tokenize': set()
    }
    
    if force_retokenization:
        print("🔄 Force re-tokenization enabled - will re-tokenize all files")
        plan['tokenize_bios'] = len(unique_bio_paths) > 0
        plan['tokenize_descriptions'] = len(unique_case_paths) > 0
        plan['bios_to_tokenize'] = unique_bio_paths.copy()
        plan['descriptions_to_tokenize'] = unique_case_paths.copy()
    else:
        # Check what's missing for biographies
        if unique_bio_paths:
            missing_bios = unique_bio_paths - tokenization_status['tokenized_bios']
            if missing_bios:
                plan['tokenize_bios'] = True
                plan['bios_to_tokenize'] = missing_bios
                print(f"📝 Need to tokenize {len(missing_bios)} biography files")
            else:
                print(f"✅ All biography files already tokenized")
        
        # Check what's missing for descriptions
        if unique_case_paths:
            missing_descriptions = unique_case_paths - tokenization_status['tokenized_descriptions']
            if missing_descriptions:
                plan['tokenize_descriptions'] = True
                plan['descriptions_to_tokenize'] = missing_descriptions
                print(f"📝 Need to tokenize {len(missing_descriptions)} description files")
            else:
                print(f"✅ All description files already tokenized")
    
    # Filter out missing files
    existing_bios = {path for path in plan['bios_to_tokenize'] if os.path.exists(path)}
    missing_bios = plan['bios_to_tokenize'] - existing_bios
    plan['bios_to_tokenize'] = existing_bios
    plan['missing_bio_files'] = list(missing_bios)
    
    existing_descriptions = {path for path in plan['descriptions_to_tokenize'] if os.path.exists(path)}
    missing_descriptions = plan['descriptions_to_tokenize'] - existing_descriptions
    plan['descriptions_to_tokenize'] = existing_descriptions
    plan['missing_description_files'] = list(missing_descriptions)
    
    if missing_bios:
        print(f"⚠️  {len(missing_bios)} biography files are missing and will be skipped")
    if missing_descriptions:
        print(f"⚠️  {len(missing_descriptions)} description files are missing and will be skipped")
    
    return plan

def create_file_list_for_tokenization(file_paths: Set[str], temp_dir: str, list_name: str) -> str:
    """Create a temporary file list for tokenizing specific files."""
    os.makedirs(temp_dir, exist_ok=True)
    list_file = os.path.join(temp_dir, f"{list_name}_files.txt")
    
    with open(list_file, 'w', encoding='utf-8') as f:
        for file_path in sorted(file_paths):
            f.write(f"{file_path}\n")
    
    print(f"📝 Created file list: {list_file} ({len(file_paths)} files)")
    return list_file

def tokenize_biographies(bios_to_tokenize: Set[str], output_file: str, 
                        existing_tokenizations: Set[str] = None) -> bool:
    """Tokenize biography files."""
    if not bios_to_tokenize:
        print("✅ No biographies need tokenization")
        return True
    
    print_step(1, 2, f"Tokenizing {len(bios_to_tokenize)} Biography Files")
    
    # If we have existing tokenizations, we need to merge them
    if existing_tokenizations and os.path.exists(output_file):
        print(f"🔄 Merging with {len(existing_tokenizations)} existing tokenizations...")
        # Create temporary directory for the new tokenizations
        temp_output = output_file.replace('.pkl', '_temp.pkl')
    else:
        temp_output = output_file
    
    # Create temporary directory for file list
    temp_dir = "temp_tokenization"
    bio_list_file = create_file_list_for_tokenization(bios_to_tokenize, temp_dir, "bios")
    
    try:
        # Run biography tokenization
        bio_config = get_bio_config()
        args = [
            "--file-list", bio_list_file,
            "--output", temp_output,
            "--model-name", bio_config['model_name'],
            "--batch-size", str(bio_config['batch_size'])
        ]
        
        success = run_script("scripts/tokenization/encode_bios.py", args, 
                           f"Tokenizing {len(bios_to_tokenize)} biography files")
        
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

def tokenize_descriptions(descriptions_to_tokenize: Set[str], output_file: str,
                         existing_tokenizations: Set[str] = None) -> bool:
    """Tokenize case description files."""
    if not descriptions_to_tokenize:
        print("✅ No descriptions need tokenization")
        return True
    
    print_step(2, 2, f"Tokenizing {len(descriptions_to_tokenize)} Case Description Files")
    
    # If we have existing tokenizations, we need to merge them
    if existing_tokenizations and os.path.exists(output_file):
        print(f"🔄 Merging with {len(existing_tokenizations)} existing tokenizations...")
        temp_output = output_file.replace('.pkl', '_temp.pkl')
    else:
        temp_output = output_file
    
    # Create temporary directory for file list
    temp_dir = "temp_tokenization"
    desc_list_file = create_file_list_for_tokenization(descriptions_to_tokenize, temp_dir, "descriptions")
    
    try:
        # Run description tokenization
        desc_config = get_description_config()
        args = [
            "--file-list", desc_list_file,
            "--output", temp_output,
            "--model-name", desc_config['model_name'],
            "--batch-size", str(desc_config['batch_size'])
        ]
        
        success = run_script("scripts/tokenization/encode_descriptions.py", args,
                           f"Tokenizing {len(descriptions_to_tokenize)} case description files")
        
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
        print(f"🔄 Merging tokenizations...")
        
        # Load both files
        with open(existing_file, 'rb') as f:
            existing_data = pickle.load(f)
        with open(new_file, 'rb') as f:
            new_data = pickle.load(f)
        
        existing_tokenized = existing_data['tokenized_data']
        new_tokenized = new_data['tokenized_data']
        
        print(f"   📊 Existing: {len(existing_tokenized)} tokenizations")
        print(f"   📊 New: {len(new_tokenized)} tokenizations")
        
        # Combine tokenized data, with new ones overwriting existing ones for same paths
        combined_tokenized = existing_tokenized.copy()
        combined_tokenized.update(new_tokenized)  # New tokenizations take precedence
        
        print(f"   📊 Combined: {len(combined_tokenized)} tokenizations")
        
        # Merge metadata
        combined_metadata = existing_data['metadata'].copy()
        combined_metadata['num_tokenized'] = len(combined_tokenized)
        combined_metadata['merged_from'] = [existing_file, new_file]
        combined_metadata['merge_timestamp'] = time.time()
        
        # Save merged file
        merged_data = {
            'tokenized_data': combined_tokenized,
            'metadata': combined_metadata
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(merged_data, f)
        
        print(f"✅ Successfully merged tokenizations")
        return True
        
    except Exception as e:
        print(f"❌ Error merging files: {e}")
        return False

def update_dataset_with_tokenizations(dataset_file: str, bio_tokenized_file: str, 
                                     description_tokenized_file: str) -> bool:
    """Update the dataset with tokenized file locations."""
    print("\n🔄 Updating dataset with tokenized file locations...")
    
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
            
            # New format: [justice_bio_paths, case_description_path, voting_percentages, tokenized_locations]
            tokenized_locations = {
                'bio_tokenized_file': bio_tokenized_file if os.path.exists(bio_tokenized_file) else None,
                'description_tokenized_file': description_tokenized_file if os.path.exists(description_tokenized_file) else None
            }
            
            # Update the case data
            dataset[case_id] = [justice_bio_paths, case_description_path, voting_percentages, tokenized_locations]
            updated_cases += 1
        
        # Save updated dataset
        with open(dataset_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"✅ Updated {updated_cases} cases with tokenized locations")
        return True
        
    except Exception as e:
        print(f"❌ Error updating dataset: {e}")
        return False

def validate_final_tokenizations(dataset_file: str, bio_tokenized_file: str, 
                               description_tokenized_file: str) -> Dict:
    """Validate that all required files are tokenized."""
    print("\n🔍 Validating final tokenizations...")
    
    # Load dataset and analyze requirements
    dataset = load_case_dataset(dataset_file)
    unique_bio_paths, unique_case_paths, stats = analyze_tokenization_requirements(dataset)
    
    # Check final tokenization status
    final_status = check_existing_tokenizations(bio_tokenized_file, description_tokenized_file,
                                              unique_bio_paths, unique_case_paths)
    
    validation_results = {
        'total_required_bios': len(unique_bio_paths),
        'total_required_descriptions': len(unique_case_paths),
        'tokenized_bios': len(final_status['tokenized_bios']),
        'tokenized_descriptions': len(final_status['tokenized_descriptions']),
        'bio_coverage': final_status['bio_coverage'],
        'description_coverage': final_status['description_coverage'],
        'missing_bio_tokenizations': unique_bio_paths - final_status['tokenized_bios'],
        'missing_description_tokenizations': unique_case_paths - final_status['tokenized_descriptions']
    }
    
    print(f"\n📊 FINAL VALIDATION RESULTS:")
    print(f"   👥 Biography coverage: {validation_results['bio_coverage']:.1%} "
          f"({validation_results['tokenized_bios']}/{validation_results['total_required_bios']})")
    print(f"   📖 Description coverage: {validation_results['description_coverage']:.1%} "
          f"({validation_results['tokenized_descriptions']}/{validation_results['total_required_descriptions']})")
    
    if validation_results['missing_bio_tokenizations']:
        print(f"   ⚠️  Missing {len(validation_results['missing_bio_tokenizations'])} biography tokenizations")
    if validation_results['missing_description_tokenizations']:
        print(f"   ⚠️  Missing {len(validation_results['missing_description_tokenizations'])} description tokenizations")
    
    return validation_results

def run_tokenization_pipeline(dataset_file: str = None,
                             bio_tokenized_file: str = None,
                             description_tokenized_file: str = None,
                             bios_only: bool = False,
                             descriptions_only: bool = False,
                             force_retokenization: bool = False) -> bool:
    """Run the complete tokenization pipeline."""
    
    # Load configuration and set defaults
    config = get_config()
    bio_config = get_bio_config()
    desc_config = get_description_config()
    
    if dataset_file is None:
        dataset_file = config.dataset_file
    if bio_tokenized_file is None:
        bio_tokenized_file = bio_config['output_file']
    if description_tokenized_file is None:
        description_tokenized_file = desc_config['output_file']
    
    print_header("🚀 SCOTUS AI TOKENIZATION PIPELINE")
    start_time = time.time()
    
    try:
        # Step 1: Load and analyze dataset
        dataset = load_case_dataset(dataset_file)
        unique_bio_paths, unique_case_paths, stats = analyze_tokenization_requirements(dataset)
        
        if not unique_bio_paths and not unique_case_paths:
            print("❌ No files found to tokenize in the dataset")
            return False
        
        # Step 2: Check existing tokenizations
        tokenization_status = check_existing_tokenizations(bio_tokenized_file, description_tokenized_file,
                                                          unique_bio_paths, unique_case_paths)
        
        # Step 3: Determine tokenization plan
        tokenization_plan = determine_tokenization_plan(unique_bio_paths, unique_case_paths, 
                                                       tokenization_status, force_retokenization)
        
        # Apply filters for bios_only or descriptions_only
        if bios_only:
            tokenization_plan['tokenize_descriptions'] = False
            tokenization_plan['descriptions_to_tokenize'] = set()
        elif descriptions_only:
            tokenization_plan['tokenize_bios'] = False
            tokenization_plan['bios_to_tokenize'] = set()
        
        # Check if there's anything to do
        if not tokenization_plan['tokenize_bios'] and not tokenization_plan['tokenize_descriptions']:
            print("\n✅ All required files are already tokenized!")
            
            # Still update dataset with tokenization locations
            update_success = update_dataset_with_tokenizations(dataset_file, bio_tokenized_file, 
                                                              description_tokenized_file)
            if update_success:
                validate_final_tokenizations(dataset_file, bio_tokenized_file, description_tokenized_file)
            
            return update_success
        
        print(f"\n🎯 TOKENIZATION PLAN:")
        if tokenization_plan['tokenize_bios']:
            print(f"   👥 Tokenize {len(tokenization_plan['bios_to_tokenize'])} biography files")
        if tokenization_plan['tokenize_descriptions']:
            print(f"   📖 Tokenize {len(tokenization_plan['descriptions_to_tokenize'])} description files")
        
        # Step 4: Create output directories
        os.makedirs(os.path.dirname(bio_tokenized_file), exist_ok=True)
        os.makedirs(os.path.dirname(description_tokenized_file), exist_ok=True)
        
        # Step 5: Run tokenizations
        success = True
        
        if tokenization_plan['tokenize_bios']:
            bio_success = tokenize_biographies(tokenization_plan['bios_to_tokenize'], bio_tokenized_file,
                                              tokenization_status['tokenized_bios'])
            success = success and bio_success
        
        if tokenization_plan['tokenize_descriptions']:
            desc_success = tokenize_descriptions(tokenization_plan['descriptions_to_tokenize'], 
                                                description_tokenized_file,
                                                tokenization_status['tokenized_descriptions'])
            success = success and desc_success
        
        if not success:
            print("❌ Tokenization pipeline failed")
            return False
        
        # Step 6: Update dataset with tokenization locations
        update_success = update_dataset_with_tokenizations(dataset_file, bio_tokenized_file, 
                                                          description_tokenized_file)
        
        if not update_success:
            print("⚠️  Tokenizations completed but failed to update dataset")
            return False
        
        # Step 7: Final validation
        validation_results = validate_final_tokenizations(dataset_file, bio_tokenized_file, 
                                                         description_tokenized_file)
        
        # Pipeline completion
        total_time = time.time() - start_time
        
        print_header("🎉 TOKENIZATION PIPELINE COMPLETED!", "=")
        print(f"⏱️  Total execution time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"📊 Final coverage:")
        print(f"   👥 Biographies: {validation_results['bio_coverage']:.1%}")
        print(f"   📖 Descriptions: {validation_results['description_coverage']:.1%}")
        print(f"💾 Output files:")
        print(f"   👥 Biography tokenizations: {bio_tokenized_file}")
        print(f"   📖 Description tokenizations: {description_tokenized_file}")
        print(f"   📋 Updated dataset: {dataset_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Tokenization pipeline failed with error: {e}")
        return False

def check_tokenization_status(dataset_file: str = None,
                             bio_tokenized_file: str = None,
                             description_tokenized_file: str = None):
    """Check and report tokenization status without running the pipeline."""
    print_header("🔍 TOKENIZATION STATUS CHECK")
    
    # Load configuration and set defaults
    config = get_config()
    bio_config = get_bio_config()
    desc_config = get_description_config()
    
    if dataset_file is None:
        dataset_file = config.dataset_file
    if bio_tokenized_file is None:
        bio_tokenized_file = bio_config['output_file']
    if description_tokenized_file is None:
        description_tokenized_file = desc_config['output_file']
    
    try:
        # Load and analyze dataset
        dataset = load_case_dataset(dataset_file)
        unique_bio_paths, unique_case_paths, stats = analyze_tokenization_requirements(dataset)
        
        # Check existing tokenizations
        tokenization_status = check_existing_tokenizations(bio_tokenized_file, description_tokenized_file,
                                                          unique_bio_paths, unique_case_paths)
        
        # Print comprehensive status
        print(f"\n📊 COMPREHENSIVE STATUS REPORT:")
        print(f"   📋 Dataset: {dataset_file}")
        print(f"   📚 Cases: {len(dataset)}")
        print(f"   👥 Unique biography files needed: {len(unique_bio_paths)}")
        print(f"   📖 Unique description files needed: {len(unique_case_paths)}")
        print(f"")
        print(f"   👥 Biography tokenizations:")
        print(f"      📁 File: {bio_tokenized_file}")
        print(f"      ✅ Exists: {tokenization_status['bio_file_exists']}")
        if tokenization_status['bio_file_exists']:
            print(f"      📊 Coverage: {tokenization_status['bio_coverage']:.1%}")
            print(f"      🔢 Tokenized files: {len(tokenization_status['tokenized_bios'])}")
        print(f"")
        print(f"   📖 Description tokenizations:")
        print(f"      📁 File: {description_tokenized_file}")
        print(f"      ✅ Exists: {tokenization_status['description_file_exists']}")
        if tokenization_status['description_file_exists']:
            print(f"      📊 Coverage: {tokenization_status['description_coverage']:.1%}")
            print(f"      🔢 Tokenized files: {len(tokenization_status['tokenized_descriptions'])}")
        
        # Determine what needs to be done
        tokenization_plan = determine_tokenization_plan(unique_bio_paths, unique_case_paths, tokenization_status)
        
        if not tokenization_plan['tokenize_bios'] and not tokenization_plan['tokenize_descriptions']:
            print(f"\n✅ All required files are already tokenized!")
        else:
            print(f"\n📝 ACTIONS NEEDED:")
            if tokenization_plan['tokenize_bios']:
                print(f"   👥 Tokenize {len(tokenization_plan['bios_to_tokenize'])} biography files")
            if tokenization_plan['tokenize_descriptions']:
                print(f"   📖 Tokenize {len(tokenization_plan['descriptions_to_tokenize'])} description files")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking tokenization status: {e}")
        return False

def main():
    # Load configuration for defaults
    config = get_config()
    bio_config = get_bio_config()
    desc_config = get_description_config()
    
    parser = argparse.ArgumentParser(
        description="SCOTUS AI Main Tokenization Pipeline - Pre-tokenize all texts for efficient training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_encoder.py                    # Run full tokenization pipeline
  python main_encoder.py --check           # Check tokenization status only
  python main_encoder.py --bios-only       # Tokenize only biography files
  python main_encoder.py --descriptions-only # Tokenize only case description files
  python main_encoder.py --force           # Force re-tokenization (overwrite existing)
  
  # Custom file locations
  python main_encoder.py --dataset data/my_dataset.json --bio-output my_bios.pkl
  
Pipeline Steps:
  1. Load case dataset and analyze requirements
  2. Check existing tokenizations for resumption
  3. Tokenize missing biography files (AutoTokenizer)
  4. Tokenize missing case description files (AutoTokenizer)
  5. Update dataset with tokenized file locations
  6. Validate final coverage

Features:
  - Smart resumption (only tokenizes missing files)
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
        help="Output file for biography tokenizations"
    )
    
    parser.add_argument(
        "--description-output", 
        default=desc_config['output_file'],
        help="Output file for case description tokenizations"
    )
    
    parser.add_argument(
        "--config",
        help="Path to configuration file (default: config.env in same directory)"
    )
    
    parser.add_argument(
        "--bios-only",
        action="store_true",
        help="Tokenize only biography files"
    )
    
    parser.add_argument(
        "--descriptions-only",
        action="store_true",
        help="Tokenize only case description files"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-tokenization of all files (overwrite existing)"
    )
    
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check tokenization status without running pipeline"
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
        success = check_tokenization_status(args.dataset, args.bio_output, args.description_output)
        sys.exit(0 if success else 1)
    
    # Run tokenization pipeline
    success = run_tokenization_pipeline(
        dataset_file=args.dataset,
        bio_tokenized_file=args.bio_output,
        description_tokenized_file=args.description_output,
        bios_only=args.bios_only,
        descriptions_only=args.descriptions_only,
        force_retokenization=args.force
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 
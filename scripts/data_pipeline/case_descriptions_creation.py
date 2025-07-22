#!/usr/bin/env python3
import pandas as pd
import os
import re
import argparse
from pathlib import Path
import sys
from typing import Dict, List, Optional

# Import existing decoding functionality
from case_metadata_creation import (
    PETITIONER_CODES, 
    STATE_CODES, 
    CASE_ORIGIN_CODES, 
    ISSUE_AREA_CODES,
    decode_value
)

# Import the Augmenter class
from augmenter import Augmenter, create_augmenter

def sanitize_filename(citation: str) -> str:
    """Convert citation to a safe filename."""
    if pd.isna(citation):
        return "unknown.txt"
    
    filename = re.sub(r'[^\w\s.-]', '', str(citation))
    filename = re.sub(r'\s+', '_', filename)
    return filename + ".txt"

def load_case_descriptions(descriptions_dir: str) -> dict:
    """Load all case descriptions from the AI-filtered directory."""
    descriptions = {}
    
    if not os.path.exists(descriptions_dir):
        print(f"❌ Descriptions directory not found: {descriptions_dir}")
        return descriptions
    
    description_files = [f for f in os.listdir(descriptions_dir) if f.endswith('.txt')]
    print(f"📁 Found {len(description_files)} description files")
    
    for filename in description_files:
        filepath = os.path.join(descriptions_dir, filename)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Extract metadata from the description file
                lines = content.split('\n')
                case_id = None
                us_cite = None
                description_text = ""
                
                # Parse header to extract case ID and citation
                for i, line in enumerate(lines):
                    if line.startswith('Case ID:'):
                        case_id = line.replace('Case ID:', '').strip()
                    elif line.startswith('Citation:'):
                        us_cite = line.replace('Citation:', '').strip()
                    elif line.startswith('===='):
                        # Everything after the separator is the actual description
                        description_text = '\n'.join(lines[i+1:]).strip()
                        break
                
                if case_id and description_text:
                    descriptions[case_id] = {
                        'citation': us_cite,
                        'text': description_text,
                        'filename': filename
                    }
                    
        except Exception as e:
            print(f"⚠️  Error reading {filename}: {e}")
            continue
    
    print(f"✅ Loaded {len(descriptions)} case descriptions")
    return descriptions

# All decoding functions are now imported from case_description_creation.py

def format_case_summary(row, description_text: str) -> str:
    """Format a complete case summary combining metadata and description."""
    
    # Extract case information using existing comprehensive decoding
    case_name = row.get('caseName', 'Unknown Case')
    citation = row.get('usCite', 'Unknown Citation')
    chief_justice = row.get('chief', 'Unknown Chief Justice')
    
    # Use existing comprehensive decoding functions
    petitioner = decode_value(row.get('petitioner'), PETITIONER_CODES, "Unknown petitioner")
    petitioner_state = decode_value(row.get('petitionerState'), STATE_CODES, "")
    respondent_state = decode_value(row.get('respondentState'), STATE_CODES, "")
    case_origin = decode_value(row.get('caseOrigin'), CASE_ORIGIN_CODES, "Unknown court")
    case_origin_state = decode_value(row.get('caseOriginState'), STATE_CODES, "")
    issue_area = decode_value(row.get('issueArea'), ISSUE_AREA_CODES, "Unknown issue area")
    
    # Build summary with comprehensive information
    summary = f"""Case: {case_name} ({citation})

This Supreme Court case was decided under Chief Justice {chief_justice}. The petitioner was {petitioner}"""
    
    # Add petitioner state if available
    if petitioner_state and petitioner_state != "Not specified":
        summary += f" from {petitioner_state}"
    
    # Add respondent state if available
    if respondent_state and respondent_state != "Not specified":
        summary += f". The respondent was from {respondent_state}"
    
    summary += f".\n\nThe case originated from {case_origin}"
    
    # Add case origin state if available
    if case_origin_state and case_origin_state != "Not specified":
        summary += f" in {case_origin_state}"
    
    summary += f". The primary issue area was {issue_area}.\n\n{description_text}"
    
    return summary

def create_case_descriptions(metadata_file: str, descriptions_dir: str, output_dir: str, 
                           verbose: bool = True, quiet: bool = False,
                           use_augmentation: bool = False, augmentation_config: Optional[Dict] = None):
    """
    Create complete case descriptions by combining metadata with AI-filtered descriptions.
    
    Args:
        metadata_file: Path to the metadata CSV file
        descriptions_dir: Directory containing AI-filtered descriptions
        output_dir: Output directory for case descriptions
        verbose: Whether to print verbose output
        quiet: Whether to minimize output
        use_augmentation: Whether to create augmented versions
        augmentation_config: Configuration for text augmentation
    """
    if verbose:
        print(f"🚀 Creating complete case descriptions...")
        print(f"📊 Metadata file: {metadata_file}")
        print(f"📁 Descriptions directory: {descriptions_dir}")
        print(f"💾 Output directory: {output_dir}")
    
    # Initialize augmenter if requested
    augmenter = None
    if use_augmentation and augmentation_config:
        try:
            augmenter = create_augmenter(
                augmentations=augmentation_config.get('augmentations', ['synonym_augmentation','summarization']),
                iterations=augmentation_config.get('iterations', 1),
                seed=augmentation_config.get('seed', 42),
                verbose=augmentation_config.get('verbose', verbose)
            )
            if verbose:
                print(f"🔧 Initialized augmenter with: {augmentation_config}")
        except Exception as e:
            print(f"⚠️  Warning: Failed to initialize augmenter: {e}")
            augmenter = None
    
    # Load case metadata
    if verbose:
        print(f"\n📊 Loading case metadata...")
    try:
        df = pd.read_csv(metadata_file)
        if verbose:
            print(f"✅ Loaded {len(df)} cases from metadata")
    except Exception as e:
        print(f"❌ Error loading metadata: {e}")
        return 0
    
    # Load case descriptions
    if verbose:
        print(f"\n📖 Loading case descriptions...")
    descriptions = load_case_descriptions(descriptions_dir)
    
    if not descriptions:
        print("❌ No descriptions found. Make sure to run the AI scraper first.")
        return 0
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    if verbose:
        print(f"📁 Created output directory: {output_dir}")
    
    # Process each case
    successful = 0
    missing_descriptions = 0
    errors = 0
    
    if not quiet:
        print(f"Processing {len(df)} cases...")
    
    for idx, row in df.iterrows():
        case_id = row.get('caseIssuesId')
        us_cite = row.get('usCite')
        case_name = row.get('caseName', 'Unknown Case')
        
        if pd.isna(case_id):
            if verbose:
                print(f"⚠️  [{idx+1}] Missing case ID for: {case_name}")
            errors += 1
            continue
        
        # Look for description by case ID
        if case_id in descriptions:
            description_data = descriptions[case_id]
            description_text = description_data['text']
            
            # Create complete case summary
            complete_summary = format_case_summary(row, description_text)
            
            # Create base filename
            if pd.notna(us_cite):
                base_filename = sanitize_filename(us_cite).replace('.txt', '')
            else:
                base_filename = f"case_{case_id}"
            
            # Save original version (version 0)
            original_filename = f"{base_filename}_v0.txt"
            original_output_path = os.path.join(output_dir, original_filename)
            
            try:
                # Save original complete summary
                with open(original_output_path, 'w', encoding='utf-8') as f:
                    f.write(complete_summary)
                
                successful += 1
                word_count = len(complete_summary.split())
                versions_created = 1
                
                # Create augmented versions if augmenter is available
                if augmenter:
                    try:
                        # Augment the complete summary
                        augmented_summaries = augmenter.augment_sentence(complete_summary)
                        
                        # Save augmented versions (skip the first one as it's the original)
                        for version_num, augmented_summary in enumerate(augmented_summaries[1:], 1):
                            if augmented_summary and augmented_summary != complete_summary:
                                augmented_filename = f"{base_filename}_v{version_num}.txt"
                                augmented_output_path = os.path.join(output_dir, augmented_filename)
                                
                                with open(augmented_output_path, 'w', encoding='utf-8') as f:
                                    f.write(augmented_summary)
                                
                                versions_created += 1
                                
                    except Exception as e:
                        if verbose:
                            print(f"⚠️  Error creating augmented versions for {case_name}: {e}")
                
                if (idx % 10 == 0 or successful <= 5) and verbose:  # Show first few and every 10th
                    if augmenter and versions_created > 1:
                        print(f"✅ [{successful:,}] {case_name} -> {original_filename} + {versions_created-1} augmented versions ({word_count:,} words)")
                    else:
                        print(f"✅ [{successful:,}] {case_name} -> {original_filename} ({word_count:,} words)")
                
            except Exception as e:
                if verbose:
                    print(f"❌ Error saving {case_name}: {e}")
                errors += 1
                continue
                
        else:
            missing_descriptions += 1
            if missing_descriptions <= 5 and verbose:  # Show first few missing
                print(f"⚠️  [{idx+1}] No description found for: {case_name} (ID: {case_id})")
    
    if not quiet:
        print(f"Created {successful}/{len(df)} complete case descriptions ({successful/len(df)*100:.1f}% success rate)")
    
    # Summary report
    if verbose:
        print(f"\n{'='*60}")
        print(f"📋 DETAILED PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Total cases in metadata: {len(df):,}")
        print(f"Available descriptions: {len(descriptions):,}")
        print(f"Successfully processed: {successful:,}")
        print(f"Missing descriptions: {missing_descriptions:,}")
        print(f"Errors: {errors:,}")
        print(f"Success rate: {successful/len(df)*100:.1f}%")
        print(f"Output directory: {output_dir}")
        
        if augmenter:
            print(f"Text augmentation enabled: {augmentation_config}")
            print(f"Augmentation techniques: {augmentation_config.get('augmentations', [])}")
            print(f"Augmentation iterations: {augmentation_config.get('iterations', 1)}")
        
        if missing_descriptions > 0:
            print(f"\n💡 To get more descriptions, run the AI scraper on missing cases:")
            print(f"   python src/data_pipeline/scraper_case_descriptions.py --limit {missing_descriptions}")
    
    return successful

def main():
    parser = argparse.ArgumentParser(
        description="Create complete case descriptions by combining metadata with AI-filtered descriptions"
    )
    parser.add_argument(
        "--metadata",
        "-m",
        default="data/processed/cases_metadata.csv",
        help="Path to cases metadata CSV file"
    )
    parser.add_argument(
        "--descriptions",
        "-d",
        default="data/external/case_descriptions_ai_filtered",
        help="Directory containing AI-filtered case descriptions"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data/processed/case_descriptions",
        help="Output directory for complete case descriptions"
    )
    
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimize output")
    
    # Augmentation arguments
    parser.add_argument("--use-augmentation", action="store_true", 
                       help="Enable text augmentation for case descriptions")
    parser.add_argument("--augmentations", nargs="+", 
                       default=["synonym_augmentation"],
                       choices=["word_embedding_augmentation", "synonym_augmentation", 
                               "back_translation", "summarization"],
                       help="Augmentation techniques to use")
    parser.add_argument("--augmentation-iterations", type=int, default=1,
                       help="Number of augmentation iterations")
    parser.add_argument("--augmentation-seed", type=int, default=42,
                       help="Random seed for augmentation")
    
    args = parser.parse_args()
    
    verbose = args.verbose and not args.quiet
    
    # Prepare augmentation config if requested
    augmentation_config = None
    if args.use_augmentation:
        augmentation_config = {
            'augmentations': args.augmentations,
            'iterations': args.augmentation_iterations,
            'seed': args.augmentation_seed,
            'verbose': verbose
        }
    
    success_count = create_case_descriptions(
        args.metadata,
        args.descriptions, 
        args.output,
        verbose=verbose,
        quiet=args.quiet,
        use_augmentation=args.use_augmentation,
        augmentation_config=augmentation_config
    )
    
    if success_count > 0 and not args.quiet:
        print(f"🎉 Successfully created {success_count:,} complete case descriptions!")
    elif success_count == 0:
        print(f"❌ No case descriptions were created. Check your input files.")

if __name__ == "__main__":
    main() 
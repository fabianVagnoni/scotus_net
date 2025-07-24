#!/usr/bin/env python3
"""
SCOTUS AI Augmentation Pipeline
==============================

Main augmentation orchestrator that creates augmented versions of the complete dataset.
This pipeline takes the processed data and creates an augmented dataset with multiple
versions of each text (original + augmented versions).

Pipeline Steps:
1. Load original case dataset from data/processed/case_dataset.json
2. Create augmented justice biographies (original + augmented versions)
3. Create augmented case descriptions (original + augmented versions)  
4. Build new augmented case dataset with file paths to augmented versions
5. Save augmented dataset to data/augmented/case_dataset.json

Usage:
    python main.py                    # Run full augmentation pipeline
    python main.py --bios-only        # Only augment justice biographies
    python main.py --descriptions-only # Only augment case descriptions
    python main.py --no-augmentation  # Copy originals only (no text augmentation)
"""

import os
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

# Add scripts to path for imports
sys.path.append('scripts/data_pipeline')

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

def load_case_dataset(dataset_file: str) -> Dict:
    """
    Load the original case dataset.
    
    Args:
        dataset_file: Path to the case dataset JSON file
        
    Returns:
        Dictionary containing the case dataset
    """
    print(f"📖 Loading case dataset from: {dataset_file}")
    
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
    
    with open(dataset_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"✅ Loaded {len(dataset)} cases from dataset")
    return dataset

def extract_unique_file_paths(dataset: Dict) -> Tuple[set, set]:
    """
    Extract unique file paths for bios and case descriptions from the dataset.
    
    Args:
        dataset: The case dataset dictionary
        
    Returns:
        Tuple of (unique_bio_paths, unique_case_paths)
    """
    unique_bio_paths = set()
    unique_case_paths = set()
    
    for case_id, case_data in dataset.items():
        # Extract bio paths (first element is list of bio paths)
        bio_paths = case_data[0]
        for bio_path in bio_paths:
            if bio_path and bio_path != 'nan':
                unique_bio_paths.add(bio_path)
        
        # Extract case description path (second element)
        case_path = case_data[1]
        if case_path and case_path != 'nan':
            unique_case_paths.add(case_path)
    
    print(f"📊 Found {len(unique_bio_paths)} unique justice bio files")
    print(f"📊 Found {len(unique_case_paths)} unique case description files")
    
    return unique_bio_paths, unique_case_paths

def create_augmented_bios_step(bios_dir: str, output_dir: str, 
                             use_augmentation: bool = True,
                             augmentation_config: Optional[Dict] = None,
                             verbose: bool = True) -> bool:
    """
    Step 1: Create augmented justice biographies.
    """
    print_step(1, 4, "Creating Augmented Justice Biographies")
    
    try:
        from .justice_bios_augmentation import create_augmented_bios
        
        success_count = create_augmented_bios(
            bios_dir=bios_dir,
            output_dir=output_dir,
            verbose=verbose,
            quiet=False,
            use_augmentation=use_augmentation,
            augmentation_config=augmentation_config
        )
        
        if success_count > 0:
            print(f"✅ Successfully created {success_count} augmented justice biographies")
            return True
        else:
            print(f"❌ Failed to create augmented justice biographies")
            return False
            
    except Exception as e:
        print(f"❌ Error in justice bios augmentation: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_augmented_case_descriptions_step(descriptions_dir: str, output_dir: str,
                                          use_augmentation: bool = True,
                                          augmentation_config: Optional[Dict] = None,
                                          verbose: bool = True) -> bool:
    """
    Step 2: Create augmented case descriptions.
    """
    print_step(2, 4, "Creating Augmented Case Descriptions")
    
    try:
        from .case_descriptions_augmentation import create_augmented_case_descriptions
        
        success_count = create_augmented_case_descriptions(
            descriptions_dir=descriptions_dir,
            output_dir=output_dir,
            verbose=verbose,
            quiet=False,
            use_augmentation=use_augmentation,
            augmentation_config=augmentation_config
        )
        
        if success_count > 0:
            print(f"✅ Successfully created {success_count} augmented case descriptions")
            return True
        else:
            print(f"❌ Failed to create augmented case descriptions")
            return False
            
    except Exception as e:
        print(f"❌ Error in case descriptions augmentation: {e}")
        import traceback
        traceback.print_exc()
        return False

def build_augmented_dataset_step(original_dataset: Dict, 
                               augmented_bios_dir: str,
                               augmented_descriptions_dir: str,
                               output_file: str,
                               verbose: bool = True) -> bool:
    """
    Step 3: Build the augmented case dataset with file paths to augmented versions.
    """
    print_step(3, 4, "Building Augmented Case Dataset")
    
    try:
        augmented_dataset = {}
        
        # Track statistics
        total_cases = len(original_dataset)
        cases_with_bios = 0
        cases_with_descriptions = 0
        total_augmented_versions = 0
        
        for case_id, case_data in original_dataset.items():
            bio_paths = case_data[0]
            case_description_path = case_data[1]
            voting_percentages = case_data[2]
            
            # Find augmented bio paths
            augmented_bio_paths = []
            for bio_path in bio_paths:
                if bio_path and bio_path != 'nan':
                    # Extract justice name from path
                    justice_name = os.path.basename(bio_path).replace('.txt', '')
                    
                    # Find all augmented versions
                    base_filename = justice_name
                    augmented_versions = []
                    
                    # Check for v0 (original) and augmented versions
                    for version in range(10):  # Check up to v9
                        version_filename = f"{base_filename}_v{version}.txt"
                        version_path = os.path.join(augmented_bios_dir, version_filename)
                        
                        if os.path.exists(version_path):
                            # Convert to relative path for the augmented dataset
                            relative_path = os.path.join("data/augmented/bios", version_filename)
                            augmented_versions.append(relative_path)
                    
                    if augmented_versions:
                        augmented_bio_paths.extend(augmented_versions)
                        cases_with_bios += 1
            
            # Find augmented case description path
            augmented_case_path = None
            if case_description_path and case_description_path != 'nan':
                # Extract case ID from path
                case_filename = os.path.basename(case_description_path).replace('.txt', '')
                
                # Check for v0 (original) version
                version_filename = f"{case_filename}_v0.txt"
                version_path = os.path.join(augmented_descriptions_dir, version_filename)
                
                if os.path.exists(version_path):
                    # Convert to relative path for the augmented dataset
                    augmented_case_path = os.path.join("data/augmented/case_descriptions", version_filename)
                    cases_with_descriptions += 1
            
            # Build augmented case entry
            # Format: case_id: [augmented_bio_paths, augmented_case_path, voting_percentages]
            augmented_dataset[case_id] = [
                augmented_bio_paths,
                augmented_case_path,
                voting_percentages
            ]
            
            total_augmented_versions += len(augmented_bio_paths)
            
            if verbose and (len(augmented_dataset) % 1000 == 0 or len(augmented_dataset) <= 5):
                print(f"✅ Processed {len(augmented_dataset):,}/{total_cases:,} cases...")
        
        # Save the augmented dataset
        print(f"💾 Saving augmented dataset to: {output_file}")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(augmented_dataset, f, indent=2, ensure_ascii=False)
        
        # Print statistics
        print(f"\n📊 AUGMENTED DATASET STATISTICS:")
        print(f"   Total cases: {total_cases:,}")
        print(f"   Cases with augmented bios: {cases_with_bios:,}")
        print(f"   Cases with augmented descriptions: {cases_with_descriptions:,}")
        print(f"   Total augmented bio versions: {total_augmented_versions:,}")
        print(f"   Average bio versions per case: {total_augmented_versions/total_cases:.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error building augmented dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_augmented_dataset_step(augmented_dataset_file: str, verbose: bool = True) -> bool:
    """
    Step 4: Validate the augmented dataset.
    """
    print_step(4, 4, "Validating Augmented Dataset")
    
    try:
        with open(augmented_dataset_file, 'r', encoding='utf-8') as f:
            augmented_dataset = json.load(f)
        
        # Validation checks
        total_cases = len(augmented_dataset)
        cases_with_bios = 0
        cases_with_descriptions = 0
        total_bio_files = 0
        total_description_files = 0
        
        for case_id, case_data in augmented_dataset.items():
            bio_paths = case_data[0]
            case_description_path = case_data[1]
            voting_percentages = case_data[2]
            
            if bio_paths:
                cases_with_bios += 1
                total_bio_files += len(bio_paths)
            
            if case_description_path:
                cases_with_descriptions += 1
                total_description_files += 1
        
        print(f"✅ Validation completed successfully!")
        print(f"📊 Validation Results:")
        print(f"   Total cases in augmented dataset: {total_cases:,}")
        print(f"   Cases with bio files: {cases_with_bios:,} ({cases_with_bios/total_cases*100:.1f}%)")
        print(f"   Cases with description files: {cases_with_descriptions:,} ({cases_with_descriptions/total_cases*100:.1f}%)")
        print(f"   Total bio file references: {total_bio_files:,}")
        print(f"   Total description file references: {total_description_files:,}")
        
        # Check file existence
        missing_bio_files = 0
        missing_description_files = 0
        
        for case_id, case_data in augmented_dataset.items():
            bio_paths = case_data[0]
            case_description_path = case_data[1]
            
            for bio_path in bio_paths:
                if not os.path.exists(bio_path):
                    missing_bio_files += 1
            
            if case_description_path and not os.path.exists(case_description_path):
                missing_description_files += 1
        
        if missing_bio_files > 0 or missing_description_files > 0:
            print(f"⚠️  Warnings:")
            if missing_bio_files > 0:
                print(f"   Missing bio files: {missing_bio_files}")
            if missing_description_files > 0:
                print(f"   Missing description files: {missing_description_files}")
        else:
            print(f"✅ All referenced files exist!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error validating augmented dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_augmentation_pipeline(
    input_dataset: str = "data/processed/case_dataset.json",
    output_dataset: str = "data/augmented/case_dataset.json",
    bios_dir: str = "data/processed/bios",
    descriptions_dir: str = "data/processed/case_descriptions",
    augmented_bios_dir: str = "data/augmented/bios",
    augmented_descriptions_dir: str = "data/augmented/case_descriptions",
    use_augmentation: bool = True,
    augmentation_config: Optional[Dict] = None,
    bios_only: bool = False,
    descriptions_only: bool = False,
    verbose: bool = True
) -> bool:
    """
    Run the complete augmentation pipeline.
    
    Args:
        input_dataset: Path to original case dataset
        output_dataset: Path to output augmented case dataset
        bios_dir: Directory containing original justice bios
        descriptions_dir: Directory containing original case descriptions
        augmented_bios_dir: Output directory for augmented bios
        augmented_descriptions_dir: Output directory for augmented descriptions
        use_augmentation: Whether to use text augmentation
        augmentation_config: Configuration for text augmentation
        bios_only: Only process justice bios
        descriptions_only: Only process case descriptions
        verbose: Whether to print verbose output
    
    Returns:
        True if successful, False otherwise
    """
    print_header("🚀 SCOTUS AI AUGMENTATION PIPELINE", "=")
    
    start_time = time.time()
    
    try:
        # Load original dataset
        print(f"📖 Loading original dataset from: {input_dataset}")
        original_dataset = load_case_dataset(input_dataset)
        
        # Extract unique file paths
        unique_bio_paths, unique_case_paths = extract_unique_file_paths(original_dataset)
        
        # Set default augmentation config if not provided
        if augmentation_config is None and use_augmentation:
            augmentation_config = {
                'augmentations': ['synonym_augmentation', 'back_translation', 'summarization'],
                'iterations': 3,
                'seed': 42,
                'verbose': verbose,
                'random_selection_prob': 0.5
            }
        
        # Step 1: Create augmented justice biographies
        if not descriptions_only:
            success = create_augmented_bios_step(
                bios_dir=bios_dir,
                output_dir=augmented_bios_dir,
                use_augmentation=use_augmentation,
                augmentation_config=augmentation_config,
                verbose=verbose
            )
            if not success:
                print("❌ Justice bios augmentation failed. Stopping pipeline.")
                return False
        
        # Step 2: Create augmented case descriptions
        if not bios_only:
            success = create_augmented_case_descriptions_step(
                descriptions_dir=descriptions_dir,
                output_dir=augmented_descriptions_dir,
                use_augmentation=use_augmentation,
                augmentation_config=augmentation_config,
                verbose=verbose
            )
            if not success:
                print("❌ Case descriptions augmentation failed. Stopping pipeline.")
                return False
        
        # Step 3: Build augmented dataset
        if not (bios_only or descriptions_only):
            success = build_augmented_dataset_step(
                original_dataset=original_dataset,
                augmented_bios_dir=augmented_bios_dir,
                augmented_descriptions_dir=augmented_descriptions_dir,
                output_file=output_dataset,
                verbose=verbose
            )
            if not success:
                print("❌ Building augmented dataset failed. Stopping pipeline.")
                return False
            
            # Step 4: Validate augmented dataset
            success = validate_augmented_dataset_step(
                augmented_dataset_file=output_dataset,
                verbose=verbose
            )
            if not success:
                print("❌ Dataset validation failed.")
                return False
        
        # Pipeline completion
        total_time = time.time() - start_time
        
        print_header("🎉 AUGMENTATION PIPELINE COMPLETED SUCCESSFULLY!", "=")
        print(f"⏱️  Total execution time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        
        # Show output summary
        output_locations = [
            (augmented_bios_dir, "Augmented justice biographies"),
            (augmented_descriptions_dir, "Augmented case descriptions"),
            (output_dataset, "Augmented case dataset")
        ]
        
        print(f"\n📋 Generated outputs:")
        for output_path, description in output_locations:
            if os.path.exists(output_path):
                if os.path.isdir(output_path):
                    count = len([f for f in os.listdir(output_path) if f.endswith('.txt')])
                    print(f"   ✅ {description}: {output_path} ({count} files)")
                else:
                    size_mb = os.path.getsize(output_path) / (1024 * 1024)
                    print(f"   ✅ {description}: {output_path} ({size_mb:.1f} MB)")
            else:
                print(f"   ⚪ {description}: {output_path} (not created)")
        
        print(f"\n💡 Your augmented SCOTUS AI dataset is ready!")
        print(f"   🎯 Augmented dataset file: {output_dataset}")
        print(f"   📊 All augmented data preserved in data/augmented/ directories")
        
        return True
        
    except Exception as e:
        total_time = time.time() - start_time
        print(f"\n❌ Augmentation pipeline failed after {total_time:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(
        description="SCOTUS AI Augmentation Pipeline - Create augmented versions of the complete dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Run full augmentation pipeline
  python main.py --bios-only                        # Only augment justice biographies
  python main.py --descriptions-only                # Only augment case descriptions
  python main.py --no-augmentation                  # Copy originals only (no text augmentation)
  python main.py --augmentation-iterations 5        # Use 5 augmentation iterations
  python main.py --augmentations word_embedding_augmentation synonym_augmentation  # Use specific techniques
  python main.py --random-selection-prob 0.7        # 70% chance of selecting each augmentation
  
Augmentation Techniques:
  - word_embedding_augmentation: Replace words with similar embeddings
  - synonym_augmentation: Replace words with synonyms
  - back_translation: Translate to another language and back
  - summarization: Create summary versions

Requirements:
  - Python packages: nlpaug, transformers, sentence-transformers
  - Original processed dataset must exist in data/processed/
"""
    )
    
    parser.add_argument(
        "--input-dataset",
        default="data/processed/case_dataset.json",
        help="Path to original case dataset"
    )
    parser.add_argument(
        "--output-dataset",
        default="data/augmented/case_dataset.json",
        help="Path to output augmented case dataset"
    )
    parser.add_argument(
        "--bios-dir",
        default="data/processed/bios",
        help="Directory containing original justice bios"
    )
    parser.add_argument(
        "--descriptions-dir",
        default="data/processed/case_descriptions",
        help="Directory containing original case descriptions"
    )
    parser.add_argument(
        "--augmented-bios-dir",
        default="data/augmented/bios",
        help="Output directory for augmented bios"
    )
    parser.add_argument(
        "--augmented-descriptions-dir",
        default="data/augmented/case_descriptions",
        help="Output directory for augmented descriptions"
    )
    
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimize output")
    
    # Processing options
    parser.add_argument("--bios-only", action="store_true", help="Only process justice biographies")
    parser.add_argument("--descriptions-only", action="store_true", help="Only process case descriptions")
    
    # Augmentation options
    parser.add_argument("--no-augmentation", action="store_true", 
                       help="Copy originals only (no text augmentation)")
    parser.add_argument("--augmentations", nargs="+", 
                       default=["word_embedding_augmentation"],
                       choices=["word_embedding_augmentation", "synonym_augmentation", 
                               "back_translation", "summarization"],
                       help="Augmentation techniques to use")
    parser.add_argument("--augmentation-iterations", type=int, default=3,
                       help="Number of augmentation iterations")
    parser.add_argument("--augmentation-seed", type=int, default=42,
                       help="Random seed for augmentation")
    parser.add_argument("--random-selection-prob", type=float, default=0.5,
                       help="Probability of selecting each augmentation in each iteration (0.0-1.0)")
    
    args = parser.parse_args()
    
    # Validate mutually exclusive arguments
    if args.bios_only and args.descriptions_only:
        print("❌ Error: Cannot use --bios-only and --descriptions-only together")
        return False
    
    verbose = args.verbose and not args.quiet
    use_augmentation = not args.no_augmentation
    
    # Prepare augmentation config
    augmentation_config = None
    if use_augmentation:
        augmentation_config = {
            'augmentations': args.augmentations,
            'iterations': args.augmentation_iterations,
            'seed': args.augmentation_seed,
            'verbose': verbose,
            'random_selection_prob': args.random_selection_prob
        }
    
    # Run the pipeline
    success = run_augmentation_pipeline(
        input_dataset=args.input_dataset,
        output_dataset=args.output_dataset,
        bios_dir=args.bios_dir,
        descriptions_dir=args.descriptions_dir,
        augmented_bios_dir=args.augmented_bios_dir,
        augmented_descriptions_dir=args.augmented_descriptions_dir,
        use_augmentation=use_augmentation,
        augmentation_config=augmentation_config,
        bios_only=args.bios_only,
        descriptions_only=args.descriptions_only,
        verbose=verbose
    )
    
    if success:
        print("🎉 Augmentation pipeline completed successfully!")
        sys.exit(0)
    else:
        print("❌ Augmentation pipeline failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 
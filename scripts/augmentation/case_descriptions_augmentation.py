#!/usr/bin/env python3
"""
Case Descriptions Augmentation Script

This script loads case descriptions from data/processed/case_descriptions and creates
augmented versions using the Augmenter class. Each case description will have
multiple versions (v0 for original, v1, v2, etc. for augmented versions).
"""

import os
import re
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import sys

# Import the Augmenter class
sys.path.append('scripts/augmentation')
from .augmenter import Augmenter, create_augmenter

def load_case_descriptions(descriptions_dir: str) -> Dict[str, str]:
    """
    Load all case descriptions from the descriptions directory.
    
    Args:
        descriptions_dir: Directory containing case description files
        
    Returns:
        Dictionary mapping case filenames to their content
    """
    descriptions = {}
    
    if not os.path.exists(descriptions_dir):
        print(f"❌ Case descriptions directory not found: {descriptions_dir}")
        return descriptions
    
    description_files = [f for f in os.listdir(descriptions_dir) if f.endswith('.txt')]
    print(f"📁 Found {len(description_files)} case description files")
    
    for filename in description_files:
        filepath = os.path.join(descriptions_dir, filename)
        case_id = filename.replace('.txt', '')
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
            if content:
                descriptions[case_id] = content
                
        except Exception as e:
            print(f"⚠️  Error reading {filename}: {e}")
            continue
    
    print(f"✅ Loaded {len(descriptions)} case descriptions")
    return descriptions

def sanitize_filename(case_id: str) -> str:
    """
    Convert case ID to a safe filename.
    
    Args:
        case_id: ID of the case
        
    Returns:
        Safe filename string
    """
    # Replace spaces and special characters with underscores
    filename = re.sub(r'[^\w\s.-]', '', case_id)
    filename = re.sub(r'\s+', '_', filename)
    return filename

def create_augmented_case_descriptions(descriptions_dir: str, output_dir: str, 
                                     verbose: bool = True, quiet: bool = False,
                                     use_augmentation: bool = False, 
                                     augmentation_config: Optional[Dict] = None):
    """
    Create augmented versions of case descriptions.
    
    Args:
        descriptions_dir: Directory containing original case descriptions
        output_dir: Output directory for augmented descriptions
        verbose: Whether to print verbose output
        quiet: Whether to minimize output
        use_augmentation: Whether to create augmented versions
        augmentation_config: Configuration for text augmentation
    """
    if verbose:
        print(f"🚀 Creating augmented case descriptions...")
        print(f"📁 Descriptions directory: {descriptions_dir}")
        print(f"💾 Output directory: {output_dir}")
    
    # Initialize augmenter if requested
    augmenter = None
    if use_augmentation and augmentation_config:
        try:
            if verbose:
                print(f"🔧 Creating augmenter with config: {augmentation_config}")
            
            augmenter = create_augmenter(
                augmentations=augmentation_config.get('augmentations', ['word_embedding_augmentation','synonym_augmentation','back_translation']),
                iterations=augmentation_config.get('iterations', 3),
                seed=augmentation_config.get('seed', 42),
                verbose=augmentation_config.get('verbose', verbose),
                random_selection_prob=augmentation_config.get('random_selection_prob', 0.5)
            )
            if verbose:
                print(f"🔧 Successfully initialized augmenter with: {augmentation_config}")
                print(f"🔧 Augmenter augmentations: {augmenter.augmentations}")
                print(f"🔧 Augmenter iterations: {augmenter.iterations}")
        except Exception as e:
            print(f"⚠️  Warning: Failed to initialize augmenter: {e}")
            import traceback
            traceback.print_exc()
            augmenter = None
    
    # Load case descriptions
    if verbose:
        print(f"\n📖 Loading case descriptions...")
    descriptions = load_case_descriptions(descriptions_dir)
    
    if not descriptions:
        print("❌ No case descriptions found. Check the descriptions directory.")
        return 0
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    if verbose:
        print(f"📁 Created output directory: {output_dir}")
    
    # Process each case description
    successful = 0
    errors = 0
    total_versions = 0
    
    if not quiet:
        print(f"Processing {len(descriptions)} case descriptions...")
    
    for case_id, description_content in descriptions.items():
        try:
            # Create base filename
            base_filename = sanitize_filename(case_id)
            
            # Save original version (version 0)
            original_filename = f"{base_filename}_v0.txt"
            original_output_path = os.path.join(output_dir, original_filename)
            
            # Save original description
            with open(original_output_path, 'w', encoding='utf-8') as f:
                f.write(description_content)
            
            successful += 1
            word_count = len(description_content.split())
            versions_created = 1
            
            # Create augmented versions if augmenter is available
            if augmenter:
                try:
                    # Augment the description content
                    if verbose:
                        print(f"🔧 Augmenting {case_id} with {len(augmentation_config.get('augmentations', []))} techniques, {augmentation_config.get('iterations', 1)} iterations")
                    
                    augmented_descriptions = augmenter.augment_sentence(description_content)
                    
                    if verbose:
                        print(f"🔧 Augmenter returned {len(augmented_descriptions)} versions for {case_id}")
                        print(f"🔧 Original length: {len(description_content)}, Augmented versions: {[len(desc) for desc in augmented_descriptions]}")
                    
                    # Save augmented versions (skip the first one as it's the original)
                    for version_num, augmented_description in enumerate(augmented_descriptions[1:], 1):
                        if augmented_description and augmented_description != description_content:
                            augmented_filename = f"{base_filename}_v{version_num}.txt"
                            augmented_output_path = os.path.join(output_dir, augmented_filename)
                            
                            with open(augmented_output_path, 'w', encoding='utf-8') as f:
                                f.write(augmented_description)
                            
                            versions_created += 1
                            if verbose:
                                print(f"🔧 Created {augmented_filename} (length: {len(augmented_description)})")
                        else:
                            if verbose:
                                print(f"🔧 Skipped version {version_num} - same as original or empty")
                            
                except Exception as e:
                    if verbose:
                        print(f"⚠️  Error creating augmented versions for {case_id}: {e}")
                        import traceback
                        traceback.print_exc()
            
            total_versions += versions_created
            
            if (successful % 100 == 0 or successful <= 5) and verbose:  # Show first few and every 100th
                if augmenter and versions_created > 1:
                    print(f"✅ [{successful:,}] {case_id} -> {original_filename} + {versions_created-1} augmented versions ({word_count:,} words)")
                else:
                    print(f"✅ [{successful:,}] {case_id} -> {original_filename} ({word_count:,} words)")
            
        except Exception as e:
            if verbose:
                print(f"❌ Error processing {case_id}: {e}")
            errors += 1
            continue
    
    if not quiet:
        print(f"Created {successful}/{len(descriptions)} case descriptions ({successful/len(descriptions)*100:.1f}% success rate)")
        if augmenter:
            print(f"Total versions created: {total_versions} (including {successful} originals)")
    
    # Summary report
    if verbose:
        print(f"\n{'='*60}")
        print(f"📋 CASE DESCRIPTIONS AUGMENTATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total case descriptions found: {len(descriptions):,}")
        print(f"Successfully processed: {successful:,}")
        print(f"Errors: {errors:,}")
        print(f"Success rate: {successful/len(descriptions)*100:.1f}%")
        print(f"Output directory: {output_dir}")
        
        if augmenter:
            print(f"Text augmentation enabled: {augmentation_config}")
            print(f"Augmentation techniques: {augmentation_config.get('augmentations', [])}")
            print(f"Augmentation iterations: {augmentation_config.get('iterations', 1)}")
            print(f"Total versions created: {total_versions:,}")
            print(f"Average versions per case: {total_versions/successful:.1f}")
        
        # Show some example files
        if successful > 0:
            print(f"\n📄 Example output files:")
            example_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.txt')])[:5]
            for filename in example_files:
                print(f"   {filename}")
            if len(os.listdir(output_dir)) > 5:
                print(f"   ... and {len(os.listdir(output_dir)) - 5} more files")
    
    return successful

def main():
    parser = argparse.ArgumentParser(
        description="Create augmented versions of case descriptions using text augmentation"
    )
    parser.add_argument(
        "--descriptions-dir",
        "-d",
        default="data/processed/case_descriptions",
        help="Directory containing case description files"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data/augmented/case_descriptions",
        help="Output directory for augmented case descriptions"
    )
    
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimize output")
    
    # Augmentation arguments
    parser.add_argument("--use-augmentation", action="store_true", 
                       help="Enable text augmentation for case descriptions")
    parser.add_argument("--augmentations", nargs="+", 
                       default=["word_embedding_augmentation"],
                       choices=["word_embedding_augmentation", "synonym_augmentation", 
                               "back_translation", "summarization"],
                       help="Augmentation techniques to use")
    parser.add_argument("--augmentation-iterations", type=int, default=3,
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
    
    success_count = create_augmented_case_descriptions(
        descriptions_dir=args.descriptions_dir,
        output_dir=args.output,
        verbose=verbose,
        quiet=args.quiet,
        use_augmentation=args.use_augmentation,
        augmentation_config=augmentation_config
    )
    
    if success_count > 0 and not args.quiet:
        print(f"🎉 Successfully processed {success_count:,} case descriptions!")
        if args.use_augmentation:
            print(f"📁 Augmented versions saved to: {args.output}")
    elif success_count == 0:
        print(f"❌ No case descriptions were processed. Check your input directory.")

if __name__ == "__main__":
    main() 
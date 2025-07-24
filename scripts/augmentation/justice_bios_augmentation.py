#!/usr/bin/env python3
"""
Justice Bios Augmentation Script

This script loads justice biographies from data/processed/bios and creates
augmented versions using the Augmenter class. Each justice bio will have
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

def load_justice_bios(bios_dir: str) -> Dict[str, str]:
    """
    Load all justice biographies from the bios directory.
    
    Args:
        bios_dir: Directory containing justice bio files
        
    Returns:
        Dictionary mapping justice names to their bio content
    """
    bios = {}
    
    if not os.path.exists(bios_dir):
        print(f"❌ Bios directory not found: {bios_dir}")
        return bios
    
    bio_files = [f for f in os.listdir(bios_dir) if f.endswith('.txt')]
    print(f"📁 Found {len(bio_files)} justice bio files")
    
    for filename in bio_files:
        filepath = os.path.join(bios_dir, filename)
        justice_name = filename.replace('.txt', '')
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
            if content:
                bios[justice_name] = content
                
        except Exception as e:
            print(f"⚠️  Error reading {filename}: {e}")
            continue
    
    print(f"✅ Loaded {len(bios)} justice biographies")
    return bios

def sanitize_filename(justice_name: str) -> str:
    """
    Convert justice name to a safe filename.
    
    Args:
        justice_name: Name of the justice
        
    Returns:
        Safe filename string
    """
    # Replace spaces and special characters with underscores
    filename = re.sub(r'[^\w\s.-]', '', justice_name)
    filename = re.sub(r'\s+', '_', filename)
    return filename

def create_augmented_bios(bios_dir: str, output_dir: str, 
                         verbose: bool = True, quiet: bool = False,
                         use_augmentation: bool = False, 
                         augmentation_config: Optional[Dict] = None):
    """
    Create augmented versions of justice biographies.
    
    Args:
        bios_dir: Directory containing original justice bios
        output_dir: Output directory for augmented bios
        verbose: Whether to print verbose output
        quiet: Whether to minimize output
        use_augmentation: Whether to create augmented versions
        augmentation_config: Configuration for text augmentation
    """
    if verbose:
        print(f"🚀 Creating augmented justice biographies...")
        print(f"📁 Bios directory: {bios_dir}")
        print(f"💾 Output directory: {output_dir}")
    
    # Initialize augmenter if requested
    augmenter = None
    if use_augmentation and augmentation_config:
        try:
            if verbose:
                print(f"🔧 Creating augmenter with config: {augmentation_config}")
            
            augmenter = create_augmenter(
                augmentations=augmentation_config.get('augmentations', ['synonym_augmentation', 'word_embedding_augmentation', 'back_translation', 'summarization']),
                iterations=augmentation_config.get('iterations', 2),
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
    
    # Load justice bios
    if verbose:
        print(f"\n📖 Loading justice biographies...")
    bios = load_justice_bios(bios_dir)
    
    if not bios:
        print("❌ No justice bios found. Check the bios directory.")
        return 0
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    if verbose:
        print(f"📁 Created output directory: {output_dir}")
    
    # Process each justice bio
    successful = 0
    errors = 0
    total_versions = 0
    
    if not quiet:
        print(f"Processing {len(bios)} justice biographies...")
    
    for justice_name, bio_content in bios.items():
        try:
            # Create base filename
            base_filename = sanitize_filename(justice_name)
            
            # Save original version (version 0)
            original_filename = f"{base_filename}_v0.txt"
            original_output_path = os.path.join(output_dir, original_filename)
            
            # Save original bio
            with open(original_output_path, 'w', encoding='utf-8') as f:
                f.write(bio_content)
            
            successful += 1
            word_count = len(bio_content.split())
            versions_created = 1
            
            # Create augmented versions if augmenter is available
            if augmenter:
                try:
                    # Augment the bio content
                    if verbose:
                        print(f"🔧 Augmenting {justice_name} with {len(augmentation_config.get('augmentations', []))} techniques, {augmentation_config.get('iterations', 1)} iterations")
                    
                    augmented_bios = augmenter.augment_sentence(bio_content)
                    
                    if verbose:
                        print(f"🔧 Augmenter returned {len(augmented_bios)} versions for {justice_name}")
                        print(f"🔧 Original length: {len(bio_content)}, Augmented versions: {[len(bio) for bio in augmented_bios]}")
                    
                    # Save augmented versions (skip the first one as it's the original)
                    for version_num, augmented_bio in enumerate(augmented_bios[1:], 1):
                        if augmented_bio and augmented_bio != bio_content:
                            augmented_filename = f"{base_filename}_v{version_num}.txt"
                            augmented_output_path = os.path.join(output_dir, augmented_filename)
                            
                            with open(augmented_output_path, 'w', encoding='utf-8') as f:
                                f.write(augmented_bio)
                            
                            versions_created += 1
                            if verbose:
                                print(f"🔧 Created {augmented_filename} (length: {len(augmented_bio)})")
                        else:
                            if verbose:
                                print(f"🔧 Skipped version {version_num} - same as original or empty")
                            
                except Exception as e:
                    if verbose:
                        print(f"⚠️  Error creating augmented versions for {justice_name}: {e}")
                        import traceback
                        traceback.print_exc()
            
            total_versions += versions_created
            
            if (successful % 10 == 0 or successful <= 5) and verbose:  # Show first few and every 10th
                if augmenter and versions_created > 1:
                    print(f"✅ [{successful:,}] {justice_name} -> {original_filename} + {versions_created-1} augmented versions ({word_count:,} words)")
                else:
                    print(f"✅ [{successful:,}] {justice_name} -> {original_filename} ({word_count:,} words)")
            
        except Exception as e:
            if verbose:
                print(f"❌ Error processing {justice_name}: {e}")
            errors += 1
            continue
    
    if not quiet:
        print(f"Created {successful}/{len(bios)} justice biographies ({successful/len(bios)*100:.1f}% success rate)")
        if augmenter:
            print(f"Total versions created: {total_versions} (including {successful} originals)")
    
    # Summary report
    if verbose:
        print(f"\n{'='*60}")
        print(f"📋 JUSTICE BIOS AUGMENTATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total justice bios found: {len(bios):,}")
        print(f"Successfully processed: {successful:,}")
        print(f"Errors: {errors:,}")
        print(f"Success rate: {successful/len(bios)*100:.1f}%")
        print(f"Output directory: {output_dir}")
        
        if augmenter:
            print(f"Text augmentation enabled: {augmentation_config}")
            print(f"Augmentation techniques: {augmentation_config.get('augmentations', [])}")
            print(f"Augmentation iterations: {augmentation_config.get('iterations', 1)}")
            print(f"Total versions created: {total_versions:,}")
            print(f"Average versions per justice: {total_versions/successful:.1f}")
        
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
        description="Create augmented versions of justice biographies using text augmentation"
    )
    parser.add_argument(
        "--bios-dir",
        "-b",
        default="data/processed/bios",
        help="Directory containing justice biography files"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data/processed/bios_augmented",
        help="Output directory for augmented justice biographies"
    )
    
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimize output")
    
    # Augmentation arguments
    parser.add_argument("--use-augmentation", action="store_true", 
                       help="Enable text augmentation for justice bios")
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
    
    success_count = create_augmented_bios(
        bios_dir=args.bios_dir,
        output_dir=args.output,
        verbose=verbose,
        quiet=args.quiet,
        use_augmentation=args.use_augmentation,
        augmentation_config=augmentation_config
    )
    
    if success_count > 0 and not args.quiet:
        print(f"🎉 Successfully processed {success_count:,} justice biographies!")
        if args.use_augmentation:
            print(f"📁 Augmented versions saved to: {args.output}")
    elif success_count == 0:
        print(f"❌ No justice biographies were processed. Check your input directory.")

if __name__ == "__main__":
    main() 
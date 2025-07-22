#!/usr/bin/env python3
"""
Test script to demonstrate justice bio augmentation functionality.

This script shows how to use the justice_bios_augmentation.py with sample data
and how the output files are structured.
"""

import os
import sys
from pathlib import Path

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_test_environment():
    """Set up test environment using real justice bios."""
    
    # Create test directories
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test")
    output_dir = os.path.join(test_dir, "output")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Use real bios directory
    real_bios_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "processed", "bios")
    
    if not os.path.exists(real_bios_dir):
        print(f"❌ Real bios directory not found: {real_bios_dir}")
        print("Please make sure you have run the bios scraping process first.")
        return None, None, None
    
    return test_dir, real_bios_dir, output_dir

def test_without_augmentation(bios_dir, output_dir):
    """Test justice bio processing without augmentation."""
    print("=" * 60)
    print("TESTING WITHOUT AUGMENTATION")
    print("=" * 60)
    
    from justice_bios_augmentation import create_augmented_bios
    
    success_count = create_augmented_bios(
        bios_dir=bios_dir,
        output_dir=output_dir,
        verbose=True,
        quiet=False,
        use_augmentation=False
    )
    
    print(f"\nCreated {success_count} justice bios without augmentation")
    
    # List output files
    output_files = sorted(os.listdir(output_dir))
    print(f"Output files: {output_files}")
    
    return output_files

def test_with_augmentation(bios_dir, output_dir, test_subset=True):
    """Test justice bio processing with augmentation."""
    print("\n" + "=" * 60)
    print("TESTING WITH AUGMENTATION")
    print("=" * 60)
    
    from justice_bios_augmentation import create_augmented_bios
    
    # Use word embedding augmentation for reliable results
    augmentation_config = {
        'augmentations': ['word_embedding_augmentation'],
        'iterations': 3,
        'seed': 42,
        'verbose': True
    }
    
    if test_subset:
        # Test with a subset of justices for faster processing
        print("🧪 Testing with a subset of justices for demonstration...")
        
        # Create a temporary subset directory
        import tempfile
        import shutil
        
        temp_dir = tempfile.mkdtemp()
        subset_bios_dir = os.path.join(temp_dir, "subset_bios")
        os.makedirs(subset_bios_dir, exist_ok=True)
        
        # Copy a few representative justice bios
        test_justices = ['John_Roberts', 'Sonia_Sotomayor', 'Clarence_Thomas', 'Elena_Kagan']
        copied_count = 0
        
        for justice in test_justices:
            source_file = os.path.join(bios_dir, f"{justice}.txt")
            if os.path.exists(source_file):
                dest_file = os.path.join(subset_bios_dir, f"{justice}.txt")
                shutil.copy2(source_file, dest_file)
                copied_count += 1
        
        if copied_count == 0:
            print("⚠️  No test justice files found, using all bios")
            subset_bios_dir = bios_dir
        else:
            print(f"📋 Using subset of {copied_count} justices: {test_justices[:copied_count]}")
            bios_dir = subset_bios_dir
    
    success_count = create_augmented_bios(
        bios_dir=bios_dir,
        output_dir=output_dir,
        verbose=True,
        quiet=False,
        use_augmentation=True,
        augmentation_config=augmentation_config
    )
    
    print(f"\nCreated {success_count} justice bios with augmentation")
    
    # List output files
    output_files = sorted(os.listdir(output_dir))
    print(f"Output files: {len(output_files)} total files")
    
    # Clean up temporary directory if used
    if test_subset and 'temp_dir' in locals():
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
    
    return output_files

def test_with_all_justices(bios_dir, output_dir):
    """Test justice bio processing with all available justices."""
    print("\n" + "=" * 60)
    print("TESTING WITH ALL JUSTICES")
    print("=" * 60)
    
    from justice_bios_augmentation import create_augmented_bios
    
    # Use conservative augmentation settings for full dataset
    augmentation_config = {
        'augmentations': ['word_embedding_augmentation'],
        'iterations': 3,
        'seed': 42,
        'verbose': True
    }
    
    print("🚀 Processing all available justice biographies...")
    
    success_count = create_augmented_bios(
        bios_dir=bios_dir,
        output_dir=output_dir,
        verbose=True,
        quiet=False,
        use_augmentation=True,
        augmentation_config=augmentation_config
    )
    
    print(f"\nCreated {success_count} justice bios with augmentation")
    
    # List output files
    output_files = sorted(os.listdir(output_dir))
    print(f"Total output files: {len(output_files)}")
    
    return output_files

def analyze_output_files(output_dir, files):
    """Analyze the structure of output files."""
    print(f"\n" + "=" * 60)
    print("OUTPUT FILE ANALYSIS")
    print("=" * 60)
    
    for filename in files:
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'r') as f:
            content = f.read()
            word_count = len(content.split())
            
        print(f"\n📄 {filename}")
        print(f"   Words: {word_count}")
        print(f"   Size: {len(content)} characters")
        
        # Show first 150 characters
        preview = content[:150].replace('\n', ' ')
        print(f"   Preview: {preview}...")

def compare_original_vs_augmented(output_dir):
    """Compare original and augmented versions."""
    print(f"\n" + "=" * 60)
    print("ORIGINAL VS AUGMENTED COMPARISON")
    print("=" * 60)
    
    # Find original and augmented versions
    files = os.listdir(output_dir)
    original_files = [f for f in files if f.endswith('_v0.txt')]
    
    for original_file in original_files:
        base_name = original_file.replace('_v0.txt', '')
        augmented_files = [f for f in files if f.startswith(base_name) and f != original_file]
        
        if augmented_files:
            print(f"\n🔍 {base_name}:")
            
            # Read original
            original_path = os.path.join(output_dir, original_file)
            with open(original_path, 'r') as f:
                original_content = f.read()
            
            # Read first augmented version
            aug_path = os.path.join(output_dir, augmented_files[0])
            with open(aug_path, 'r') as f:
                aug_content = f.read()
            
            # Compare word counts
            original_words = len(original_content.split())
            aug_words = len(aug_content.split())
            
            print(f"   Original ({original_file}): {original_words} words")
            print(f"   Augmented ({augmented_files[0]}): {aug_words} words")
            
            # Show a sample difference
            original_sample = original_content[:100].replace('\n', ' ')
            aug_sample = aug_content[:100].replace('\n', ' ')
            
            if original_sample != aug_sample:
                print(f"   Sample difference:")
                print(f"     Original: {original_sample}...")
                print(f"     Augmented: {aug_sample}...")
            else:
                print(f"   No visible differences in first 100 characters")

def main():
    """Run the test demonstration."""
    print("🧪 JUSTICE BIOS AUGMENTATION TEST")
    print("=" * 60)
    
    try:
        # Set up test environment
        test_dir, bios_dir, output_dir = setup_test_environment()
        
        if test_dir is None:
            return
        
        print(f"📁 Test directory: {test_dir}")
        print(f"📖 Bios directory: {bios_dir}")
        print(f"💾 Output directory: {output_dir}")
        
        # Check how many bios are available
        bio_files = [f for f in os.listdir(bios_dir) if f.endswith('.txt')]
        print(f"📋 Found {len(bio_files)} justice biographies")
        
        # Clear output directory
        if os.path.exists(output_dir):
            for file in os.listdir(output_dir):
                os.remove(os.path.join(output_dir, file))
        
        # Ask user for test mode
        print(f"\n🔧 Choose test mode:")
        print(f"1. Test subset (4 justices) - Fast")
        print(f"2. Test all justices - Slow but comprehensive")
        print(f"3. Test without augmentation - Just copy files")
        
        try:
            choice = input("Enter choice (1-3, default=1): ").strip()
            if not choice:
                choice = "1"
        except KeyboardInterrupt:
            print("\n👋 Test cancelled by user")
            return
        
        if choice == "1":
            # Test with subset
            files_with_aug = test_with_augmentation(bios_dir, output_dir, test_subset=True)
        elif choice == "2":
            # Test with all justices
            files_with_aug = test_with_all_justices(bios_dir, output_dir)
        elif choice == "3":
            # Test without augmentation
            files_with_aug = test_without_augmentation(bios_dir, output_dir)
        else:
            print("❌ Invalid choice, using subset test")
            files_with_aug = test_with_augmentation(bios_dir, output_dir, test_subset=True)
        
        # Analyze results
        if files_with_aug:
            analyze_output_files(output_dir, files_with_aug)
            
            # Compare original vs augmented (only if augmentation was used)
            if choice in ["1", "2"]:
                compare_original_vs_augmented(output_dir)
        
        print(f"\n✅ Test completed successfully!")
        print(f"📁 Check the output directory: {output_dir}")
        
        # Show summary
        print(f"\n📊 SUMMARY:")
        print(f"   Input bios: {len(bio_files)}")
        print(f"   Output files: {len(files_with_aug) if files_with_aug else 0}")
        print(f"   Test directory: {test_dir}")
        
        # Keep the test directory for inspection
        print(f"\n💡 To inspect the results, check: {test_dir}")
        print(f"   Test files are saved permanently in the test directory")
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # No cleanup needed - files are saved permanently
        pass

if __name__ == "__main__":
    main() 
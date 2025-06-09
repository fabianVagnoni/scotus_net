#!/usr/bin/env python3
import os
import re
import argparse
from pathlib import Path

def find_scotus_section(text: str) -> int:
    """
    Find the position where the first SCOTUS-related section starts.
    Returns the position, or -1 if no SCOTUS section is found.
    """
    # Common patterns for SCOTUS-related sections
    scotus_patterns = [
        r'=== .*Supreme Court.*===',
        r'=== .*SCOTUS.*===',
        r'=== .*U\.S\. Supreme Court.*===',
        r'=== .*Supreme Court justice.*===',
        r'=== .*Supreme Court nomination.*===',
        r'=== .*Supreme Court tenure.*===',
        r'=== .*Supreme Court service.*===',
        r'=== .*Supreme Court and.*===',
        r'=== .*Associate Justice.*===',
        r'=== .*Chief Justice.*===',
        r'=== .*Justice of the Supreme Court.*===',
    ]
    
    earliest_match = -1
    
    for pattern in scotus_patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE))
        if matches:
            match_pos = matches[0].start()
            if earliest_match == -1 or match_pos < earliest_match:
                earliest_match = match_pos
    
    return earliest_match

def process_biography(input_path: str, output_path: str) -> dict:
    """
    Process a single biography file by removing SCOTUS-related content.
    Returns stats about the processing.
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        original_text = f.read()
    
    original_words = len(original_text.split())
    original_chars = len(original_text)
    
    # Find where SCOTUS content starts
    scotus_start = find_scotus_section(original_text)
    
    if scotus_start == -1:
        # No SCOTUS section found, keep the entire text
        processed_text = original_text
        truncated = False
    else:
        # Truncate at the SCOTUS section
        processed_text = original_text[:scotus_start].strip()
        truncated = True
    
    # Clean up any trailing whitespace and normalize line endings
    processed_text = re.sub(r'\s+$', '', processed_text, flags=re.MULTILINE)
    processed_text = processed_text.strip()
    
    # Save the processed text
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(processed_text)
    
    processed_words = len(processed_text.split())
    processed_chars = len(processed_text)
    
    return {
        'original_words': original_words,
        'original_chars': original_chars,
        'processed_words': processed_words,
        'processed_chars': processed_chars,
        'truncated': truncated,
        'reduction_words': original_words - processed_words,
        'reduction_chars': original_chars - processed_chars,
    }

def main(input_dir: str, output_dir: str):
    """
    Process all biography files in the input directory.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        return
    
    # Get all .txt files in the input directory
    bio_files = list(input_path.glob("*.txt"))
    
    if not bio_files:
        print(f"No .txt files found in '{input_dir}'")
        return
    
    print(f"Processing {len(bio_files)} biography files...")
    print("=" * 80)
    
    total_stats = {
        'processed': 0,
        'truncated': 0,
        'total_original_words': 0,
        'total_processed_words': 0,
        'total_reduction_words': 0,
    }
    
    for bio_file in sorted(bio_files):
        justice_name = bio_file.stem  # filename without extension
        output_file = output_path / bio_file.name
        
        try:
            stats = process_biography(str(bio_file), str(output_file))
            
            status = "TRUNCATED" if stats['truncated'] else "KEPT FULL"
            reduction_pct = (stats['reduction_words'] / stats['original_words'] * 100) if stats['original_words'] > 0 else 0
            
            print(f"[{status:10}] {justice_name}")
            print(f"              {stats['original_words']} → {stats['processed_words']} words "
                  f"(-{stats['reduction_words']}, -{reduction_pct:.1f}%)")
            
            # Update totals
            total_stats['processed'] += 1
            if stats['truncated']:
                total_stats['truncated'] += 1
            total_stats['total_original_words'] += stats['original_words']
            total_stats['total_processed_words'] += stats['processed_words']
            total_stats['total_reduction_words'] += stats['reduction_words']
            
        except Exception as e:
            print(f"[ERROR     ] {justice_name}: {e}")
    
    print("=" * 80)
    print(f"SUMMARY:")
    print(f"  Files processed: {total_stats['processed']}")
    print(f"  Files truncated: {total_stats['truncated']}")
    print(f"  Files kept full: {total_stats['processed'] - total_stats['truncated']}")
    
    if total_stats['total_original_words'] > 0:
        overall_reduction = (total_stats['total_reduction_words'] / total_stats['total_original_words'] * 100)
        print(f"  Total words: {total_stats['total_original_words']:,} → {total_stats['total_processed_words']:,}")
        print(f"  Overall reduction: {total_stats['total_reduction_words']:,} words ({overall_reduction:.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process justice biographies by removing SCOTUS-related content"
    )
    parser.add_argument(
        "--input",
        "-i",
        default="data/raw/bios",
        help="Directory containing biography .txt files"
    )
    parser.add_argument(
        "--output",
        "-o", 
        default="data/processed",
        help="Directory to save processed biographies"
    )
    
    args = parser.parse_args()
    main(args.input, args.output) 
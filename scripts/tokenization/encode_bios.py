#!/usr/bin/env python3
"""
Pre-encode all justice biographies using the sentence transformer.

This script reads all biography files, encodes them using the bio sentence transformer,
and saves the embeddings for fast training without repeated encoding.
"""

import os
import sys
import json
import torch
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
from transformers import AutoTokenizer
import argparse

# Import configuration
from config import get_config, get_bio_config

def encode_biography_files(
    bios_dir: str = None,
    output_file: str = None,
    model_name: str = None,
    embedding_dim: int = None,
    max_sequence_length: int = None,
    batch_size: int = None,
    device: str = None,
    file_list: str = None
):
    """
    Encode all biography files and save embeddings.
    
    Args:
        bios_dir: Directory containing biography .txt files
        output_file: Path to save encoded embeddings
        model_name: HuggingFace model name for encoding
        embedding_dim: Target embedding dimension (for projection)
        max_sequence_length: Maximum sequence length for tokenization
        batch_size: Batch size for encoding
        device: Device to use ('cuda', 'cpu', or None for auto)
        file_list: Optional text file containing specific files to encode
    """
    # Load configuration and set defaults
    config = get_config()
    bio_config = get_bio_config()
    
    # Use config values as defaults if not provided
    if bios_dir is None:
        bios_dir = bio_config['input_dir']
    if output_file is None:
        output_file = bio_config['output_file']
    if model_name is None:
        model_name = bio_config['model_name']
    if embedding_dim is None:
        embedding_dim = bio_config['embedding_dim']
    if max_sequence_length is None:
        max_sequence_length = bio_config['max_sequence_length']
    if batch_size is None:
        batch_size = bio_config['batch_size']
    
    # Set device
    if device is None:
        device_config = bio_config['device']
        if device_config == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = device_config
    
    print(f"🚀 Pre-tokenizing Justice Biographies")
    print(f"📁 Bios directory: {bios_dir}")
    print(f"🤖 Model: {model_name}")
    print(f"💾 Device: {device}")
    print(f"📊 Max sequence length: {max_sequence_length}")
    
    # Initialize tokenizer only (model will be used during training)
    print("\n📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"✅ Tokenizer loaded successfully")
    
    # Find all biography files
    bio_files = []
    
    if file_list:
        # Load files from list
        print(f"\n📋 Loading file list from: {file_list}")
        with open(file_list, 'r', encoding='utf-8') as f:
            file_paths = [line.strip() for line in f if line.strip()]
        
        bio_files = [Path(path) for path in file_paths if os.path.exists(path)]
        missing_files = [path for path in file_paths if not os.path.exists(path)]
        
        print(f"📚 Found {len(bio_files)} biography files from list")
        if missing_files:
            print(f"⚠️  {len(missing_files)} files from list are missing")
            
    elif bios_dir:
        # Load files from directory
        bios_path = Path(bios_dir)
        if not bios_path.exists():
            raise FileNotFoundError(f"Bios directory not found: {bios_dir}")
        
        bio_files = list(bios_path.glob("*.txt"))
        print(f"\n📚 Found {len(bio_files)} biography files in directory")
    else:
        raise ValueError("Either --bios-dir or --file-list must be provided")
    
    if len(bio_files) == 0:
        raise ValueError("No .txt files found to encode")
    
    # Read all biography texts
    print("📖 Reading biography files...")
    bio_data = {}
    failed_files = []
    
    for bio_file in tqdm(bio_files, desc="Reading files"):
        try:
            with open(bio_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    bio_data[str(bio_file)] = content
                else:
                    print(f"⚠️  Empty file: {bio_file}")
        except Exception as e:
            print(f"❌ Error reading {bio_file}: {e}")
            failed_files.append(str(bio_file))
    
    print(f"✅ Successfully read {len(bio_data)} files")
    if failed_files:
        print(f"❌ Failed to read {len(failed_files)} files")
    
    # Tokenize biographies
    print(f"\n🧠 Tokenizing biographies (batch size: {batch_size})...")
    
    bio_paths = list(bio_data.keys())
    bio_texts = list(bio_data.values())
    tokenized_data = {}
    
    # Process in batches for memory efficiency
    for i in tqdm(range(0, len(bio_texts), batch_size), desc="Tokenizing batches"):
        batch_texts = bio_texts[i:i + batch_size]
        batch_paths = bio_paths[i:i + batch_size]
        
        # Tokenize batch
        tokenized = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_sequence_length,
            return_tensors="pt"
        )
        
        # Store tokenized data for each file
        for j, path in enumerate(batch_paths):
            tokenized_data[path] = {
                'input_ids': tokenized['input_ids'][j],
                'attention_mask': tokenized['attention_mask'][j]
            }
    
    print(f"✅ Tokenized {len(tokenized_data)} biographies")
    
    # Create metadata
    metadata = {
        'model_name': model_name,
        'max_sequence_length': max_sequence_length,
        'num_tokenized': len(tokenized_data),
        'failed_files': failed_files,
        'device_used': device,
        'tokenization_method': 'transformers_autotokenizer'
    }
    
    # Save tokenized data and metadata
    print(f"\n💾 Saving tokenized data to: {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    save_data = {
        'tokenized_data': tokenized_data,
        'metadata': metadata
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(save_data, f)
    
    # Calculate file size
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"✅ Saved {len(tokenized_data)} tokenized biographies ({file_size_mb:.1f} MB)")
    
    # Print summary
    print(f"\n📊 TOKENIZATION SUMMARY")
    print(f"   Total files found: {len(bio_files)}")
    print(f"   Successfully tokenized: {len(tokenized_data)}")
    print(f"   Failed files: {len(failed_files)}")
    print(f"   Output file: {output_file}")
    print(f"   File size: {file_size_mb:.1f} MB")
    
    return output_file, metadata

def load_tokenized_bios(tokenized_file: str) -> tuple:
    """
    Load pre-tokenized biography data.
    
    Returns:
        (tokenized_dict, metadata)
    """
    print(f"📥 Loading pre-tokenized biographies from: {tokenized_file}")
    
    with open(tokenized_file, 'rb') as f:
        data = pickle.load(f)
    
    tokenized_data = data['tokenized_data']
    metadata = data['metadata']
    
    print(f"✅ Loaded {metadata['num_tokenized']} pre-tokenized biographies")
    print(f"🤖 Original model: {metadata['model_name']}")
    print(f"📏 Max sequence length: {metadata['max_sequence_length']}")
    
    return tokenized_data, metadata

def main():
    # Load configuration for defaults
    config = get_config()
    bio_config = get_bio_config()
    
    parser = argparse.ArgumentParser(
        description="Pre-tokenize justice biographies for efficient training"
    )
    parser.add_argument(
        "--bios-dir",
        default=bio_config['input_dir'],
        help="Directory containing biography .txt files"
    )
    parser.add_argument(
        "--file-list",
        help="Text file containing list of specific biography files to encode (one per line)"
    )
    parser.add_argument(
        "--output",
        "-o",
        default=bio_config['output_file'],
        help="Output file for tokenized data"
    )
    parser.add_argument(
        "--model-name",
        default=bio_config['model_name'],
        help="HuggingFace model name for tokenization"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=bio_config['batch_size'],
        help="Batch size for tokenization"
    )
    parser.add_argument(
        "--device",
        choices=['cuda', 'cpu', 'auto'],
        default=bio_config['device'],
        help="Device preference (not used for tokenization, kept for compatibility)"
    )
    parser.add_argument(
        "--config",
        help="Path to configuration file (default: config.env in same directory)"
    )
    
    args = parser.parse_args()
    
    # Load custom config if provided
    if args.config:
        config = get_config(args.config)
        print(f"📋 Using custom config: {args.config}")
    
    device = args.device if args.device != 'auto' else None
    
    try:
        encode_biography_files(
            bios_dir=args.bios_dir,
            output_file=args.output,
            model_name=args.model_name,
            batch_size=args.batch_size,
            device=device,
            file_list=args.file_list
        )
        
        print("\n🎉 Biography tokenization completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during tokenization: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Pre-encode all case descriptions using the legal sentence transformer.

This script reads all case description files, encodes them using the legal sentence transformer,
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
from sentence_transformers import SentenceTransformer
import argparse

# Import configuration
from config import get_config, get_description_config

def encode_description_files(
    descriptions_dir: str = None,
    output_file: str = None,
    model_name: str = None,
    embedding_dim: int = None,
    max_sequence_length: int = None,
    batch_size: int = None,
    device: str = None,
    file_list: str = None
):
    """
    Encode all case description files and save embeddings.
    
    Args:
        descriptions_dir: Directory containing case description .txt files
        output_file: Path to save encoded embeddings
        model_name: HuggingFace model name for encoding (legal model)
        embedding_dim: Target embedding dimension (for projection)
        max_sequence_length: Maximum sequence length for tokenization
        batch_size: Batch size for encoding
        device: Device to use ('cuda', 'cpu', or None for auto)
        file_list: Optional text file containing specific files to encode
    """
    # Load configuration and set defaults
    config = get_config()
    desc_config = get_description_config()
    
    # Use config values as defaults if not provided
    if descriptions_dir is None:
        descriptions_dir = desc_config['input_dir']
    if output_file is None:
        output_file = desc_config['output_file']
    if model_name is None:
        model_name = desc_config['model_name']
    if embedding_dim is None:
        embedding_dim = desc_config['embedding_dim']
    if max_sequence_length is None:
        max_sequence_length = desc_config['max_sequence_length']
    if batch_size is None:
        batch_size = desc_config['batch_size']
    
    # Set device
    if device is None:
        device_config = desc_config['device']
        if device_config == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = device_config
    
    print(f"🚀 Pre-encoding Case Descriptions")
    print(f"📁 Descriptions directory: {descriptions_dir}")
    print(f"🤖 Model: {model_name}")
    print(f"💾 Device: {device}")
    print(f"📊 Target embedding dim: {embedding_dim}")
    
    # Initialize sentence transformer model
    print("\n📥 Loading legal sentence transformer model...")
    model = SentenceTransformer(model_name, device=device)
    
    # Get actual embedding dimension from the model
    actual_embedding_dim = model.get_sentence_embedding_dimension()
    print(f"📏 Model embedding dim: {actual_embedding_dim}")
    
    # Check if embedding dimensions match
    if actual_embedding_dim != embedding_dim:
        print(f"⚠️  Warning: Model dimension ({actual_embedding_dim}) doesn't match target ({embedding_dim})")
        print(f"   Using model's native dimension: {actual_embedding_dim}")
        embedding_dim = actual_embedding_dim
    
    # Find all case description files
    description_files = []
    
    if file_list:
        # Load files from list
        print(f"\n📋 Loading file list from: {file_list}")
        with open(file_list, 'r', encoding='utf-8') as f:
            file_paths = [line.strip() for line in f if line.strip()]
        
        description_files = [Path(path) for path in file_paths if os.path.exists(path)]
        missing_files = [path for path in file_paths if not os.path.exists(path)]
        
        print(f"📚 Found {len(description_files)} case description files from list")
        if missing_files:
            print(f"⚠️  {len(missing_files)} files from list are missing")
            
    elif descriptions_dir:
        # Load files from directory
        descriptions_path = Path(descriptions_dir)
        if not descriptions_path.exists():
            raise FileNotFoundError(f"Descriptions directory not found: {descriptions_dir}")
        
        description_files = list(descriptions_path.glob("*.txt"))
        print(f"\n📚 Found {len(description_files)} case description files in directory")
    else:
        raise ValueError("Either --descriptions-dir or --file-list must be provided")
    
    if len(description_files) == 0:
        raise ValueError("No .txt files found to encode")
    
    # Read all case description texts
    print("📖 Reading case description files...")
    description_data = {}
    failed_files = []
    skipped_large_files = []
    
    for desc_file in tqdm(description_files, desc="Reading files"):
        try:
            with open(desc_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    # Check file size (skip extremely large files that might cause memory issues)
                    word_count = len(content.split())
                    max_words = desc_config.get('max_words', 10000)
                    if word_count > max_words:  # Skip files with more than max_words
                        skipped_large_files.append((str(desc_file), word_count))
                        continue
                    
                    description_data[str(desc_file)] = content
                else:
                    print(f"⚠️  Empty file: {desc_file}")
        except Exception as e:
            print(f"❌ Error reading {desc_file}: {e}")
            failed_files.append(str(desc_file))
    
    print(f"✅ Successfully read {len(description_data)} files")
    if failed_files:
        print(f"❌ Failed to read {len(failed_files)} files")
    if skipped_large_files:
        print(f"⚠️  Skipped {len(skipped_large_files)} very large files (>10k words)")
    
    # Encode case descriptions using sentence transformers
    print(f"\n🧠 Encoding case descriptions (batch size: {batch_size})...")
    
    desc_paths = list(description_data.keys())
    desc_texts = list(description_data.values())
    encoded_embeddings = {}
    
    # Process in batches for memory efficiency
    for i in tqdm(range(0, len(desc_texts), batch_size), desc="Encoding batches"):
        batch_texts = desc_texts[i:i + batch_size]
        batch_paths = desc_paths[i:i + batch_size]
        
        try:
            # Encode batch using sentence transformer (handles tokenization, pooling internally)
            batch_embeddings = model.encode(
                batch_texts,
                batch_size=len(batch_texts),
                show_progress_bar=False,
                convert_to_tensor=True,
                device=device
            )
            
            # Store embeddings (move to CPU to save memory)
            for j, path in enumerate(batch_paths):
                encoded_embeddings[path] = batch_embeddings[j].cpu()
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"⚠️  GPU memory issue at batch {i//batch_size + 1}. Clearing cache...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Try processing this batch one by one
                for single_text, single_path in zip(batch_texts, batch_paths):
                    try:
                        single_embedding = model.encode(
                            [single_text],
                            convert_to_tensor=True,
                            device=device
                        )
                        encoded_embeddings[single_path] = single_embedding[0].cpu()
                    except Exception as single_e:
                        print(f"❌ Failed to encode {single_path}: {single_e}")
                        failed_files.append(single_path)
            else:
                print(f"❌ Error encoding batch {i//batch_size + 1}: {e}")
                failed_files.extend(batch_paths)
    
    print(f"✅ Encoded {len(encoded_embeddings)} case descriptions")
    
    # Create metadata
    metadata = {
        'model_name': model_name,
        'embedding_dim': embedding_dim,
        'max_sequence_length': max_sequence_length,
        'num_embeddings': len(encoded_embeddings),
        'failed_files': failed_files,
        'skipped_large_files': skipped_large_files,
        'device_used': device,
        'encoding_method': 'sentence_transformers'
    }
    
    # Save embeddings and metadata
    print(f"\n💾 Saving embeddings to: {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    save_data = {
        'embeddings': encoded_embeddings,
        'metadata': metadata
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(save_data, f)
    
    # Calculate file size
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"✅ Saved {len(encoded_embeddings)} embeddings ({file_size_mb:.1f} MB)")
    
    # Print summary
    print(f"\n📊 ENCODING SUMMARY")
    print(f"   Total files found: {len(description_files)}")
    print(f"   Successfully encoded: {len(encoded_embeddings)}")
    print(f"   Failed files: {len(failed_files)}")
    print(f"   Skipped large files: {len(skipped_large_files)}")
    print(f"   Output file: {output_file}")
    print(f"   File size: {file_size_mb:.1f} MB")
    
    return output_file, metadata

def load_encoded_descriptions(embeddings_file: str) -> tuple:
    """
    Load pre-encoded case description embeddings.
    
    Returns:
        (embeddings_dict, metadata)
    """
    print(f"📥 Loading pre-encoded case descriptions from: {embeddings_file}")
    
    with open(embeddings_file, 'rb') as f:
        data = pickle.load(f)
    
    embeddings = data['embeddings']
    metadata = data['metadata']
    
    print(f"✅ Loaded {metadata['num_embeddings']} pre-encoded case descriptions")
    print(f"🤖 Original model: {metadata['model_name']}")
    print(f"📏 Embedding dimension: {metadata['embedding_dim']}")
    
    return embeddings, metadata

def encode_from_dataset(dataset_file: str, descriptions_dir: str, output_file: str, **kwargs):
    """
    Encode only the case descriptions that are referenced in the dataset.
    This is more efficient than encoding all files.
    """
    print(f"🎯 Encoding only case descriptions referenced in dataset: {dataset_file}")
    
    # Load dataset to find referenced descriptions
    with open(dataset_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # Extract unique case description paths
    referenced_paths = set()
    for case_id, case_data in dataset.items():
        # Handle both old format (3 elements) and new format (4 elements with encoded_locations)
        if len(case_data) == 3:
            justice_bio_paths, case_description_path, voting_percentages = case_data
        elif len(case_data) == 4:
            justice_bio_paths, case_description_path, voting_percentages, encoded_locations = case_data
        else:
            print(f"⚠️  Unexpected case data format for case {case_id}: {len(case_data)} elements")
            continue
            
        if case_description_path and case_description_path.strip():
            # Normalize path separators for cross-platform compatibility
            normalized_path = case_description_path.strip().replace('\\', '/')
            referenced_paths.add(normalized_path)
    
    print(f"📚 Found {len(referenced_paths)} unique case descriptions referenced in dataset")
    
    # Filter to only existing files
    existing_paths = []
    missing_paths = []
    
    for path in referenced_paths:
        if os.path.exists(path):
            existing_paths.append(path)
        else:
            missing_paths.append(path)
    
    print(f"✅ {len(existing_paths)} files exist")
    if missing_paths:
        print(f"⚠️  {len(missing_paths)} referenced files are missing")
    
    # Create a temporary file list for the existing paths
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for path in existing_paths:
            f.write(f"{path}\n")
        temp_file_list = f.name
    
    try:
        # Call the main encoding function with the file list
        return encode_description_files(
            descriptions_dir=descriptions_dir,
            output_file=output_file,
            file_list=temp_file_list,
            **kwargs
        )
    finally:
        # Clean up the temporary file
        os.unlink(temp_file_list)

def main():
    # Load configuration for defaults
    config = get_config()
    desc_config = get_description_config()
    
    parser = argparse.ArgumentParser(
        description="Pre-encode case descriptions for fast training"
    )
    parser.add_argument(
        "--descriptions-dir",
        default=desc_config['input_dir'],
        help="Directory containing case description .txt files"
    )
    parser.add_argument(
        "--file-list",
        help="Text file containing list of specific description files to encode (one per line)"
    )
    parser.add_argument(
        "--output",
        "-o",
        default=desc_config['output_file'],
        help="Output file for encoded embeddings"
    )
    parser.add_argument(
        "--model-name",
        default=desc_config['model_name'],
        help="HuggingFace model name for encoding (legal model)"
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=desc_config['embedding_dim'],
        help="Target embedding dimension"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=desc_config['batch_size'],
        help="Batch size for encoding (smaller for legal models)"
    )
    parser.add_argument(
        "--device",
        choices=['cuda', 'cpu', 'auto'],
        default=desc_config['device'],
        help="Device to use for encoding"
    )
    parser.add_argument(
        "--dataset-file",
        default=config.dataset_file,
        help="Optional: Only encode descriptions referenced in this dataset file"
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
        if args.dataset_file:
            # Encode only descriptions referenced in dataset
            encode_from_dataset(
                dataset_file=args.dataset_file,
                descriptions_dir=args.descriptions_dir,
                output_file=args.output,
                model_name=args.model_name,
                embedding_dim=args.embedding_dim,
                batch_size=args.batch_size,
                device=device
            )
        else:
            # Encode all descriptions in directory
            encode_description_files(
                descriptions_dir=args.descriptions_dir,
                output_file=args.output,
                model_name=args.model_name,
                embedding_dim=args.embedding_dim,
                batch_size=args.batch_size,
                device=device,
                file_list=args.file_list
            )
        
        print("\n🎉 Case description encoding completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during encoding: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
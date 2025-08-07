#!/usr/bin/env python3
"""
Holdout Test Set Creation for SCOTUS AI

This module creates a holdout test set by selecting the most recent 15% of cases
that have descriptions. These cases will be excluded from training and validation
to ensure unbiased evaluation on truly unseen data.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
import argparse

# Add scripts to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import get_logger


class HoldoutTestSetManager:
    """Manages the creation and loading of holdout test sets."""
    
    def __init__(self, dataset_file: str = None, holdout_file: str = None):
        """
        Initialize the holdout test set manager.
        
        Args:
            dataset_file: Path to the case dataset JSON file
            holdout_file: Path to save/load the holdout test set case IDs
        """
        self.logger = get_logger(__name__)
        
        # Default paths
        if dataset_file is None:
            dataset_file = "data/processed/case_dataset.json"
        if holdout_file is None:
            holdout_file = "data/processed/holdout_test_cases.json"
            
        self.dataset_file = dataset_file
        self.holdout_file = holdout_file
        
    def load_dataset(self) -> Dict:
        """Load the case dataset."""
        if not os.path.exists(self.dataset_file):
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_file}")
        
        self.logger.info(f"Loading dataset from: {self.dataset_file}")
        
        with open(self.dataset_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        self.logger.info(f"Loaded {len(dataset)} cases from dataset")
        return dataset
    
    def filter_cases_with_descriptions(self, dataset: Dict) -> Dict:
        """
        Filter cases to only include those with valid descriptions.
        
        Args:
            dataset: Full case dataset
            
        Returns:
            Dictionary of cases that have valid descriptions
        """
        cases_with_descriptions = {}
        
        for case_id, case_data in dataset.items():
            # Handle both old format (3 elements) and new format (4 elements)
            if len(case_data) >= 2:
                case_description_path = case_data[1]  # Second element is description path
                
                # Check if description path exists and is not empty
                if case_description_path and case_description_path.strip():
                    # Normalize path separators for cross-platform compatibility
                    normalized_path = case_description_path.replace('\\', '/')
                    
                    # Verify the file exists
                    if os.path.exists(normalized_path):
                        cases_with_descriptions[case_id] = case_data
                    else:
                        self.logger.debug(f"Description file missing for case {case_id}: {normalized_path}")
                else:
                    self.logger.debug(f"No description path for case {case_id}")
        
        self.logger.info(f"Found {len(cases_with_descriptions)} cases with valid descriptions "
                        f"out of {len(dataset)} total cases")
        
        return cases_with_descriptions
    
    def extract_year_from_case_id(self, case_id: str) -> int:
        """
        Extract year from case ID.
        
        Args:
            case_id: Case ID in format YYYY-XXX-XX-XX
            
        Returns:
            Year as integer
        """
        try:
            return int(case_id[:4])
        except (ValueError, IndexError):
            self.logger.warning(f"Could not extract year from case ID: {case_id}")
            return 0
    
    def select_most_recent_cases(self, cases_with_descriptions: Dict, percentage: float = 0.15) -> List[str]:
        """
        Select the most recent percentage of cases for holdout test set.
        
        Args:
            cases_with_descriptions: Dictionary of cases with descriptions
            percentage: Percentage of cases to select (default 0.15 for 15%)
            
        Returns:
            List of case IDs for holdout test set
        """
        # Sort cases by year (most recent first)
        case_ids_by_year = sorted(
            cases_with_descriptions.keys(),
            key=self.extract_year_from_case_id,
            reverse=True
        )
        
        # Calculate number of cases to select
        total_cases = len(case_ids_by_year)
        num_holdout_cases = int(total_cases * percentage)
        
        # Select the most recent cases
        holdout_case_ids = case_ids_by_year[:num_holdout_cases]
        
        # Log statistics
        if holdout_case_ids:
            holdout_years = [self.extract_year_from_case_id(case_id) for case_id in holdout_case_ids]
            min_year = min(holdout_years)
            max_year = max(holdout_years)
            
            self.logger.info(f"Selected {len(holdout_case_ids)} cases ({percentage*100:.1f}%) for holdout test set")
            self.logger.info(f"Holdout cases span from {min_year} to {max_year} - these years will be LEFT OUT for testing")
            
            # Count cases by year
            year_counts = {}
            for year in holdout_years:
                year_counts[year] = year_counts.get(year, 0) + 1
            
            self.logger.info("Cases by year in holdout set:")
            for year in sorted(year_counts.keys(), reverse=True):
                self.logger.info(f"  {year}: {year_counts[year]} cases")
        
        return holdout_case_ids
    
    def create_holdout_test_set(self, percentage: float = 0.15, force_recreate: bool = False) -> List[str]:
        """
        Create the holdout test set by selecting the most recent cases with descriptions.
        
        Args:
            percentage: Percentage of cases to select for holdout (default for 15%)
            force_recreate: Whether to recreate even if holdout file already exists
            
        Returns:
            List of case IDs in the holdout test set
        """
        # Check if holdout file already exists
        if os.path.exists(self.holdout_file) and not force_recreate:
            self.logger.info(f"Holdout test set already exists at: {self.holdout_file}")
            self.logger.info("Use force_recreate=True to recreate it")
            return self.load_holdout_test_set()
        
        self.logger.info("Creating holdout test set...")
        
        # Load dataset
        dataset = self.load_dataset()
        
        # Filter to cases with descriptions
        cases_with_descriptions = self.filter_cases_with_descriptions(dataset)
        
        if not cases_with_descriptions:
            raise ValueError("No cases with descriptions found in dataset")
        
        # Select most recent cases
        holdout_case_ids = self.select_most_recent_cases(cases_with_descriptions, percentage)
        
        # Save holdout test set
        self.save_holdout_test_set(holdout_case_ids)
        
        return holdout_case_ids
    
    def save_holdout_test_set(self, holdout_case_ids: List[str]):
        """
        Save the holdout test set case IDs to file.
        
        Args:
            holdout_case_ids: List of case IDs for holdout test set
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.holdout_file), exist_ok=True)
        
        # Create metadata
        holdout_data = {
            "description": "Holdout test set case IDs - DO NOT USE FOR TRAINING OR VALIDATION",
            "creation_date": str(Path(__file__).stat().st_mtime),
            "total_cases": len(holdout_case_ids),
            "source_dataset": self.dataset_file,
            "case_ids": holdout_case_ids
        }
        
        # Save to file
        with open(self.holdout_file, 'w', encoding='utf-8') as f:
            json.dump(holdout_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved holdout test set with {len(holdout_case_ids)} cases to: {self.holdout_file}")
    
    def load_holdout_test_set(self) -> List[str]:
        """
        Load the holdout test set case IDs from file.
        
        Returns:
            List of case IDs in the holdout test set
        """
        if not os.path.exists(self.holdout_file):
            raise FileNotFoundError(f"Holdout test set file not found: {self.holdout_file}")
        
        with open(self.holdout_file, 'r', encoding='utf-8') as f:
            holdout_data = json.load(f)
        
        case_ids = holdout_data.get("case_ids", [])
        
        self.logger.info(f"Loaded holdout test set with {len(case_ids)} cases from: {self.holdout_file}")
        
        return case_ids
    
    def get_holdout_case_ids(self) -> Set[str]:
        """
        Get the holdout test set case IDs as a set for fast lookup.
        
        Returns:
            Set of case IDs in the holdout test set
        """
        try:
            case_ids = self.load_holdout_test_set()
            return set(case_ids)
        except FileNotFoundError:
            self.logger.warning("Holdout test set not found. Creating it now...")
            case_ids = self.create_holdout_test_set()
            return set(case_ids)
    
    def filter_dataset_exclude_holdout(self, dataset: Dict) -> Dict:
        """
        Filter dataset to exclude holdout test cases.
        
        Args:
            dataset: Full dataset dictionary
            
        Returns:
            Dataset with holdout cases removed
        """
        holdout_case_ids = self.get_holdout_case_ids()
        
        filtered_dataset = {
            case_id: case_data 
            for case_id, case_data in dataset.items()
            if case_id not in holdout_case_ids
        }
        
        removed_count = len(dataset) - len(filtered_dataset)
        self.logger.info(f"Filtered dataset: removed {removed_count} holdout cases, "
                        f"keeping {len(filtered_dataset)} cases for training/validation")
        
        return filtered_dataset
    
    def get_holdout_dataset(self, full_dataset: Dict) -> Dict:
        """
        Get only the holdout test cases from the full dataset.
        
        Args:
            full_dataset: Full dataset dictionary
            
        Returns:
            Dataset containing only holdout test cases
        """
        holdout_case_ids = self.get_holdout_case_ids()
        
        holdout_dataset = {
            case_id: case_data 
            for case_id, case_data in full_dataset.items()
            if case_id in holdout_case_ids
        }
        
        self.logger.info(f"Created holdout dataset with {len(holdout_dataset)} cases")
        
        return holdout_dataset


class TimeBasedCrossValidator:
    """Manages time-based cross-validation splits for temporal data."""
    
    def __init__(self, n_folds: int = 3, train_size: int = 1000, val_size: int = 100):
        """
        Initialize the time-based cross-validator.
        
        Args:
            n_folds: Number of CV folds to create
            train_size: Number of cases in each training set
            val_size: Number of cases in each validation set
        """
        self.logger = get_logger(__name__)
        self.n_folds = n_folds
        self.train_size = train_size
        self.val_size = val_size
    
    def create_time_based_cv_splits(self, dataset: Dict, holdout_manager: HoldoutTestSetManager) -> List[Tuple[Dict, Dict]]:
        """
        Create time-based cross-validation splits.
        
        Args:
            dataset: Full case dataset
            holdout_manager: Holdout test set manager to exclude holdout cases
            
        Returns:
            List of (train_dataset, val_dataset) tuples for each fold
        """
        # Exclude holdout cases
        available_dataset = holdout_manager.filter_dataset_exclude_holdout(dataset)
        
        # Filter to cases with descriptions
        cases_with_descriptions = holdout_manager.filter_cases_with_descriptions(available_dataset)
        
        if not cases_with_descriptions:
            raise ValueError("No cases with descriptions found for CV splits")
        
        # Sort cases by year (oldest first for temporal CV)
        case_ids_by_year = sorted(
            cases_with_descriptions.keys(),
            key=holdout_manager.extract_year_from_case_id
        )
        
        total_cases = len(case_ids_by_year)
        fold_size = self.train_size + self.val_size
        
        self.logger.info(f"Creating {self.n_folds} time-based CV folds...")
        self.logger.info(f"   Total available cases: {total_cases}")
        self.logger.info(f"   Cases per fold: {fold_size} (train: {self.train_size}, val: {self.val_size})")
        
        cv_splits = []
        
        for fold_idx in range(self.n_folds):
            # Calculate start position for this fold
            start_idx = fold_idx * fold_size
            
            if start_idx + fold_size > total_cases:
                self.logger.warning(f"Not enough cases for fold {fold_idx + 1}. Skipping.")
                break
            
            # Get case IDs for this fold
            train_end_idx = start_idx + self.train_size
            val_end_idx = train_end_idx + self.val_size
            
            train_case_ids = case_ids_by_year[start_idx:train_end_idx]
            val_case_ids = case_ids_by_year[train_end_idx:val_end_idx]
            
            # Create datasets for this fold
            train_dataset = {case_id: cases_with_descriptions[case_id] for case_id in train_case_ids}
            val_dataset = {case_id: cases_with_descriptions[case_id] for case_id in val_case_ids}
            
            # Log fold statistics
            train_years = [holdout_manager.extract_year_from_case_id(case_id) for case_id in train_case_ids]
            val_years = [holdout_manager.extract_year_from_case_id(case_id) for case_id in val_case_ids]
            
            self.logger.info(f"   Fold {fold_idx + 1}:")
            self.logger.info(f"     Train: {len(train_case_ids)} cases ({min(train_years)}-{max(train_years)})")
            self.logger.info(f"     Val:   {len(val_case_ids)} cases ({min(val_years)}-{max(val_years)})")
            
            cv_splits.append((train_dataset, val_dataset))
        
        self.logger.info(f"Created {len(cv_splits)} time-based CV folds")
        return cv_splits


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Create holdout test set for SCOTUS AI")
    parser.add_argument("--dataset-file", type=str, default="data/processed/case_dataset.json",
                       help="Path to case dataset JSON file")
    parser.add_argument("--holdout-file", type=str, default="data/processed/holdout_test_cases.json",
                       help="Path to save holdout test set case IDs")
    parser.add_argument("--percentage", type=float, default=0.15,
                       help="Percentage of cases to select for holdout (default: 0.15)")
    parser.add_argument("--force-recreate", action="store_true",
                       help="Force recreate holdout set even if it exists")
    parser.add_argument("--show-stats", action="store_true",
                       help="Show detailed statistics about the holdout set")
    
    args = parser.parse_args()
    
    # Create manager
    manager = HoldoutTestSetManager(args.dataset_file, args.holdout_file)
    
    # Create or load holdout test set
    holdout_case_ids = manager.create_holdout_test_set(
        percentage=args.percentage,
        force_recreate=args.force_recreate
    )
    
    if args.show_stats:
        print(f"\n📊 Holdout Test Set Statistics:")
        print(f"   Total cases: {len(holdout_case_ids)}")
        print(f"   Percentage: {args.percentage*100:.1f}%")
        
        # Show year distribution
        years = [manager.extract_year_from_case_id(case_id) for case_id in holdout_case_ids]
        year_counts = {}
        for year in years:
            year_counts[year] = year_counts.get(year, 0) + 1
        
        print(f"   Year range: {min(years)} - {max(years)}")
        print(f"   Cases by year:")
        for year in sorted(year_counts.keys(), reverse=True):
            print(f"     {year}: {year_counts[year]} cases")


if __name__ == "__main__":
    main()
"""Data processing module for SCOTUS case data."""

import json
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import spacy

from ..utils.logger import get_logger
from ..utils.config import config


class SCOTUSDataProcessor:
    """SCOTUS case data processor."""
    
    def __init__(self):
        """Initialize the processor."""
        self.logger = get_logger(__name__)
        self.config = config
        
        # Download required NLTK data
        self._download_nltk_data()
        
        # Initialize processing tools
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def _download_nltk_data(self):
        """Download required NLTK data."""
        required_data = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
        
        for data in required_data:
            try:
                nltk.download(data, quiet=True)
            except Exception as e:
                self.logger.warning(f"Could not download NLTK data '{data}': {str(e)}")
    
    def load_raw_data(self, data_path: str = None) -> pd.DataFrame:
        """Load raw scraped data.
        
        Args:
            data_path: Path to raw data file. If None, uses config default.
            
        Returns:
            DataFrame with raw case data.
        """
        if data_path is None:
            data_path = Path(self.config.get('data.raw_data_path')) / 'all_cases.json'
        else:
            data_path = Path(data_path)
        
        self.logger.info(f"Loading raw data from {data_path}")
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            df = pd.DataFrame(data)
            self.logger.info(f"Loaded {len(df)} cases")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            return pd.DataFrame()
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text.
        
        Args:
            text: Raw text to clean.
            
        Returns:
            Cleaned text.
        """
        if not isinstance(text, str):
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        if self.config.get('pipeline.preprocessing.remove_punctuation', False):
            text = re.sub(r'[^\w\s]', '', text)
        
        # Convert to lowercase if configured
        if self.config.get('pipeline.preprocessing.lowercase', True):
            text = text.lower()
        
        return text.strip()
    
    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """Tokenize and lemmatize text.
        
        Args:
            text: Text to process.
            
        Returns:
            List of processed tokens.
        """
        if not text:
            return []
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords if configured
        if self.config.get('pipeline.preprocessing.remove_stopwords', True):
            tokens = [token for token in tokens if token.lower() not in self.stop_words]
        
        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(token.lower()) for token in tokens]
        
        # Filter out very short tokens
        tokens = [token for token in tokens if len(token) > 2]
        
        return tokens
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from case data.
        
        Args:
            df: DataFrame with case data.
            
        Returns:
            DataFrame with extracted features.
        """
        self.logger.info("Extracting features from case data")
        
        processed_df = df.copy()
        
        # Clean text fields
        processed_df['clean_title'] = processed_df['title'].apply(self.clean_text)
        processed_df['clean_opinion'] = processed_df['opinion_text'].apply(self.clean_text)
        
        # Text length features
        processed_df['title_length'] = processed_df['clean_title'].str.len()
        processed_df['opinion_length'] = processed_df['clean_opinion'].str.len()
        processed_df['opinion_word_count'] = processed_df['clean_opinion'].apply(
            lambda x: len(word_tokenize(x)) if x else 0
        )
        
        # Date features
        processed_df['date_decided'] = pd.to_datetime(processed_df['date_decided'], errors='coerce')
        processed_df['year'] = processed_df['date_decided'].dt.year
        processed_df['month'] = processed_df['date_decided'].dt.month
        processed_df['day_of_year'] = processed_df['date_decided'].dt.dayofyear
        
        # Source features
        processed_df['source_encoded'] = pd.Categorical(processed_df['source']).codes
        
        # Case type features
        processed_df['case_type_encoded'] = pd.Categorical(processed_df['case_type']).codes
        
        # Justice count (if available)
        processed_df['justice_count'] = processed_df['justices'].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )
        
        # Legal keywords features
        legal_keywords = [
            'constitutional', 'amendment', 'due process', 'equal protection',
            'commerce clause', 'first amendment', 'fourth amendment', 'federal',
            'jurisdiction', 'standing', 'precedent', 'statutory', 'regulation'
        ]
        
        for keyword in legal_keywords:
            processed_df[f'has_{keyword.replace(" ", "_")}'] = processed_df['clean_opinion'].str.contains(
                keyword, case=False, na=False
            ).astype(int)
        
        # Outcome encoding
        outcome_mapping = {'approved': 1, 'denied': 0, 'unknown': -1}
        processed_df['outcome_encoded'] = processed_df['outcome'].map(outcome_mapping)
        
        # Filter valid outcomes for training
        processed_df = processed_df[processed_df['outcome_encoded'].isin([0, 1])].copy()
        
        return processed_df
    
    def filter_quality_cases(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter cases based on quality criteria.
        
        Args:
            df: DataFrame with case data.
            
        Returns:
            Filtered DataFrame.
        """
        initial_count = len(df)
        
        # Filter by text length
        min_length = self.config.get('pipeline.preprocessing.min_text_length', 50)
        max_length = self.config.get('pipeline.preprocessing.max_text_length', 5000)
        
        df = df[
            (df['opinion_length'] >= min_length) & 
            (df['opinion_length'] <= max_length)
        ].copy()
        
        # Remove cases without valid dates
        df = df[df['date_decided'].notna()].copy()
        
        # Remove cases without clear outcomes
        df = df[df['outcome'].isin(['approved', 'denied'])].copy()
        
        # Remove duplicate cases (based on title and date)
        df = df.drop_duplicates(subset=['clean_title', 'date_decided']).copy()
        
        final_count = len(df)
        self.logger.info(f"Filtered {initial_count} cases to {final_count} high-quality cases")
        
        return df
    
    def create_train_test_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create train/validation/test splits.
        
        Args:
            df: Processed DataFrame.
            
        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        train_split = self.config.get('data.train_split', 0.8)
        val_split = self.config.get('data.val_split', 0.1)
        test_split = self.config.get('data.test_split', 0.1)
        
        # Stratified split to maintain outcome distribution
        train_df, temp_df = train_test_split(
            df, 
            test_size=(1 - train_split),
            stratify=df['outcome_encoded'],
            random_state=42
        )
        
        # Calculate validation and test proportions from the remaining data
        val_proportion = val_split / (val_split + test_split)
        
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_proportion),
            stratify=temp_df['outcome_encoded'],
            random_state=42
        )
        
        self.logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def save_processed_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """Save processed data to files.
        
        Args:
            train_df: Training data.
            val_df: Validation data.
            test_df: Test data.
        """
        output_path = Path(self.config.get('data.processed_data_path'))
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save as CSV
        train_df.to_csv(output_path / 'train.csv', index=False)
        val_df.to_csv(output_path / 'val.csv', index=False)
        test_df.to_csv(output_path / 'test.csv', index=False)
        
        # Save as parquet for faster loading
        train_df.to_parquet(output_path / 'train.parquet', index=False)
        val_df.to_parquet(output_path / 'val.parquet', index=False)
        test_df.to_parquet(output_path / 'test.parquet', index=False)
        
        self.logger.info(f"Saved processed data to {output_path}")
    
    def process_full_pipeline(self, data_path: str = None):
        """Run the complete data processing pipeline.
        
        Args:
            data_path: Path to raw data file.
        """
        self.logger.info("Starting full data processing pipeline")
        
        # Load raw data
        df = self.load_raw_data(data_path)
        if df.empty:
            self.logger.error("No data loaded. Exiting.")
            return
        
        # Extract features
        df = self.extract_features(df)
        
        # Filter quality cases
        df = self.filter_quality_cases(df)
        
        if len(df) < 10:
            self.logger.error("Insufficient high-quality cases for training. Exiting.")
            return
        
        # Create splits
        train_df, val_df, test_df = self.create_train_test_split(df)
        
        # Save processed data
        self.save_processed_data(train_df, val_df, test_df)
        
        # Generate summary statistics
        self._generate_summary_stats(train_df, val_df, test_df)
        
        self.logger.info("Data processing pipeline completed successfully")
    
    def _generate_summary_stats(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """Generate and log summary statistics."""
        self.logger.info("=== Data Processing Summary ===")
        
        for name, df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
            outcome_dist = df['outcome'].value_counts()
            self.logger.info(f"{name} set: {len(df)} cases")
            self.logger.info(f"  - Approved: {outcome_dist.get('approved', 0)}")
            self.logger.info(f"  - Denied: {outcome_dist.get('denied', 0)}")
            self.logger.info(f"  - Average opinion length: {df['opinion_length'].mean():.0f} chars")
        
        self.logger.info("================================")


if __name__ == "__main__":
    processor = SCOTUSDataProcessor()
    processor.process_full_pipeline() 
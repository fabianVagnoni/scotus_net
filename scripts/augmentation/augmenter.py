import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import logging
from typing import List, Optional, Union
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Augmenter:
    """
    A class for text augmentation using various NLP techniques.
    
    This class provides methods to augment text data using techniques like:
    - Word embedding substitution
    - Synonym replacement
    - Back translation
    - Summarization
    
    Attributes:
        augmentations (List[str]): List of augmentation techniques to apply
        iterations (int): Number of times to apply augmentations
        seed (int): Random seed for reproducibility
        verbose (bool): Whether to print verbose output
    """
    
    def __init__(self, 
                 augmentations: List[str], 
                 iterations: int = 1, 
                 seed: int = 42,
                 verbose: bool = True,
                 random_selection_prob: float = 0.5):
        """
        Initialize the Augmenter.
        
        Args:
            augmentations: List of augmentation techniques to use
            iterations: Number of times to apply augmentations
            seed: Random seed for reproducibility
            verbose: Whether to print verbose output
            random_selection_prob: Probability of selecting each augmentation in each iteration (0.0-1.0)
        """
        self.augmentations = augmentations
        self.iterations = iterations
        self.seed = seed
        self.verbose = verbose
        self.random_selection_prob = random_selection_prob
        
        # Set random seed for reproducibility
        random.seed(seed)
        
        # Validate augmentations
        valid_augmentations = [
            "word_embedding_augmentation",
            "synonym_augmentation", 
            "back_translation",
            "summarization"
        ]
        
        for aug in augmentations:
            if aug not in valid_augmentations:
                raise ValueError(f"Invalid augmentation '{aug}'. Valid options: {valid_augmentations}")
            elif aug == "word_embedding_augmentation":
                self.word_embedding_augmenter = naw.ContextualWordEmbsAug(
                    model_path='bert-base-uncased',
                    action='substitute',
                    aug_p=0.5  # Probability of word substitution
                )
            elif aug == "synonym_augmentation":
                self.synonym_augmenter = naw.SynonymAug(aug_src='wordnet', aug_p=0.7)
            elif aug == "back_translation":
                self.back_translation_augmenter = naw.BackTranslationAug(
                    from_model_name='facebook/wmt19-en-de',
                    to_model_name='facebook/wmt19-de-en'
                )
            elif aug == "summarization":
                self.summarization_augmenter = nas.AbstSummAug(model_path='t5-base', num_beam=1)

    def augment_sentence(self, sentence: str) -> List[str]:
        """
        Augment a single sentence with randomly selected augmentations.
        
        Args:
            sentence: The input sentence to augment
            
        Returns:
            List of augmented sentences including the original
        """
        if not sentence or not sentence.strip():
            return []
            
        augmented_sentences = [sentence]
        
        for iteration in range(self.iterations):
            sentence_copy = sentence
            
            # Randomly select which augmentations to apply in this iteration
            selected_augmentations = []
            for augmentation in self.augmentations:
                if random.random() < self.random_selection_prob:
                    selected_augmentations.append(augmentation)
            
            # If no augmentations were selected, skip this iteration
            if not selected_augmentations:
                if self.verbose:
                    logger.debug(f"Iteration {iteration + 1}: No augmentations selected, skipping")
                continue
            
            if self.verbose:
                logger.debug(f"Iteration {iteration + 1}: Selected augmentations: {selected_augmentations}")
            
            # Apply selected augmentations in random order
            random.shuffle(selected_augmentations)
            
            for augmentation in selected_augmentations:
                try:
                    augmented = self.augment(sentence_copy, augmentation)
                    if augmented and augmented != sentence_copy:
                        augmented_sentences.append(augmented)
                        sentence_copy = augmented  # Use augmented version for next augmentation
                except Exception as e:
                    if self.verbose:
                        logger.warning(f"Error applying {augmentation}: {e}")
                    continue
                    
        return augmented_sentences
    
    def augment_data(self, data: List[str]) -> List[str]:
        """
        Augment a list of sentences.
        
        Args:
            data: List of sentences to augment
            
        Returns:
            List of all augmented sentences
        """
        if not data:
            return []
            
        augmented_data = []
        total_sentences = len(data)
        
        for i, sentence in enumerate(data):
            if self.verbose and i % 100 == 0:
                logger.info(f"Augmenting sentence {i+1}/{total_sentences}")
                
            augmented_sentences = self.augment_sentence(sentence)
            augmented_data.extend(augmented_sentences)
            
        if self.verbose:
            logger.info(f"Augmentation complete. Original: {len(data)}, Augmented: {len(augmented_data)}")
            
        return augmented_data

    def augment(self, sentence: str, augmentation: str) -> str:
        """
        Apply a specific augmentation technique to a sentence.
        
        Args:
            sentence: The input sentence
            augmentation: The augmentation technique to apply
            
        Returns:
            The augmented sentence
        """
        if not sentence or not sentence.strip():
            return sentence
            
        if augmentation == "word_embedding_augmentation":
            try:
                augmented = self.word_embedding_augmenter.augment(sentence)
                return augmented[0] if isinstance(augmented, list) else augmented
            except Exception as e:
                if self.verbose:
                    logger.error(f"Error in word embedding augmentation: {e}")
                return sentence
        
        elif augmentation == "synonym_augmentation":
            try:
                augmented = self.synonym_augmenter.augment(sentence)
                return augmented[0] if isinstance(augmented, list) else augmented
            except Exception as e:
                if self.verbose:
                    logger.error(f"Error in synonym augmentation: {e}")
                return sentence
        
        elif augmentation == "back_translation":
            try:
                augmented = self.back_translation_augmenter.augment(sentence)
                return augmented[0] if isinstance(augmented, list) else augmented
            except Exception as e:
                if self.verbose:
                    logger.error(f"Error in back translation: {e}")
                return sentence
            
        elif augmentation == "summarization":
            try:
                augmented = self.summarization_augmenter.augment(sentence)
                return augmented[0] if isinstance(augmented, list) else augmented
            except Exception as e:
                if self.verbose:
                    logger.error(f"Error in summarization: {e}")
                return sentence

        else:
            raise ValueError(f"Augmentation '{augmentation}' not found")

# Convenience functions for easy use from other modules
def create_augmenter(augmentations: List[str], 
                    iterations: int = 1, 
                    seed: int = 42,
                    verbose: bool = True,
                    random_selection_prob: float = 0.5) -> Augmenter:
    """
    Create an Augmenter instance with the specified parameters.
    
    Args:
        augmentations: List of augmentation techniques
        iterations: Number of iterations
        seed: Random seed
        verbose: Whether to print verbose output
        random_selection_prob: Probability of selecting each augmentation in each iteration
        
    Returns:
        Configured Augmenter instance
    """
    return Augmenter(augmentations, iterations, seed, verbose, random_selection_prob)

def augment_text_list(texts: List[str], 
                     augmentations: List[str],
                     iterations: int = 1,
                     seed: int = 42,
                     verbose: bool = True,
                     random_selection_prob: float = 0.5) -> List[str]:
    """
    Convenience function to augment a list of texts.
    
    Args:
        texts: List of texts to augment
        augmentations: List of augmentation techniques
        iterations: Number of iterations
        seed: Random seed
        verbose: Whether to print verbose output
        random_selection_prob: Probability of selecting each augmentation in each iteration
        
    Returns:
        List of augmented texts
    """
    augmenter = Augmenter(augmentations, iterations, seed, verbose, random_selection_prob)
    return augmenter.augment_data(texts)
        

        
        
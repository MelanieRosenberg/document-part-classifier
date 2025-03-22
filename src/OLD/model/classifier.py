import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import logging
from typing import List, Dict, Optional, Set
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add valid tags set
VALID_TAGS = {
    'TEXT', 'FORM', 'TABLE',  # Primary tags
    'TITLE', 'SECTION_HEADER', 'PAGE_HEADER', 'PAGE_FOOTER',  # Structure tags
    'CAPTION', 'UNSPECIFIED'  # Other valid tags
}

class DocumentPartDataset(Dataset):
    """Dataset for document part classification"""
    
    def __init__(
        self,
        lines_file: str,
        tags_file: str,
        tokenizer: RobertaTokenizer,
        max_length: int = 512,
        label_to_id: Optional[Dict[str, int]] = None,
        context_lines: int = 1  # Number of lines before/after for context
    ):
        """
        Initialize dataset.
        
        Args:
            lines_file: Path to file containing text lines
            tags_file: Path to file containing corresponding tags
            tokenizer: RoBERTa tokenizer
            max_length: Maximum sequence length
            label_to_id: Mapping of tags to label IDs. If None, will be created from tags.
            context_lines: Number of lines before/after to use as context
        """
        # Load data
        with open(lines_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f]
        with open(tags_file, 'r', encoding='utf-8') as f:
            tags = [tag.strip() for tag in f]
            
        # Filter out invalid tags
        valid_indices = [i for i, tag in enumerate(tags) if tag in VALID_TAGS]
        self.lines = [lines[i] for i in valid_indices]
        self.tags = [tags[i] for i in valid_indices]
        
        logger.info(f"Loaded {len(lines)} lines, kept {len(self.lines)} with valid tags")
            
        assert len(self.lines) == len(self.tags), "Number of lines and tags must match"
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.context_lines = context_lines
        
        # Create label mapping if not provided
        if label_to_id is None:
            unique_tags = sorted(set(self.tags))
            self.label_to_id = {tag: i for i, tag in enumerate(unique_tags)}
        else:
            self.label_to_id = label_to_id
            
        self.id_to_label = {i: tag for tag, i in self.label_to_id.items()}
        logger.info(f"Label mapping: {self.label_to_id}")
        
    def get_line_with_context(self, idx: int) -> str:
        """Get a line with its surrounding context."""
        context_before = []
        context_after = []
        
        # Get previous lines as context
        for i in range(max(0, idx - self.context_lines), idx):
            context_before.append(f"[PREV] {self.lines[i]}")
            
        # Get following lines as context
        for i in range(idx + 1, min(len(self.lines), idx + self.context_lines + 1)):
            context_after.append(f"[NEXT] {self.lines[i]}")
            
        # Combine all parts with the current line in the middle
        full_text = "\n".join(
            context_before + 
            [f"[CURRENT] {self.lines[idx]}"] +
            context_after
        )
        
        return full_text
        
    def __len__(self):
        return len(self.lines)
        
    def __getitem__(self, idx):
        # Get text with context
        text = self.get_line_with_context(idx)
        tag = self.tags[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Convert to flat tensors
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.label_to_id[tag])
        }
        
        return item

def setup_model(num_labels: int):
    """
    Initialize RoBERTa model for sequence classification.
    
    Args:
        num_labels: Number of classification labels
        
    Returns:
        model: Initialized model
        tokenizer: Associated tokenizer
    """
    model_name = "roberta-base"
    
    # Initialize tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="single_label_classification"
    )
    
    return model, tokenizer

def create_dataloaders(
    data_dir: str,
    tokenizer: RobertaTokenizer,
    batch_size: int = 16,
    max_length: int = 512,
    context_lines: int = 1
) -> tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """
    Create DataLoaders for training, validation and test sets.
    
    Args:
        data_dir: Directory containing the data files
        tokenizer: RoBERTa tokenizer
        batch_size: Batch size for DataLoaders
        max_length: Maximum sequence length
        context_lines: Number of lines before/after to use as context
        
    Returns:
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        test_loader: DataLoader for test set
        label_to_id: Mapping of labels to IDs
    """
    # Create training dataset first to get label mapping
    train_dataset = DocumentPartDataset(
        os.path.join(data_dir, 'train_lines.txt'),
        os.path.join(data_dir, 'train_tags.txt'),
        tokenizer,
        max_length=max_length,
        context_lines=context_lines
    )
    
    # Use same label mapping for validation and test sets
    val_dataset = DocumentPartDataset(
        os.path.join(data_dir, 'val_lines.txt'),
        os.path.join(data_dir, 'val_tags.txt'),
        tokenizer,
        max_length=max_length,
        label_to_id=train_dataset.label_to_id,
        context_lines=context_lines
    )
    
    test_dataset = DocumentPartDataset(
        os.path.join(data_dir, 'test_lines.txt'),
        os.path.join(data_dir, 'test_tags.txt'),
        tokenizer,
        max_length=max_length,
        label_to_id=train_dataset.label_to_id,
        context_lines=context_lines
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader, train_dataset.label_to_id

if __name__ == "__main__":
    # Example setup (not training)
    data_dir = "document-part-classifier/data"
    
    # Initialize model and tokenizer
    num_labels = len(VALID_TAGS)  # Using our defined valid tags
    model, tokenizer = setup_model(num_labels)
    
    # Create dataloaders with context
    train_loader, val_loader, test_loader, label_to_id = create_dataloaders(
        data_dir,
        tokenizer,
        batch_size=16,
        context_lines=1  # Use 1 line before and after as context
    )
    
    logger.info("Model and data loaders set up successfully")
    logger.info(f"Number of labels: {num_labels}")
    logger.info(f"Label mapping: {label_to_id}")
    
    # Log dataset sizes
    logger.info(f"Training batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")
    
    # Test context format
    sample_batch = next(iter(train_loader))
    
    # Get sample text and its label
    sample_text = tokenizer.decode(sample_batch['input_ids'][0], skip_special_tokens=False)
    sample_label_id = sample_batch['labels'][0].item()
    sample_label = [k for k, v in label_to_id.items() if v == sample_label_id][0]
    
    # Clean up the output by removing padding tokens
    clean_text = sample_text.replace('<s>', '').replace('</s>', '').replace('<pad>', '').strip()
    
    logger.info("\nSample input with context:")
    logger.info(f"Label: {sample_label}")
    logger.info("Text:")
    logger.info(clean_text)
    
    # Show tokenization details
    tokens = tokenizer.tokenize(clean_text)
    logger.info(f"\nNumber of tokens: {len(tokens)}")
    logger.info(f"First 10 tokens: {tokens[:10]}") 
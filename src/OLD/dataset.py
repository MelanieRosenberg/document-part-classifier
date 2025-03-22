import os
from typing import List, Dict, Optional, Tuple
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import logging
from collections import defaultdict
from .preprocess import preprocess_document
from .constants import PRIMARY_TAGS, CONTEXT_TOKENS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentPartDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        tokenizer_name: str = "microsoft/deberta-v3-base",
        max_length: int = 512,
        label_map: Optional[Dict[str, int]] = None,
        context_window: int = 2
    ):
        """
        Initialize the dataset.
        Only includes documents containing primary tags, but preserves all tags in those documents.
        
        Args:
            data_dir: Directory containing document files
            tokenizer_name: Name of the pretrained tokenizer
            max_length: Maximum sequence length
            label_map: Optional mapping of labels to indices
            context_window: Number of lines before/after to use as context
        """
        self.data_dir = data_dir
        self.max_length = max_length
        self.context_window = context_window
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Get valid files and process them
        self.examples = []
        self.example_positions = []  # (doc_id, line_id) for each example
        self.doc_lines = {}  # doc_id -> list of lines
        self.doc_tags = {}   # doc_id -> list of tags
        unique_labels = set() if label_map is None else set(label_map.keys())
        
        # First pass: identify valid files (those containing primary tags)
        valid_files = []
        skipped_files = 0
        for fname in os.listdir(data_dir):
            file_path = os.path.join(data_dir, fname)
            try:
                lines, tags = preprocess_document(file_path)
                if not lines or not tags:
                    skipped_files += 1
                    continue
                    
                # Only include files that contain at least one primary tag
                if not any(tag in PRIMARY_TAGS for tag in tags):
                    skipped_files += 1
                    continue
                    
                valid_files.append(fname)
                if label_map is None:
                    # Collect ALL tags from valid documents
                    unique_labels.update(tags)
            except Exception as e:
                logger.error(f"Error processing {fname}: {str(e)}")
                skipped_files += 1
                continue
        
        logger.info(f"\nFile Processing:")
        logger.info(f"Valid files (containing primary tags): {len(valid_files)}")
        logger.info(f"Skipped files: {skipped_files}")
        
        if not valid_files:
            raise ValueError(f"No valid files with primary tags found in {data_dir}")
        
        # Create label map if not provided
        if label_map is None:
            self.label_map = {
                label: idx for idx, label in enumerate(sorted(unique_labels))
            }
        else:
            self.label_map = label_map
        
        # Second pass: load content from valid files
        for doc_id, fname in enumerate(valid_files):
            try:
                file_path = os.path.join(data_dir, fname)
                lines, tags = preprocess_document(file_path)
                
                # Store document structure
                self.doc_lines[doc_id] = lines
                self.doc_tags[doc_id] = tags
                
                # Create examples with document position info
                # Include ALL tags from valid documents
                for line_id, (text, tag) in enumerate(zip(lines, tags)):
                    if tag in self.label_map:
                        self.examples.append((text, tag))
                        self.example_positions.append((doc_id, line_id))
            except Exception as e:
                logger.error(f"Error processing {fname}: {str(e)}")
                continue
        
        if not self.examples:
            raise ValueError(f"No valid examples found in {data_dir}")
            
        logger.info(f"Loaded {len(self.examples)} examples")
        self._log_label_distribution()
    
    def _log_label_distribution(self):
        """Log distribution of labels in the dataset."""
        label_counts = defaultdict(int)
        for _, tag in self.examples:
            label_counts[tag] += 1
        
        total = len(self.examples)
        logger.info("\nLabel distribution:")
        for tag, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total) * 100
            is_primary = "* " if tag in PRIMARY_TAGS else "  "
            logger.info(f"{is_primary}{tag}: {count} ({percentage:.1f}%)")
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def get_context(
        self,
        doc_id: int,
        line_id: int,
        before: bool = True
    ) -> str:
        """Get context lines before or after the current line."""
        lines = self.doc_lines[doc_id]
        window = self.context_window
        
        if before:
            start = max(0, line_id - window)
            end = line_id
        else:
            start = line_id + 1
            end = min(len(lines), line_id + window + 1)
            
        if start >= end:
            return ""
            
        return f" {CONTEXT_TOKENS['sep']} ".join(lines[start:end])
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example with its context."""
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
            
        text, tag = self.examples[idx]
        doc_id, line_id = self.example_positions[idx]
        
        # Get context
        context_before = self.get_context(doc_id, line_id, before=True)
        context_after = self.get_context(doc_id, line_id, before=False)
        
        # Combine text with context
        parts = []
        if context_before:
            parts.append(context_before)
        parts.append(f"{CONTEXT_TOKENS['line']} {text}")
        if context_after:
            parts.append(context_after)
        
        full_text = f" {CONTEXT_TOKENS['sep']} ".join(parts)
        
        try:
            # Tokenize text
            encoding = self.tokenizer(
                full_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'labels': torch.tensor(self.label_map[tag]),
                'input_texts': text,
                'doc_id': torch.tensor(doc_id),
                'line_id': torch.tensor(line_id)
            }
            
        except Exception as e:
            logger.error(f"Error processing example {idx}: {str(e)}")
            raise
    
    @property
    def num_labels(self) -> int:
        """Get number of unique labels."""
        return len(self.label_map)
    
    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Custom collate function to handle variable length inputs."""
        if not batch:
            raise ValueError("Empty batch provided to collate_fn")
            
        # Sort batch by text length for more efficient processing
        batch = sorted(batch, key=lambda x: len(x['input_ids']), reverse=True)
        
        return {
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
            'labels': torch.stack([item['labels'] for item in batch]),
            'input_texts': [item['input_texts'] for item in batch],
            'doc_ids': torch.stack([item['doc_id'] for item in batch]),
            'line_ids': torch.stack([item['line_id'] for item in batch])
        }
    
    @classmethod
    def create_train_val_datasets(
        cls,
        train_dir: str,
        val_dir: str,
        tokenizer_name: str = "microsoft/deberta-v3-base",
        max_length: int = 512,
        context_window: int = 2
    ) -> Tuple['DocumentPartDataset', 'DocumentPartDataset']:
        """
        Create training and validation datasets with consistent label mapping.
        
        Args:
            train_dir: Directory containing training files
            val_dir: Directory containing validation files
            tokenizer_name: Name of the pretrained tokenizer
            max_length: Maximum sequence length
            context_window: Number of lines before/after to use as context
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        train_dataset = cls(
            train_dir,
            tokenizer_name=tokenizer_name,
            max_length=max_length,
            context_window=context_window
        )
        
        val_dataset = cls(
            val_dir,
            tokenizer_name=tokenizer_name,
            max_length=max_length,
            label_map=train_dataset.label_map,
            context_window=context_window
        )
        
        return train_dataset, val_dataset 
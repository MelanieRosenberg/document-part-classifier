import os
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import json
from typing import List, Tuple, Dict, Optional
import numpy as np

class DocumentPartDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        tokenizer_name: str,
        max_length: int = 512,
        context_window: int = 5
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing the data files
            tokenizer_name: Name of the tokenizer to use
            max_length: Maximum sequence length
            context_window: Number of lines to include before and after the target line
        """
        self.data_dir = data_dir
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.context_window = context_window
        
        # Load all examples
        self.examples = []
        self.label_map = {}
        self._load_data()
        
        # Create label to id mapping
        self.label_to_id = {label: idx for idx, label in enumerate(self.label_map)}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}
        
    def _load_data(self):
        """Load all examples from the data directory."""
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.json'):
                with open(os.path.join(self.data_dir, filename), 'r') as f:
                    data = json.load(f)
                    for example in data:
                        self.examples.append((example['text'], example['tag']))
                        if example['tag'] not in self.label_map:
                            self.label_map[example['tag']] = len(self.label_map)
    
    @classmethod
    def create_train_val_datasets(
        cls,
        train_dir: str,
        val_dir: str,
        tokenizer_name: str,
        max_length: int = 512,
        context_window: int = 5
    ) -> Tuple['DocumentPartDataset', 'DocumentPartDataset']:
        """
        Create training and validation datasets.
        
        Args:
            train_dir: Directory containing training data
            val_dir: Directory containing validation data
            tokenizer_name: Name of the tokenizer to use
            max_length: Maximum sequence length
            context_window: Number of lines to include before and after the target line
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        train_dataset = cls(train_dir, tokenizer_name, max_length, context_window)
        val_dataset = cls(val_dir, tokenizer_name, max_length, context_window)
        return train_dataset, val_dataset
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text, tag = self.examples[idx]
        
        # Get context lines
        start_idx = max(0, idx - self.context_window)
        end_idx = min(len(self.examples), idx + self.context_window + 1)
        context_lines = [self.examples[i][0] for i in range(start_idx, end_idx)]
        
        # Combine context lines
        context_text = "\n".join(context_lines)
        
        # Tokenize
        encoding = self.tokenizer(
            context_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Convert label to id
        label_id = self.label_to_id[tag]
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label_id, dtype=torch.long)
        }
    
    @property
    def num_labels(self) -> int:
        return len(self.label_map) 
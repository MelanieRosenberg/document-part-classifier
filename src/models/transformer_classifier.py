import os
import time
import logging
from typing import List, Tuple, Dict, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import RobertaTokenizer, RobertaModel, AdamW, get_linear_schedule_with_warmup
from torchcrf import CRF
from sklearn.metrics import classification_report, f1_score

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobertaWithCRF(nn.Module):
    def __init__(self, num_labels: int):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-large')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, List[List[int]]]:
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output)
        
        if labels is not None:
            # During training
            # Create mask for padding tokens
            mask = attention_mask.bool()
            
            # Filter out -100 from labels (ignored positions)
            # TorchCRF cannot handle -100 values
            valid_labels = labels.clone()
            valid_labels = torch.where(
                valid_labels == -100, 
                torch.zeros_like(valid_labels), 
                valid_labels
            )
            
            # CRF loss (negative log likelihood)
            log_likelihood = self.crf(emissions, valid_labels, mask=mask, reduction='mean')
            return -log_likelihood
        else:
            # During inference
            # Create mask for padding tokens
            mask = attention_mask.bool()
            # CRF decoding
            predictions = self.crf.decode(emissions, mask=mask)
            return predictions

class DocumentPartClassifier:
    def __init__(
        self,
        model_name: str = 'roberta-large',
        max_length: int = 512,
        batch_size: int = 32,
        learning_rate: float = 1e-5,
        num_epochs: int = 20,
        context_window: int = 2,
        warmup_ratio: float = 0.1
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.context_window = context_window
        self.warmup_ratio = warmup_ratio
        
        # Label mapping
        self.label_map = {'FORM': 0, 'TABLE': 1, 'TEXT': 2}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        
        # Initialize tokenizer and add special tokens
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        special_tokens = {
            'additional_special_tokens': ['[TARGET_START]', '[TARGET_END]']
        }
        self.tokenizer.add_special_tokens(special_tokens)
        
        # Check for CUDA availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        try:
            self.model = RobertaWithCRF(num_labels=len(self.label_map))
            # Resize token embeddings to account for new special tokens
            self.model.roberta.resize_token_embeddings(len(self.tokenizer))
            self.model.to(self.device)
            logger.info(f"Initialized classifier with {model_name} on {self.device}")
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise
        
    def prepare_data(
        self,
        lines: List[str],
        labels: Optional[List[str]] = None,
        context_window: Optional[int] = None
    ) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """
        Prepare data for model training/inference with context from surrounding lines.
        Uses special tokens to mark the target line vs context.
        
        Args:
            lines: List of text lines to classify
            labels: Optional list of labels for training
            context_window: Number of lines to include before and after each line
            
        Returns:
            Tuple of (encodings, labels)
        """
        # Use instance context_window if not provided
        context_window = context_window if context_window is not None else self.context_window
        
        # Prepare text with context
        contextualized_lines = []
        for i in range(len(lines)):
            # Get context window
            start_idx = max(0, i - context_window)
            end_idx = min(len(lines), i + context_window + 1)
            
            # Get lines in context window
            context_before = lines[start_idx:i]
            target_line = lines[i]
            context_after = lines[i+1:end_idx]
            
            # Join with special tokens to mark target line
            context_text = " [SEP] ".join(context_before)
            if context_before:
                context_text += " [SEP] "
            context_text += f"[TARGET_START] {target_line} [TARGET_END]"
            if context_after:
                context_text += " [SEP] " + " [SEP] ".join(context_after)
            
            contextualized_lines.append(context_text)
        
        # Tokenize all lines at once with context
        encodings = self.tokenizer(
            contextualized_lines,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        if labels is not None:
            # Convert string labels to tensor
            label_ids = torch.tensor([self.label_map[label] for label in labels])
            
            # For CRF, we need a label for each token
            # But we'll handle this differently - we'll only use the target line label
            # Create sequence labels where:
            # - Target line tokens get the actual label
            # - Context tokens get a -100 (will be ignored in loss calculation)
            
            # First, create a tensor filled with -100 (ignore index)
            expanded_labels = torch.full(
                (len(lines), encodings['input_ids'].size(1)),
                -100,
                dtype=torch.long
            )
            
            # Then, for each sequence, find target tokens and set their labels
            for i in range(len(lines)):
                # Get input_ids for this sequence
                input_ids = encodings['input_ids'][i]
                
                # Find target start and end token indices
                target_start_idx = (input_ids == self.tokenizer.convert_tokens_to_ids('[TARGET_START]')).nonzero(as_tuple=True)[0]
                target_end_idx = (input_ids == self.tokenizer.convert_tokens_to_ids('[TARGET_END]')).nonzero(as_tuple=True)[0]
                
                # If both tokens are found, set labels for tokens between them
                if len(target_start_idx) > 0 and len(target_end_idx) > 0:
                    start_idx = target_start_idx[0].item() + 1  # +1 to skip [TARGET_START]
                    end_idx = target_end_idx[0].item()  # Don't include [TARGET_END]
                    
                    # Set all tokens in the target span to the label
                    expanded_labels[i, start_idx:end_idx] = label_ids[i]
                else:
                    # Fallback to using first token if target markers not found
                    expanded_labels[i, 1] = label_ids[i]  # Use first non-special token
            
            return encodings, expanded_labels
        
        return encodings, None

    def create_data_loader(
        self,
        encodings: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        shuffle: bool = True
    ) -> DataLoader:
        """Create a data loader for the given encodings and labels."""
        if labels is not None:
            dataset = TensorDataset(
                encodings['input_ids'],
                encodings['attention_mask'],
                labels
            )
        else:
            dataset = TensorDataset(
                encodings['input_ids'],
                encodings['attention_mask']
            )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            pin_memory=(self.device.type == 'cuda')  # Enable pin_memory for CUDA for faster data transfer
        )

    def train(
        self,
        train_lines_file: str,
        train_tags_file: str,
        val_lines_file: str,
        val_tags_file: str,
        model_save_dir: str,
        early_stopping_patience: int = 3
    ) -> Dict[str, List[float]]:
        """
        Train the model and save it with early stopping.
        
        Args:
            train_lines_file: Path to training lines file
            train_tags_file: Path to training tags file
            val_lines_file: Path to validation lines file
            val_tags_file: Path to validation tags file
            model_save_dir: Directory to save the model
            early_stopping_patience: Number of epochs to wait for improvement before stopping
            
        Returns:
            Training history dictionary
        """
        # Load and preprocess data
        train_lines, train_tags = self.load_data(train_lines_file, train_tags_file)
        val_lines, val_tags = self.load_data(val_lines_file, val_tags_file)
        
        # Prepare data with context
        train_encodings, train_labels = self.prepare_data(
            train_lines,
            train_tags,
            context_window=self.context_window
        )
        val_encodings, val_labels = self.prepare_data(
            val_lines,
            val_tags,
            context_window=self.context_window
        )
        
        # Create data loaders
        train_loader = self.create_data_loader(train_encodings, train_labels)
        val_loader = self.create_data_loader(val_encodings, val_labels, shuffle=False)
        
        # Initialize optimizer
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        
        # Calculate total steps for scheduler
        total_steps = len(train_loader) * self.num_epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        
        # Add learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training history
        history = {
            'train_loss': [],
            'val_f1': []
        }
        
        # Early stopping variables
        best_val_f1 = 0
        no_improvement_count = 0
        best_model_state = None
        
        # Training loop
        logger.info(f"Starting training for {self.num_epochs} epochs")
        for epoch in range(self.num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)
                
                loss = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss.backward()
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            
            # Validation phase
            val_f1, val_report = self.evaluate(val_loader, return_report=True)
            history['val_f1'].append(val_f1)
            
            logger.info(
                f"Epoch {epoch+1}/{self.num_epochs} - "
                f"Training Loss: {avg_train_loss:.4f}, "
                f"Validation F1: {val_f1:.4f}"
            )
            logger.info(f"Validation Report:\n{val_report}")
            
            # Early stopping check
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                no_improvement_count = 0
                best_model_state = self.model.state_dict().copy()
                logger.info(f"New best validation F1: {best_val_f1:.4f}")
            else:
                no_improvement_count += 1
                logger.info(f"No improvement for {no_improvement_count} epochs")
                
                if no_improvement_count >= early_stopping_patience:
                    logger.info(f"Early stopping after {epoch+1} epochs")
                    break
        
        # Load best model state
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info(f"Loaded best model state with validation F1: {best_val_f1:.4f}")
        
        # Save the model
        os.makedirs(model_save_dir, exist_ok=True)
        model_path = os.path.join(model_save_dir, "document_part_classifier.pt")
        
        # Save model with complete configuration
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'tokenizer_name': self.model_name,  # Save tokenizer info for reproducibility
            'label_map': self.label_map,
            'reverse_label_map': self.reverse_label_map,
            'context_window': self.context_window,
            'max_length': self.max_length
        }, model_path)
        logger.info(f"Model saved to {model_path}")
        
        return history

    def evaluate(
        self, 
        val_loader: DataLoader,
        return_report: bool = False
    ) -> Union[float, Tuple[float, str]]:
        """
        Evaluate the model on validation data.
        
        Args:
            val_loader: DataLoader with validation data
            return_report: Whether to return classification report
            
        Returns:
            F1 score or tuple of (F1 score, classification report)
        """
        self.model.eval()
        all_preds = []
        all_true_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Move to device
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                
                if len(batch) > 2:
                    labels = batch[2].to(self.device)
                    # Get true labels - we'll use the first target token for each sequence
                    for i in range(labels.size(0)):
                        # Find first non-ignored position
                        non_ignored = (labels[i] != -100).nonzero(as_tuple=True)[0]
                        if len(non_ignored) > 0:
                            # Get the label at this position
                            all_true_labels.append(labels[i, non_ignored[0]].item())
                        else:
                            # If all positions are ignored, skip this example
                            logger.warning(f"Sequence {i} has all positions ignored in evaluation")
                            all_true_labels.append(0)  # Default to first class
                
                # Get predictions
                predictions = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Extract predictions - each batch prediction is a list of label indices
                # For each sequence in the batch, get the majority predicted class
                for pred_sequence in predictions:
                    # Filter out padding (-100)
                    filtered_preds = [p for p in pred_sequence if p >= 0 and p < len(self.label_map)]
                    if filtered_preds:
                        # Get most common prediction (majority vote)
                        counts = {}
                        for p in filtered_preds:
                            counts[p] = counts.get(p, 0) + 1
                        majority_class = max(counts.items(), key=lambda x: x[1])[0]
                        all_preds.append(majority_class)
                    else:
                        # Default to first class if all predictions were filtered
                        all_preds.append(0)
        
        # Calculate metrics
        labels = [self.reverse_label_map[i] for i in all_true_labels]
        preds = [self.reverse_label_map[i] for i in all_preds]
        
        f1 = f1_score(labels, preds, average='weighted')
        
        if return_report:
            report = classification_report(labels, preds)
            return f1, report
        
        return f1

    def predict(self, lines: List[str]) -> Tuple[List[str], float]:
        """
        Predict document parts and measure inference time.
        
        Args:
            lines: List of text lines to classify
            
        Returns:
            Tuple of (predicted_labels, inference_time)
        """
        self.model.eval()
        start_time = time.time()
        
        # Prepare data with context
        encodings, _ = self.prepare_data(
            lines,
            context_window=self.context_window
        )
        
        # Create data loader
        data_loader = self.create_data_loader(encodings, shuffle=False)
        
        all_preds = []
        with torch.no_grad():
            for batch in data_loader:
                # Move to device
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                
                # Get predictions
                predictions = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Extract predictions - each batch prediction is a list of label indices
                # For each sequence in the batch, get the majority predicted class
                for pred_sequence in predictions:
                    # Filter out padding (-100)
                    filtered_preds = [p for p in pred_sequence if p >= 0 and p < len(self.label_map)]
                    if filtered_preds:
                        # Get most common prediction (majority vote)
                        counts = {}
                        for p in filtered_preds:
                            counts[p] = counts.get(p, 0) + 1
                        majority_class = max(counts.items(), key=lambda x: x[1])[0]
                        all_preds.append(majority_class)
                    else:
                        # Default to first class if all predictions were filtered
                        all_preds.append(0)
        
        # Convert numeric predictions to label strings
        predicted_labels = [self.reverse_label_map[pred] for pred in all_preds]
        inference_time = time.time() - start_time
        
        return predicted_labels, inference_time

    def format_document(self, lines: List[str], predictions: List[str]) -> str:
        """
        Format a document with XML tags based on predictions.
        
        Args:
            lines: List of text lines
            predictions: List of predicted tags
            
        Returns:
            Formatted document string with XML tags
        """
        result = []
        current_tag = None
        current_content = []
        
        for line, tag in zip(lines, predictions):
            if tag != current_tag:
                # Close previous tag if exists
                if current_tag is not None:
                    result.append(f"<{current_tag}>")
                    result.extend(current_content)
                    result.append(f"</{current_tag}>")
                    
                # Start new tag
                current_tag = tag
                current_content = [line]
            else:
                # Continue with current tag
                current_content.append(line)
        
        # Add the last section
        if current_tag is not None:
            result.append(f"<{current_tag}>")
            result.extend(current_content)
            result.append(f"</{current_tag}>")
        
        return "\n".join(result)

    def load_saved_model(self, model_path: str) -> None:
        """
        Load a saved model.
        
        Args:
            model_path: Path to the saved model file
        """
        try:
            # Load on CPU first to handle possible device mismatch
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Update instance variables from checkpoint
            self.label_map = checkpoint['label_map']
            self.reverse_label_map = checkpoint['reverse_label_map']
            self.context_window = checkpoint.get('context_window', self.context_window)
            self.max_length = checkpoint.get('max_length', self.max_length)
            
            # Check if tokenizer name is in checkpoint and update if needed
            tokenizer_name = checkpoint.get('tokenizer_name', self.model_name)
            if tokenizer_name != self.model_name:
                logger.info(f"Updating tokenizer from {self.model_name} to {tokenizer_name}")
                self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
                
                # Add special tokens if loading an older model that didn't have them
                special_tokens = {
                    'additional_special_tokens': ['[TARGET_START]', '[TARGET_END]']
                }
                num_added = self.tokenizer.add_special_tokens(special_tokens)
                logger.info(f"Added {num_added} special tokens to tokenizer")
                
                self.model_name = tokenizer_name
            
            # Initialize model with correct number of labels
            self.model = RobertaWithCRF(num_labels=len(self.label_map))
            
            # Resize token embeddings if needed
            self.model.roberta.resize_token_embeddings(len(self.tokenizer))
            
            # Load state dict
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Move model to correct device
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Loaded saved model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def load_data(self, lines_file: str, tags_file: str) -> Tuple[List[str], List[str]]:
        """
        Load data from separate lines and tags files.
        
        Args:
            lines_file: Path to file containing document lines
            tags_file: Path to file containing line tags
            
        Returns:
            Tuple of (lines, tags)
        """
        try:
            with open(lines_file, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines()]
            
            with open(tags_file, 'r', encoding='utf-8') as f:
                tags = [tag.strip() for tag in f.readlines()]
            
            # Validate that the number of lines and tags match
            if len(lines) != len(tags):
                logger.warning(
                    f"Warning: Number of lines ({len(lines)}) doesn't match "
                    f"number of tags ({len(tags)})"
                )
                # Use the smaller length to avoid index errors
                min_len = min(len(lines), len(tags))
                lines = lines[:min_len]
                tags = tags[:min_len]
            
            # Validate tags
            for i, tag in enumerate(tags):
                if tag not in self.label_map:
                    logger.warning(f"Warning: Invalid tag '{tag}' at line {i+1}")
                    tags[i] = 'TEXT'  # Default to TEXT for invalid tags
            
            return lines, tags
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
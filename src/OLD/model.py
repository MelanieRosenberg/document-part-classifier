from typing import Dict, Optional, Tuple, List
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, PreTrainedModel
from torchcrf import CRF
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import re

# Target tags that need 90% F1 score
TARGET_TAGS = ['TEXT', 'TABLE', 'FORM']

class DocumentClassifier(PreTrainedModel):
    """Document part classifier with LoRA for efficient line-by-line classification."""
    
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        dropout: float = 0.1,
        class_weights: Optional[torch.Tensor] = None,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        label_names: Optional[List[str]] = None,
        context_window: int = 2
    ):
        """
        Initialize the document classifier.
        
        Args:
            model_name: Name of the pretrained model
            num_labels: Number of output labels
            dropout: Dropout probability
            class_weights: Optional tensor of class weights for loss function
            use_lora: Whether to use LoRA adapters
            lora_r: Rank dimension for LoRA
            lora_alpha: Alpha parameter for LoRA scaling
            lora_dropout: Dropout probability for LoRA layers
            label_names: Optional list of label names
            context_window: Number of lines before/after to use as context
        """
        # Load config
        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        super().__init__(config)
        
        # Store label names and context window
        self.label_names = label_names
        self.context_window = context_window
        
        # Load base model
        self.base_model = AutoModel.from_pretrained(model_name)
        
        # Classifier head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            self.base_model.config.hidden_size,
            num_labels
        )
        
        # Apply LoRA if requested
        if use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.SEQUENCE_CLASSIFICATION,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules=["query", "key", "value"]
            )
            self.base_model = get_peft_model(self.base_model, lora_config)
            self.base_model.print_trainable_parameters()
        
        # Loss function with class weights if provided
        self.class_weights = class_weights
        if class_weights is not None:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Optional labels for computing loss
            
        Returns:
            Dictionary containing loss, logits, and predictions
        """
        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # Get sequence output and pool [CLS] token
        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0]
        
        # Apply dropout and get logits
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # Get predictions
        predictions = torch.argmax(logits, dim=-1)
        
        # Prepare output dict
        output_dict = {
            'logits': logits,
            'predictions': predictions
        }
        
        # Compute loss if labels provided
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            output_dict['loss'] = loss
        
        return output_dict
    
    @staticmethod
    def add_context_to_line(
        lines: List[str],
        index: int,
        window_size: int
    ) -> str:
        """Add context from surrounding lines to the current line."""
        # Get context windows
        start_idx = max(0, index - window_size)
        end_idx = min(len(lines), index + window_size + 1)
        
        # Get context lines
        prev_lines = lines[start_idx:index]
        next_lines = lines[index+1:end_idx]
        current_line = lines[index]
        
        # Combine with special tokens
        context = []
        if prev_lines:
            context.append("[PREV] " + " ".join(prev_lines))
        context.append("[LINE] " + current_line)
        if next_lines:
            context.append("[NEXT] " + " ".join(next_lines))
        
        return " ".join(context)
    
    @staticmethod
    def smooth_predictions(
        lines: List[str],
        predictions: List[str]
    ) -> List[str]:
        """Apply smoothing rules to predictions."""
        smoothed = predictions.copy()
        
        # Rule 1: Avoid single-line tag switches
        for i in range(1, len(smoothed) - 1):
            if (smoothed[i-1] == smoothed[i+1] and 
                smoothed[i] != smoothed[i-1]):
                smoothed[i] = smoothed[i-1]
        
        # Rule 2: Form elements tend to continue
        for i in range(1, len(smoothed)):
            if (smoothed[i-1] == 'FORM' and 
                DocumentClassifier.has_form_features(lines[i]) and 
                smoothed[i] != 'FORM'):
                smoothed[i] = 'FORM'
        
        # Rule 3: Tables usually have consistent structure
        for i in range(1, len(smoothed)):
            if (smoothed[i-1] == 'TABLE' and 
                DocumentClassifier.has_table_features(lines[i]) and 
                smoothed[i] != 'TABLE'):
                smoothed[i] = 'TABLE'
        
        return smoothed
    
    @staticmethod
    def has_form_features(line: str) -> bool:
        """Check if a line has form-like characteristics."""
        patterns = [
            r'^\s*[A-Za-z]+[\s_:]+$',  # Field label
            r'[\._]{3,}',              # Underscores or dots for fill-in
            r'^\s*\([X ]\)',           # Checkbox
            r'^\s*□',                  # Empty checkbox
            r'^\s*[A-Za-z]+:$'         # Field with colon
        ]
        return any(re.search(pattern, line) for pattern in patterns)
    
    @staticmethod
    def has_table_features(line: str) -> bool:
        """Check if a line has table-like characteristics."""
        patterns = [
            r'\|\s*\|',                # Multiple vertical bars
            r'[+\-=]{3,}',            # Horizontal lines
            r'^\s*\|.*\|\s*$',        # Line enclosed in vertical bars
            r'^\s*[A-Za-z0-9]+\t'     # Tab-separated values
        ]
        return any(re.search(pattern, line) for pattern in patterns)
    
    @staticmethod
    def reconstruct_document(
        lines: List[str],
        predicted_tags: List[str]
    ) -> str:
        """Convert line-by-line predictions back to tagged document format."""
        doc_parts = []
        current_tag = None
        current_content = []
        
        for line, tag in zip(lines, predicted_tags):
            if tag != current_tag:
                # Close previous section
                if current_tag is not None:
                    doc_parts.append(f"</{current_tag}>")
                
                # Start new section
                doc_parts.append(f"<{tag}>")
                current_tag = tag
            
            # Add content line
            doc_parts.append(line)
        
        # Close final section
        if current_tag is not None:
            doc_parts.append(f"</{current_tag}>")
        
        return "\n".join(doc_parts)
    
    @staticmethod
    def compute_class_weights(
        labels: np.ndarray,
        label_names: List[str],
        primary_weight_multiplier: float = 2.0
    ) -> torch.Tensor:
        """
        Compute balanced class weights with higher weights for primary tags.
        
        Args:
            labels: Array of label indices
            label_names: List of label names
            primary_weight_multiplier: Factor to multiply primary tag weights by
            
        Returns:
            Tensor of class weights
        """
        # Compute balanced weights
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(labels),
            y=labels
        )
        
        # Convert to tensor
        weights_tensor = torch.tensor(class_weights, dtype=torch.float)
        
        # Increase weights for primary tags
        for i, label in enumerate(label_names):
            if label in TARGET_TAGS:
                weights_tensor[i] *= primary_weight_multiplier
        
        return weights_tensor
    
    @staticmethod
    def compute_metrics(
        predictions: np.ndarray,
        labels: np.ndarray,
        label_names: Optional[list] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute classification metrics including per-class F1 scores.
        
        Args:
            predictions: Predicted label indices
            labels: True label indices
            label_names: Optional list of label names
            
        Returns:
            Dictionary containing classification metrics
        """
        return classification_report(
            labels,
            predictions,
            target_names=label_names,
            output_dict=True
        )
    
    def save_pretrained(self, output_dir: str, **kwargs):
        """Save model to output directory."""
        # Save config
        self.config.save_pretrained(output_dir)
        
        # Save model state and metadata
        state_dict = {
            'model_state': self.state_dict(),
            'class_weights': self.class_weights,
            'label_names': self.label_names
        }
        torch.save(state_dict, f"{output_dir}/pytorch_model.bin")
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        num_labels: int,
        **kwargs
    ) -> 'DocumentClassifier':
        """Load model from saved state."""
        # Load config
        config = AutoConfig.from_pretrained(model_path)
        
        # Load state dict
        state_dict = torch.load(
            f"{model_path}/pytorch_model.bin",
            map_location='cpu'
        )
        
        # Initialize model with saved metadata
        model = cls(
            model_name=model_path,
            num_labels=num_labels,
            class_weights=state_dict.get('class_weights'),
            label_names=state_dict.get('label_names'),
            **kwargs
        )
        
        # Load model state
        model.load_state_dict(state_dict['model_state'])
        
        return model
    
    def get_attention_weights(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Get attention weights for visualization."""
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True
        )
        return outputs.attentions
    
    def _get_transformer_outputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get transformer outputs with proper handling of different model types."""
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if hasattr(self.base_model, 'roberta') else None
        )
        return outputs.last_hidden_state
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        features: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with feature fusion and CRF/softmax output.
        
        Args:
            input_ids: Tensor of token ids
            attention_mask: Tensor of attention mask
            features: Tensor of extracted features
            token_type_ids: Optional tensor of token type ids
            labels: Optional tensor of labels for training
            
        Returns:
            Dictionary containing loss and/or predictions
        """
        # Get transformer outputs
        transformer_outputs = self._get_transformer_outputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Process features
        processed_features = features
        # Expand features to match sequence length
        processed_features = processed_features.unsqueeze(1).expand(
            -1, transformer_outputs.size(1), -1
        )
        
        # Combine transformer outputs with features
        combined = torch.cat([transformer_outputs, processed_features], dim=-1)
        fused = self.fusion(combined)
        
        # Get logits
        logits = self.classifier(fused)
        
        outputs = {"logits": logits}
        
        # Handle loss calculation
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            outputs["loss"] = loss
        
        # Add predictions during inference
        if not self.training:
            predictions = torch.argmax(logits, dim=-1)
            outputs["predictions"] = predictions
        
        return outputs
    
    @staticmethod
    def create_model_with_ensemble(
        model_names: List[str],
        num_labels: int,
        feature_size: int,
        **kwargs
    ) -> nn.ModuleList:
        """Create an ensemble of models with different architectures."""
        models = nn.ModuleList([
            DocumentClassifier(
                model_name=model_name,
                num_labels=num_labels,
                **kwargs
            )
            for model_name in model_names
        ])
        return models 
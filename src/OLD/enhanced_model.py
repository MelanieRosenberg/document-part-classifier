import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Dict, List, Optional
import numpy as np
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

class EnhancedDocumentClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        class_weights: Optional[torch.Tensor] = None,
        label_names: Optional[List[str]] = None,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        context_window: int = 5
    ):
        """
        Enhanced document classifier with LoRA fine-tuning and contextual prediction smoothing.
        
        Args:
            model_name: Name of the base transformer model
            num_labels: Number of classification labels
            class_weights: Optional tensor of class weights for loss calculation
            label_names: Optional list of label names
            use_lora: Whether to use LoRA for efficient fine-tuning
            lora_r: LoRA attention dimension
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout rate
            context_window: Number of lines to consider for context
        """
        super().__init__()
        
        # Load base model and config
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Apply LoRA if requested
        if use_lora:
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                inference_mode=False,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["query", "key", "value"]
            )
            self.transformer = get_peft_model(self.transformer, peft_config)
            
        # Classification head
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        
        # Loss function with class weights
        self.class_weights = class_weights
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Store label names and context window
        self.label_names = label_names
        self.context_window = context_window
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with optional loss calculation."""
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # Get CLS token representation
        pooled_output = outputs[0][:, 0]
        pooled_output = self.dropout(pooled_output)
        
        # Get logits
        logits = self.classifier(pooled_output)
        
        # Get predictions
        predictions = torch.argmax(logits, dim=-1)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)
            
        return {
            'loss': loss,
            'logits': logits,
            'predictions': predictions
        }
        
    def smooth_predictions(
        self,
        texts: List[str],
        raw_predictions: List[str],
        window_size: int = None
    ) -> List[str]:
        """
        Apply smoothing to raw predictions using a sliding window.
        
        Args:
            texts: List of input texts
            raw_predictions: List of raw model predictions
            window_size: Optional window size override
            
        Returns:
            List of smoothed predictions
        """
        if window_size is None:
            window_size = self.context_window
            
        smoothed_predictions = []
        for i in range(len(raw_predictions)):
            # Get window indices
            start_idx = max(0, i - window_size)
            end_idx = min(len(raw_predictions), i + window_size + 1)
            
            # Get predictions in window
            window_predictions = raw_predictions[start_idx:end_idx]
            
            # Count occurrences
            prediction_counts = {}
            for pred in window_predictions:
                prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
                
            # Get most common prediction
            smoothed_pred = max(prediction_counts.items(), key=lambda x: x[1])[0]
            smoothed_predictions.append(smoothed_pred)
            
        return smoothed_predictions
    
    @staticmethod
    def compute_class_weights(
        labels: np.ndarray,
        label_names: List[str],
        primary_weight_multiplier: float = 2.0
    ) -> torch.Tensor:
        """
        Compute class weights with additional weight for primary tags.
        
        Args:
            labels: Array of label indices
            label_names: List of label names
            primary_weight_multiplier: Weight multiplier for primary tags
            
        Returns:
            Tensor of class weights
        """
        # Get class counts
        class_counts = np.bincount(labels)
        
        # Compute weights as inverse of frequency
        weights = 1.0 / class_counts
        weights = weights / np.sum(weights)
        
        # Apply multiplier for primary tags
        primary_tags = ['TEXT', 'TABLE', 'FORM']
        for i, name in enumerate(label_names):
            if name in primary_tags:
                weights[i] *= primary_weight_multiplier
                
        return torch.tensor(weights, dtype=torch.float32) 
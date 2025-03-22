import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import logging
from typing import Dict, List, Optional
import os

logger = logging.getLogger(__name__)

class LoRADocumentClassifier:
    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        num_labels: int = 3,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        learning_rate: float = 1e-4,
        batch_size: int = 8,
        num_epochs: int = 5,
        max_length: int = 512,
        context_window: int = 3,
        output_dir: str = "lora_document_classifier"
    ):
        """
        Initialize LoRA-based document classifier.
        
        Args:
            model_name: Base model to use
            num_labels: Number of classification labels
            lora_r: Low rank dimension
            lora_alpha: Alpha parameter for LoRA scaling
            lora_dropout: Dropout probability for LoRA layers
            learning_rate: Learning rate for training
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            max_length: Maximum sequence length
            context_window: Number of lines before/after for context
            output_dir: Directory to save model outputs
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.context_window = context_window
        self.output_dir = output_dir
        
        # Setup label mappings
        self.id2label = {0: "FORM", 1: "TABLE", 2: "TEXT"}
        self.label2id = {"FORM": 0, "TABLE": 1, "TEXT": 2}
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=self.id2label,
            label2id=self.label2id
        )
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "key", "value"],
            bias="none",
        )
        
        # Apply LoRA configuration
        self.model = get_peft_model(base_model, lora_config)
        logger.info(f"Trainable parameters: {self.model.print_trainable_parameters()}")
        
        # Training arguments
        self.training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=False,
            logging_dir=os.path.join(output_dir, "logs"),
            report_to="tensorboard",
        )
    
    def preprocess_data(self, lines: List[str], tags: Optional[List[str]] = None) -> Dataset:
        """
        Preprocess text data with context window.
        
        Args:
            lines: List of text lines
            tags: Optional list of tags (for training/evaluation)
            
        Returns:
            Preprocessed dataset
        """
        processed_texts = []
        processed_labels = []
        
        for i in range(len(lines)):
            # Get context window
            start_idx = max(0, i - self.context_window)
            end_idx = min(len(lines), i + self.context_window + 1)
            
            # Combine text with context
            context = lines[start_idx:end_idx]
            text = " [SEP] ".join(context)
            
            processed_texts.append(text)
            if tags is not None:
                processed_labels.append(self.label2id[tags[i]])
        
        # Create dataset
        data_dict = {"text": processed_texts}
        if tags is not None:
            data_dict["label"] = processed_labels
        
        dataset = Dataset.from_dict(data_dict)
        
        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
        
        dataset = dataset.map(tokenize_function, batched=True)
        return dataset
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average=None, labels=[0, 1, 2]
        )
        
        # Overall metrics
        overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        
        # Per-class metrics
        results = {
            "overall_f1": overall_f1,
            "form_precision": precision[0],
            "form_recall": recall[0],
            "form_f1": f1[0],
            "table_precision": precision[1],
            "table_recall": recall[1],
            "table_f1": f1[1],
            "text_precision": precision[2],
            "text_recall": recall[2],
            "text_f1": f1[2],
        }
        
        return results
    
    def train(
        self,
        train_lines: List[str],
        train_tags: List[str],
        val_lines: List[str],
        val_tags: List[str]
    ):
        """
        Train the model.
        
        Args:
            train_lines: Training text lines
            train_tags: Training tags
            val_lines: Validation text lines
            val_tags: Validation tags
        """
        # Prepare datasets
        train_dataset = self.preprocess_data(train_lines, train_tags)
        val_dataset = self.preprocess_data(val_lines, val_tags)
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )
        
        # Train
        logger.info("Starting training...")
        trainer.train()
        
        # Save the model
        self.save_model()
    
    def evaluate(self, test_lines: List[str], test_tags: List[str]) -> Dict:
        """
        Evaluate the model on test data.
        
        Args:
            test_lines: Test text lines
            test_tags: Test tags
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Prepare test dataset
        test_dataset = self.preprocess_data(test_lines, test_tags)
        
        # Initialize trainer for evaluation
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )
        
        # Evaluate
        metrics = trainer.evaluate(test_dataset)
        
        # Detailed evaluation for TABLE category
        table_metrics = self.evaluate_table_performance(test_dataset)
        metrics.update({"table_detailed": table_metrics})
        
        return metrics
    
    def evaluate_table_performance(self, test_dataset: Dataset) -> Dict:
        """
        Detailed evaluation focusing on TABLE category.
        
        Args:
            test_dataset: Test dataset
            
        Returns:
            Dictionary of TABLE-specific metrics
        """
        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
        )
        
        # Get predictions
        predictions = trainer.predict(test_dataset)
        preds = np.argmax(predictions.predictions, axis=-1)
        labels = predictions.label_ids
        
        # Calculate metrics for TABLE category
        table_indices = [i for i, label in enumerate(labels) if label == 1]
        table_predictions = [preds[i] for i in table_indices]
        
        # Calculate confusion matrix for TABLE
        TP = sum(1 for i, pred in enumerate(table_predictions) if pred == 1)
        FP = sum(1 for i, pred in enumerate(preds) if pred == 1 and labels[i] != 1)
        FN = sum(1 for i, pred in enumerate(preds) if pred != 1 and labels[i] == 1)
        
        # Calculate metrics
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Analyze error patterns
        error_indices = [i for i in table_indices if preds[i] != 1]
        misclassified_as = {}
        for i in error_indices:
            pred = preds[i]
            pred_label = self.id2label[pred]
            misclassified_as[pred_label] = misclassified_as.get(pred_label, 0) + 1
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "num_errors": len(error_indices),
            "error_distribution": misclassified_as
        }
    
    def save_model(self):
        """Save the model and tokenizer."""
        # Save LoRA model
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        # Merge and save full model
        logger.info("Merging LoRA weights with base model...")
        merged_model = self.model.merge_and_unload()
        merged_model.save_pretrained(os.path.join(self.output_dir, "merged"))
        logger.info(f"Merged model saved to {os.path.join(self.output_dir, 'merged')}")
    
    def load_model(self, model_path: str):
        """
        Load a saved model.
        
        Args:
            model_path: Path to the saved model
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path) 
import argparse
import os
import json
import logging
import time
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    TrainingArguments,
    Trainer
)
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import classification_report, precision_recall_fscore_support
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LoRATrainer:
    def __init__(
        self,
        base_model_name: str = "microsoft/deberta-v3-base",
        train_data_dir: str = "data/train",
        val_data_dir: str = "data/val",
        output_dir: str = "models/lora",
        context_window: int = 3,
        max_length: int = 512,
        device: Optional[str] = None,
        lora_config: Optional[Dict] = None
    ):
        """
        Initialize LoRA trainer.
        
        Args:
            base_model_name: Name of the base model to use
            train_data_dir: Directory containing training data
            val_data_dir: Directory containing validation data
            output_dir: Directory to save model outputs
            context_window: Size of context window
            max_length: Maximum sequence length
            device: Device to run training on (default: auto-detect)
            lora_config: LoRA configuration parameters
        """
        self.base_model_name = base_model_name
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.output_dir = Path(output_dir)
        self.context_window = context_window
        self.max_length = max_length
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup output directories
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = self.output_dir / f"run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging to file
        self.setup_logging()
        
        # Label mappings
        self.id2label = {0: "FORM", 1: "TABLE", 2: "TEXT"}
        self.label2id = {"FORM": 0, "TABLE": 1, "TEXT": 2}
        
        # Default LoRA config
        self.lora_config = lora_config or {
            "r": 16,  # LoRA attention dimension
            "lora_alpha": 32,  # Alpha scaling
            "lora_dropout": 0.1,  # Dropout probability
            "bias": "none",  # Bias type
            "target_modules": ["query", "key", "value"]  # Which modules to apply LoRA to
        }
        
        # Load data
        logger.info("Loading datasets...")
        self.train_dataset = self.load_data(self.train_data_dir)
        self.val_dataset = self.load_data(self.val_data_dir)
        
        # Initialize model components
        self.setup_model_components()
    
    def setup_logging(self) -> None:
        """Setup logging configuration."""
        log_file = self.run_dir / "training_log.txt"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    
    def load_data(self, data_dir: str) -> Dataset:
        """Load dataset from directory."""
        try:
            dataset = load_from_disk(data_dir)
            logger.info(f"Loaded {len(dataset)} examples from {data_dir}")
            return dataset
        except Exception as e:
            logger.error(f"Error loading data from {data_dir}: {e}")
            raise
    
    def setup_model_components(self) -> None:
        """Initialize model, tokenizer, and LoRA configuration."""
        logger.info(f"Loading base model and tokenizer from {self.base_model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        
        # Load base model
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_name,
            num_labels=len(self.id2label),
            id2label=self.id2label,
            label2id=self.label2id
        )
        
        # Setup LoRA configuration
        self.lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=self.lora_config["r"],
            lora_alpha=self.lora_config["lora_alpha"],
            lora_dropout=self.lora_config["lora_dropout"],
            bias=self.lora_config["bias"],
            target_modules=self.lora_config["target_modules"]
        )
        
        # Apply LoRA config to model
        self.model = get_peft_model(self.base_model, self.lora_config)
        logger.info("Model initialized with LoRA configuration")
    
    def preprocess_function(self, examples: Dict[str, List[Any]], context_window: int) -> Dict[str, torch.Tensor]:
        """Preprocess examples with context window."""
        processed_texts = []
        for i, text in enumerate(examples["text"]):
            # Get context window
            start_idx = max(0, i - context_window)
            end_idx = min(len(examples["text"]), i + context_window + 1)
            
            # Combine text with context
            context = examples["text"][start_idx:end_idx]
            processed_text = " [SEP] ".join(context)
            processed_texts.append(processed_text)
        
        # Tokenize
        tokenized = self.tokenizer(
            processed_texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        tokenized["labels"] = examples["label"]
        return tokenized
    
    @staticmethod
    def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """Compute evaluation metrics."""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        # Class-wise metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average=None, labels=[0, 1, 2]
        )
        
        # Overall metrics
        overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        
        return {
            "overall_f1": overall_f1,
            "table_f1": f1[1],  # TABLE class
            "form_f1": f1[0],   # FORM class
            "text_f1": f1[2],   # TEXT class
            "overall_precision": overall_precision,
            "overall_recall": overall_recall
        }
    
    def train(
        self,
        num_epochs: int = 10,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        eval_steps: int = 100,
        save_steps: int = 500,
        logging_steps: int = 50
    ) -> None:
        """Train the model."""
        logger.info("Starting training...")
        
        # Preprocess datasets
        train_processed = self.train_dataset.map(
            lambda examples: self.preprocess_function(examples, self.context_window),
            batched=True
        )
        val_processed = self.val_dataset.map(
            lambda examples: self.preprocess_function(examples, self.context_window),
            batched=True
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.run_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_dir=str(self.run_dir / "logs"),
            logging_steps=logging_steps,
            eval_steps=eval_steps,
            evaluation_strategy="steps",
            save_steps=save_steps,
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="table_f1",
            greater_is_better=True,
            remove_unused_columns=False
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_processed,
            eval_dataset=val_processed,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )
        
        # Train
        logger.info("Training model...")
        trainer.train()
        
        # Final evaluation
        logger.info("Running final evaluation...")
        eval_results = trainer.evaluate()
        
        # Save results
        results_file = self.run_dir / "val_metrics.json"
        with open(results_file, 'w') as f:
            json.dump(eval_results, f, indent=2)
        logger.info(f"Saved evaluation results to {results_file}")
        
        # Save model
        logger.info("Saving model...")
        trainer.save_model(str(self.run_dir / "model"))
        logger.info(f"Model saved to {self.run_dir / 'model'}")
        
        # Save training args
        args_file = self.run_dir / "training_args.json"
        with open(args_file, 'w') as f:
            json.dump(training_args.to_dict(), f, indent=2)
        logger.info(f"Saved training arguments to {args_file}")

def main():
    """Run training from command line."""
    parser = argparse.ArgumentParser(description="Train LoRA model for document part classification")
    
    # Model and data arguments
    parser.add_argument("--base_model", type=str, default="microsoft/deberta-v3-base",
                      help="Base model to use")
    parser.add_argument("--train_data_dir", type=str, required=True,
                      help="Directory containing training data")
    parser.add_argument("--val_data_dir", type=str, required=True,
                      help="Directory containing validation data")
    parser.add_argument("--output_dir", type=str, default="models/lora",
                      help="Directory to save model outputs")
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=10,
                      help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                      help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                      help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=500,
                      help="Number of warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                      help="Weight decay")
    parser.add_argument("--eval_steps", type=int, default=100,
                      help="Number of steps between evaluations")
    parser.add_argument("--save_steps", type=int, default=500,
                      help="Number of steps between model saves")
    parser.add_argument("--logging_steps", type=int, default=50,
                      help="Number of steps between logging")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=16,
                      help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=32,
                      help="LoRA alpha scaling")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                      help="LoRA dropout probability")
    
    # Other arguments
    parser.add_argument("--context_window", type=int, default=3,
                      help="Size of context window")
    parser.add_argument("--max_length", type=int, default=512,
                      help="Maximum sequence length")
    
    args = parser.parse_args()
    
    # Setup LoRA config
    lora_config = {
        "r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "bias": "none",
        "target_modules": ["query", "key", "value"]
    }
    
    # Initialize trainer
    trainer = LoRATrainer(
        base_model_name=args.base_model,
        train_data_dir=args.train_data_dir,
        val_data_dir=args.val_data_dir,
        output_dir=args.output_dir,
        context_window=args.context_window,
        max_length=args.max_length,
        lora_config=lora_config
    )
    
    # Train model
    trainer.train(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps
    )

if __name__ == "__main__":
    main() 
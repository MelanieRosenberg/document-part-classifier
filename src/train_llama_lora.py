import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from datasets import Dataset, load_from_disk
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import os
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LlamaLoRATrainer:
    def __init__(
        self,
        base_model_name: str = "/home/azureuser/llama_models/llama-3-2-3b",
        train_data_dir: str = "data/train",
        val_data_dir: str = "data/val",
        output_dir: str = "models/llama_lora",
        use_4bit: bool = True,
        max_length: int = 512,
        device: Optional[str] = None
    ):
        """
        Initialize Llama 3 LoRA trainer.
        
        Args:
            base_model_name: Absolute path to the local Llama 3 model
            train_data_dir: Directory containing training data
            val_data_dir: Directory containing validation data
            output_dir: Directory to save model outputs
            use_4bit: Whether to use 4-bit quantization
            max_length: Maximum sequence length
            device: Device to run training on (default: auto-detect)
        """
        self.base_model_name = base_model_name
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.output_dir = Path(output_dir)
        self.use_4bit = use_4bit
        self.max_length = max_length
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup output directories
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = self.output_dir / f"run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Label mappings
        self.id2label = {0: "FORM", 1: "TABLE", 2: "TEXT"}
        self.label2id = {"FORM": 0, "TABLE": 1, "TEXT": 2}
        
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
            # Load lines and tags from files
            lines_file = os.path.join(data_dir, "lines.txt")
            tags_file = os.path.join(data_dir, "tags.txt")
            
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
            
            # Convert tags to label IDs
            label_ids = [self.label2id[tag] for tag in tags]
            
            # Create dataset
            dataset = Dataset.from_dict({
                "text": lines,
                "label": label_ids
            })
            
            logger.info(f"Loaded {len(dataset)} examples from {data_dir}")
            return dataset
        except Exception as e:
            logger.error(f"Error loading data from {data_dir}: {e}")
            raise
    
    def setup_model_components(self) -> None:
        """Initialize model, tokenizer, and LoRA configuration."""
        logger.info(f"Loading base model and tokenizer from {self.base_model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with quantization if enabled
        if self.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            self.base_model = prepare_model_for_kbit_training(self.base_model)
        else:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                trust_remote_code=True
            )
            self.base_model = self.base_model.to(self.device)
        
        logger.info("Model loaded successfully")
    
    def create_prompt(self, text: str) -> str:
        """Create prompt for document classification."""
        return f"""<s>[INST] Classify the following document segment into one of these categories: FORM, TABLE, or TEXT.
The segment is delimited by triple backticks.
```
{text}
```
Classification: [/INST]"""
    
    def preprocess_function(self, examples: Dict[str, List[Any]]) -> Dict[str, torch.Tensor]:
        """Preprocess examples for causal LM training."""
        # Create prompts
        prompts = [self.create_prompt(text) for text in examples["text"]]
        
        # Tokenize prompts
        tokenized_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Prepare labels
        labels = []
        for label_id in examples["label"]:
            label_text = f" {self.id2label[label_id]}</s>"
            tokenized_label = self.tokenizer(label_text, return_tensors="pt")
            labels.append(tokenized_label.input_ids[0])
        
        # Prepare final examples
        tokenized_examples = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }
        
        for i, prompt_ids in enumerate(tokenized_inputs.input_ids):
            prompt_length = len(prompt_ids)
            label_ids = labels[i]
            
            # Combine prompt and label
            input_ids = torch.cat([prompt_ids, label_ids])
            attention_mask = torch.ones_like(input_ids)
            
            # Create labels with ignored prompt tokens
            label_ids_with_ignore = torch.cat([
                torch.ones(prompt_length, dtype=torch.long) * -100,
                label_ids
            ])
            
            tokenized_examples["input_ids"].append(input_ids)
            tokenized_examples["attention_mask"].append(attention_mask)
            tokenized_examples["labels"].append(label_ids_with_ignore)
        
        return tokenized_examples
    
    def train(
        self,
        num_epochs: int = 3,
        batch_size: int = 32,
        learning_rate: float = 2e-4,
        warmup_steps: int = 100,
        weight_decay: float = 0.01,
        gradient_accumulation_steps: int = 4,
        eval_steps: int = 50,
        save_steps: int = 200,
        logging_steps: int = 25,
        lora_r: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.1,
        early_stopping_patience: int = 3,
        early_stopping_threshold: float = 0.0,
        save_total_limit: Optional[int] = None
    ) -> None:
        """Train the model."""
        logger.info("Starting training...")
        
        # Setup LoRA configuration
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # Apply LoRA config to model
        self.model = get_peft_model(self.base_model, lora_config)
        logger.info(f"Trainable parameters: {self.model.print_trainable_parameters()}")
        
        # Preprocess datasets
        train_processed = self.train_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=self.train_dataset.column_names
        )
        val_processed = self.val_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=self.val_dataset.column_names
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.run_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
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
            metric_for_best_model="eval_overall_f1",
            greater_is_better=True,
            fp16=True,  # Use mixed precision
            remove_unused_columns=False,
            save_total_limit=save_total_limit
        )
        
        # Initialize trainer
        trainer = DocumentClassificationTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_processed,
            eval_dataset=val_processed,
            tokenizer=self.tokenizer
        )
        
        # Train
        logger.info("Training model...")
        trainer.train()
        
        # Save final model
        logger.info("Saving model...")
        trainer.save_model(str(self.run_dir / "final_model"))
        
        # Save training args
        args_file = self.run_dir / "training_args.json"
        with open(args_file, 'w') as f:
            training_args_dict = training_args.to_dict()
            training_args_dict.update({
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout
            })
            json.dump(training_args_dict, f, indent=2)
        logger.info(f"Saved training arguments to {args_file}")

class DocumentClassificationTrainer(Trainer):
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # Run standard evaluation
        output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Custom evaluation focusing on class predictions
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        
        # Get predictions
        predictions = self.predict(eval_dataset)
        
        # Process predictions to extract class labels
        predicted_labels = []
        true_labels = []
        
        for i, example in enumerate(eval_dataset):
            # Get actual label
            label_tokens = [j for j, id in enumerate(example["labels"]) if id != -100]
            if label_tokens:
                true_label_text = self.tokenizer.decode(example["labels"][label_tokens])
                for label_id, label_name in self.model.config.id2label.items():
                    if label_name in true_label_text:
                        true_labels.append(label_id)
                        break
            
            # Get predicted label
            logits = predictions.predictions[i]
            prompt_length = sum(1 for id in example["labels"] if id == -100)
            next_token_logits = logits[prompt_length - 1]
            predicted_token_id = np.argmax(next_token_logits)
            predicted_token = self.tokenizer.decode(predicted_token_id)
            
            # Map to class
            predicted_label = None
            for label_id, label_name in self.model.config.id2label.items():
                if label_name in predicted_token or predicted_token in label_name:
                    predicted_label = label_id
                    break
            
            if predicted_label is None:
                predicted_label = 2  # Default to TEXT
            
            predicted_labels.append(predicted_label)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predicted_labels, average=None, labels=[0, 1, 2]
        )
        
        # Overall metrics
        overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
            true_labels, predicted_labels, average='weighted'
        )
        
        # Add metrics to output
        output.update({
            f"{metric_key_prefix}_overall_f1": overall_f1,
            f"{metric_key_prefix}_form_f1": f1[0],
            f"{metric_key_prefix}_table_f1": f1[1],
            f"{metric_key_prefix}_text_f1": f1[2],
        })
        
        return output

def main():
    """Run training from command line."""
    parser = argparse.ArgumentParser(description="Train Llama 3 with LoRA for document classification")
    
    # Model and data arguments
    parser.add_argument("--base_model", type=str, default="/home/azureuser/llama_models/llama-3-2-3b",
                      help="Absolute path to the local Llama 3 model")
    parser.add_argument("--train_data_dir", type=str, required=True,
                      help="Directory containing training data")
    parser.add_argument("--val_data_dir", type=str, required=True,
                      help="Directory containing validation data")
    parser.add_argument("--output_dir", type=str, default="models/llama_lora",
                      help="Directory to save model outputs")
    
    # Training arguments - optimized for H100
    parser.add_argument("--num_epochs", type=int, default=3,
                      help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                      help="Training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                      help="Number of gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                      help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100,
                      help="Number of warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                      help="Weight decay")
    parser.add_argument("--eval_steps", type=int, default=50,
                      help="Number of steps between evaluations")
    parser.add_argument("--save_steps", type=int, default=200,
                      help="Number of steps between model saves")
    parser.add_argument("--logging_steps", type=int, default=25,
                      help="Number of steps between logging")
    parser.add_argument("--early_stopping_patience", type=int, default=3,
                      help="Number of evaluations to wait before early stopping")
    parser.add_argument("--early_stopping_threshold", type=float, default=0.0,
                      help="Threshold for early stopping")
    parser.add_argument("--save_total_limit", type=int, default=None,
                      help="Limit the total number of checkpoints to save")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=32,
                      help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=64,
                      help="LoRA alpha scaling")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                      help="LoRA dropout probability")
    
    # Other arguments
    parser.add_argument("--max_length", type=int, default=512,
                      help="Maximum sequence length")
    parser.add_argument("--use_4bit", type=bool, default=True,
                      help="Whether to use 4-bit quantization")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = LlamaLoRATrainer(
        base_model_name=args.base_model,
        train_data_dir=args.train_data_dir,
        val_data_dir=args.val_data_dir,
        output_dir=args.output_dir,
        use_4bit=args.use_4bit,
        max_length=args.max_length
    )
    
    # Train model
    trainer.train(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold,
        save_total_limit=args.save_total_limit
    )

if __name__ == "__main__":
    main() 
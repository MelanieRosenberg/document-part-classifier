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
from collections import defaultdict

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
        
        # Log validation dataset distribution
        val_distribution = defaultdict(int)
        for example in self.val_dataset:
            val_distribution[self.id2label[example["label"]]] += 1
        
        logger.info("\nValidation Set Distribution:")
        for label_name, count in val_distribution.items():
            logger.info(f"{label_name}: {count} lines")
        logger.info("")
        
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
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info("Tokenizer loaded successfully")
        
        # Load model with quantization if enabled
        if self.use_4bit:
            logger.info("Setting up 4-bit quantization config...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            logger.info("Loading model with 4-bit quantization...")
            
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            logger.info("Preparing model for k-bit training...")
            self.base_model = prepare_model_for_kbit_training(self.base_model)
        else:
            logger.info("Loading model without quantization...")
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
        save_total_limit: Optional[int] = None,
        eval_batch_size: int = 8,
        max_eval_samples: int = 500
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
        
        # Log processed validation dataset distribution
        processed_val_distribution = defaultdict(int)
        for example in val_processed:
            label_tokens = [j for j, id in enumerate(example["labels"]) if id != -100]
            if label_tokens:
                label_text = self.tokenizer.decode([example["labels"][i] for i in label_tokens])
                for label_id, label_name in self.id2label.items():
                    if label_name in label_text:
                        processed_val_distribution[label_name] += 1
                        break
        
        logger.info("\nProcessed Validation Set Distribution:")
        for label_name, count in processed_val_distribution.items():
            logger.info(f"{label_name}: {count} lines")
        logger.info("")
        
        # Subset validation dataset if needed, maintaining class proportions
        if max_eval_samples and len(val_processed) > max_eval_samples:
            logger.info(f"Subsetting validation dataset from {len(val_processed)} to {max_eval_samples} examples")
            
            # Group examples by label
            examples_by_label = defaultdict(list)
            for i, example in enumerate(val_processed):
                # Get the label from the example
                label_tokens = [j for j, id in enumerate(example["labels"]) if id != -100]
                if label_tokens:
                    label_text = self.tokenizer.decode([example["labels"][i] for i in label_tokens])
                    for label_id, label_name in self.id2label.items():
                        if label_name in label_text:
                            examples_by_label[label_id].append(i)
                            break
            
            # Calculate proportions from original dataset
            total = len(val_processed)
            proportions = {label: len(indices)/total for label, indices in examples_by_label.items()}
            
            # Sample proportionally
            sampled_indices = []
            for label_id, indices in examples_by_label.items():
                n_samples = min(int(proportions[label_id] * max_eval_samples), len(indices))
                sampled_indices.extend(indices[:n_samples])
            
            # Ensure we have exactly max_eval_samples
            if len(sampled_indices) > max_eval_samples:
                sampled_indices = sampled_indices[:max_eval_samples]
            
            # Create stratified subset
            val_processed = val_processed.select(sampled_indices)
            
            # Log the distribution of the subset
            subset_distribution = defaultdict(int)
            for example in val_processed:
                label_tokens = [j for j, id in enumerate(example["labels"]) if id != -100]
                if label_tokens:
                    label_text = self.tokenizer.decode([example["labels"][i] for i in label_tokens])
                    for label_id, label_name in self.id2label.items():
                        if label_name in label_text:
                            subset_distribution[label_name] += 1
                            break
            
            logger.info("Validation subset distribution:")
            for label_name, count in subset_distribution.items():
                logger.info(f"{label_name}: {count} lines")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.run_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=eval_batch_size,  # Use separate eval batch size
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
            save_total_limit=save_total_limit,
            report_to="none",  # Disable all reporting
            logging_first_step=True,  # Log the first step
            logging_nan_inf_filter=False,  # Show all logs including NaN/Inf
            disable_tqdm=True,  # Disable progress bars
            log_level="error"  # Only show errors
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add label mappings
        self.id2label = {0: "FORM", 1: "TABLE", 2: "TEXT"}
        self.label2id = {"FORM": 0, "TABLE": 1, "TEXT": 2}
        # Initialize loss tracking
        self.current_step_losses = []

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the training loss and track it for averaging.
        """
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        
        # Track loss for averaging
        self.current_step_losses.append(loss.item())
        
        # If this is the last batch of the step, log the average
        if len(self.current_step_losses) == self.args.gradient_accumulation_steps:
            avg_loss = sum(self.current_step_losses) / len(self.current_step_losses)
            if self.state.global_step % self.args.logging_steps == 0:
                logger.info(f"Step {self.state.global_step}: average loss = {avg_loss:.4f}")
            self.current_step_losses = []  # Reset for next step
        
        return (loss, outputs) if return_outputs else loss

    def compute_metrics(self, eval_pred):
        """
        Compute metrics for evaluation.
        """
        logits, labels = eval_pred
        
        # Debug the shapes
        logger.info(f"Logits shape: {logits.shape}, Labels shape: {labels.shape}")
        
        # Create arrays to store predictions and true labels
        predicted_labels = []
        true_labels = []
        
        # Process each example
        for i in range(len(labels)):
            # Get the true label from the ignored tokens (-100)
            true_label = None
            for label_id, label_name in self.id2label.items():
                # Look for the label text in non-ignored tokens
                label_tokens = labels[i] != -100
                if any(label_tokens):
                    label_text = self.tokenizer.decode(labels[i][label_tokens])
                    if label_name in label_text:
                        true_label = label_id
                        break
            
            if true_label is None:
                logger.warning(f"Could not find true label for example {i}")
                continue
            
            # For prediction, find the next token after prompt
            # Find index where -100 ends (this is where prompt ends)
            prompt_end_idx = 0
            while prompt_end_idx < len(labels[i]) and labels[i][prompt_end_idx] == -100:
                prompt_end_idx += 1
            
            # If we're at the end of the sequence, use the last token's logits
            if prompt_end_idx >= len(logits[i]):
                prompt_end_idx = len(logits[i]) - 1
            
            # Get the logits for the first token after the prompt
            token_logits = logits[i][prompt_end_idx]
            
            # Get the predicted token
            predicted_token_id = np.argmax(token_logits)
            predicted_token = self.tokenizer.decode(predicted_token_id)
            
            # Map predicted token to a class
            predicted_label = None
            for label_id, label_name in self.id2label.items():
                if label_name in predicted_token:
                    predicted_label = label_id
                    break
            
            # Default to most common class if no match
            if predicted_label is None:
                predicted_label = 2  # Default to TEXT
            
            # Store the labels
            true_labels.append(true_label)
            predicted_labels.append(predicted_label)
        
        # Make sure we have some labels to compute metrics
        if len(true_labels) == 0 or len(predicted_labels) == 0:
            logger.warning("No valid true/predicted labels found!")
            return {
                "overall_f1": 0.0,
                "form_f1": 0.0,
                "table_f1": 0.0,
                "text_f1": 0.0,
                "overall_precision": 0.0,
                "overall_recall": 0.0
            }
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predicted_labels, average=None, labels=[0, 1, 2], zero_division=0
        )
        
        # Overall metrics
        overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
            true_labels, predicted_labels, average='weighted', zero_division=0
        )
        
        # Log some predictions for debugging
        for i in range(min(5, len(true_labels))):
            logger.info(f"Example {i}: True={self.id2label[true_labels[i]]}, Pred={self.id2label[predicted_labels[i]]}")
        
        return {
            "overall_f1": overall_f1,
            "form_f1": f1[0],
            "table_f1": f1[1],
            "text_f1": f1[2],
            "overall_precision": overall_precision,
            "overall_recall": overall_recall
        }

    def predict(self, test_dataset, ignore_keys=None, metric_key_prefix="test"):
        """
        Run prediction and return predictions and potential metrics.
        """
        # Set model to eval mode
        self.model.eval()
        
        # Process in smaller chunks to avoid OOM
        chunk_size = self.args.per_device_eval_batch_size
        all_predictions = []
        all_labels = []
        
        for i in range(0, len(test_dataset), chunk_size):
            chunk = test_dataset.select(range(i, min(i + chunk_size, len(test_dataset))))
            chunk_inputs = self.data_collator(chunk)
            
            # Move inputs to device
            chunk_inputs = self._prepare_inputs(chunk_inputs)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**chunk_inputs)
            
            # Get logits and labels
            logits = outputs.logits.cpu().numpy()
            labels = chunk_inputs["labels"].cpu().numpy()
            
            all_predictions.append(logits)
            all_labels.append(labels)
            
            # Clear GPU memory
            torch.cuda.empty_cache()
        
        # Concatenate chunks
        predictions = np.concatenate(all_predictions, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        
        return predictions, labels

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Run evaluation and return metrics.
        """
        # Temporarily disable all logging
        root_logger = logging.getLogger()
        original_level = root_logger.level
        root_logger.setLevel(logging.ERROR)
        
        # Run standard evaluation
        output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Restore logging level
        root_logger.setLevel(original_level)
        
        # Only log once at the end of evaluation
        logger.info(f"Step {self.state.global_step}: evaluation complete")
        
        # Safely log metrics if they exist
        if 'eval_loss' in output:
            logger.info(f"  Eval loss: {output['eval_loss']:.4f}")
        
        # Log F1 scores if they exist
        metrics_to_log = {
            'eval_overall_f1': 'Overall F1',
            'eval_table_f1': 'TABLE F1',
            'eval_form_f1': 'FORM F1',
            'eval_text_f1': 'TEXT F1'
        }
        
        for metric_key, display_name in metrics_to_log.items():
            if metric_key in output:
                logger.info(f"  {display_name}: {output[metric_key]:.4f}")
            else:
                logger.warning(f"  {display_name}: Not available")
        
        # Add prefix to metrics
        metrics = {f"{metric_key_prefix}_{k}": v for k, v in output.items()}
        
        return metrics

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
    parser.add_argument("--eval_batch_size", type=int, default=8,
                      help="Evaluation batch size (smaller to prevent OOM)")
    parser.add_argument("--max_eval_samples", type=int, default=500,
                      help="Maximum number of samples to use for evaluation")
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
        save_total_limit=args.save_total_limit,
        eval_batch_size=args.eval_batch_size,
        max_eval_samples=args.max_eval_samples
    )

if __name__ == "__main__":
    main() 
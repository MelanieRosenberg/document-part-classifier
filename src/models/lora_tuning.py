import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset, load_from_disk
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import optuna
import logging
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoRAHyperparameterTuner:
    def __init__(
        self,
        base_model_name: str = "microsoft/deberta-v3-base",
        train_data_dir: str = "data/train",
        val_data_dir: str = "data/val",
        output_dir: str = "lora_tuning",
        context_window: int = 3,
        max_length: int = 512,
        device: Optional[str] = None
    ):
        """
        Initialize LoRA hyperparameter tuner.
        
        Args:
            base_model_name: Name of the base model to use
            train_data_dir: Directory containing training data
            val_data_dir: Directory containing validation data
            output_dir: Directory to save tuning results
            context_window: Size of context window
            max_length: Maximum sequence length
            device: Device to run training on (default: auto-detect)
        """
        self.base_model_name = base_model_name
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.output_dir = Path(output_dir)
        self.context_window = context_window
        self.max_length = max_length
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.output_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Label mappings
        self.id2label = {0: "FORM", 1: "TABLE", 2: "TEXT"}
        self.label2id = {"FORM": 0, "TABLE": 1, "TEXT": 2}
        
        # Load data
        logger.info("Loading datasets...")
        self.train_dataset = self.load_data(self.train_data_dir)
        self.val_dataset = self.load_data(self.val_data_dir)
        
        # Load base model and tokenizer
        logger.info(f"Loading base model and tokenizer from {base_model_name}")
        self.base_model, self.tokenizer = self.get_model_and_tokenizer()
    
    def setup_logging(self) -> None:
        """Setup logging configuration."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.log_dir / f"tuning_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    
    def load_data(self, data_dir: str) -> Dataset:
        """
        Load dataset from directory.
        
        Args:
            data_dir: Directory containing the dataset
            
        Returns:
            Dataset object
        """
        try:
            dataset = load_from_disk(data_dir)
            logger.info(f"Loaded {len(dataset)} examples from {data_dir}")
            return dataset
        except Exception as e:
            logger.error(f"Error loading data from {data_dir}: {e}")
            raise
    
    def get_model_and_tokenizer(self) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
        """
        Initialize base model and tokenizer.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            base_model = AutoModelForSequenceClassification.from_pretrained(
                self.base_model_name,
                num_labels=len(self.id2label),
                id2label=self.id2label,
                label2id=self.label2id
            )
            return base_model, tokenizer
        except Exception as e:
            logger.error(f"Error initializing model and tokenizer: {e}")
            raise
    
    def preprocess_function(self, examples: Dict[str, List[Any]], context_window: int) -> Dict[str, torch.Tensor]:
        """
        Preprocess examples with context window.
        
        Args:
            examples: Dictionary of examples
            context_window: Size of context window
            
        Returns:
            Dictionary of preprocessed examples
        """
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
        """
        Compute evaluation metrics.
        
        Args:
            eval_pred: Tuple of (predictions, labels)
            
        Returns:
            Dictionary of metrics
        """
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
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for hyperparameter optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            TABLE F1 score
        """
        # Define hyperparameters to tune
        params = {
            "lora_rank": trial.suggest_int("lora_rank", 4, 32),
            "lora_alpha": trial.suggest_int("lora_alpha", 8, 64),
            "lora_dropout": trial.suggest_float("lora_dropout", 0.0, 0.5),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
            "context_window": trial.suggest_int("context_window", 1, 5),
            "target_modules": trial.suggest_categorical("target_modules", [
                ["query", "key", "value"],
                ["query", "key", "value", "dense"],
                ["query", "value"]
            ])
        }
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=params["lora_rank"],
            lora_alpha=params["lora_alpha"],
            lora_dropout=params["lora_dropout"],
            target_modules=params["target_modules"],
            bias="none"
        )
        
        # Apply LoRA configuration
        model = get_peft_model(self.base_model, lora_config)
        
        # Preprocess data with current context window
        train_processed = self.train_dataset.map(
            lambda examples: self.preprocess_function(examples, params["context_window"]),
            batched=True
        )
        val_processed = self.val_dataset.map(
            lambda examples: self.preprocess_function(examples, params["context_window"]),
            batched=True
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / f"trial_{trial.number}"),
            learning_rate=params["learning_rate"],
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=params["weight_decay"],
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="table_f1",
            greater_is_better=True,
            remove_unused_columns=False
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_processed,
            eval_dataset=val_processed,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )
        
        # Train and evaluate
        try:
            trainer.train()
            eval_results = trainer.evaluate()
            
            # Log results
            params_str = (
                f"Trial {trial.number}: "
                f"rank={params['lora_rank']}, "
                f"alpha={params['lora_alpha']}, "
                f"dropout={params['lora_dropout']:.3f}, "
                f"lr={params['learning_rate']:.2e}, "
                f"wd={params['weight_decay']:.4f}, "
                f"context={params['context_window']}, "
                f"target={params['target_modules']}"
            )
            
            results_str = (
                f"Overall F1: {eval_results['eval_overall_f1']:.4f}, "
                f"TABLE F1: {eval_results['eval_table_f1']:.4f}, "
                f"FORM F1: {eval_results['eval_form_f1']:.4f}, "
                f"TEXT F1: {eval_results['eval_text_f1']:.4f}"
            )
            
            logger.info(f"{params_str} => {results_str}")
            
            return eval_results["eval_table_f1"]
            
        except Exception as e:
            logger.error(f"Error in trial {trial.number}: {e}")
            raise optuna.TrialPruned()
    
    def run_tuning(self, n_trials: int = 20) -> Dict[str, Any]:
        """
        Run hyperparameter tuning.
        
        Args:
            n_trials: Number of trials to run
            
        Returns:
            Dictionary of best parameters
        """
        logger.info(f"Starting LoRA hyperparameter tuning with {n_trials} trials")
        
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=n_trials)
        
        # Log best results
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best parameters: {study.best_params}")
        logger.info(f"Best TABLE F1 score: {study.best_value:.4f}")
        
        # Save best parameters
        best_params_file = self.output_dir / "best_params.csv"
        pd.DataFrame([study.best_params]).to_csv(best_params_file, index=False)
        logger.info(f"Saved best parameters to {best_params_file}")
        
        return study.best_params

def main():
    """Run hyperparameter tuning from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run LoRA hyperparameter tuning")
    parser.add_argument("--base_model", type=str, default="microsoft/deberta-v3-base",
                      help="Base model to use")
    parser.add_argument("--train_data_dir", type=str, required=True,
                      help="Directory containing training data")
    parser.add_argument("--val_data_dir", type=str, required=True,
                      help="Directory containing validation data")
    parser.add_argument("--output_dir", type=str, default="lora_tuning",
                      help="Directory to save tuning results")
    parser.add_argument("--n_trials", type=int, default=20,
                      help="Number of trials to run")
    parser.add_argument("--context_window", type=int, default=3,
                      help="Initial context window size")
    parser.add_argument("--max_length", type=int, default=512,
                      help="Maximum sequence length")
    
    args = parser.parse_args()
    
    # Initialize tuner
    tuner = LoRAHyperparameterTuner(
        base_model_name=args.base_model,
        train_data_dir=args.train_data_dir,
        val_data_dir=args.val_data_dir,
        output_dir=args.output_dir,
        context_window=args.context_window,
        max_length=args.max_length
    )
    
    # Run tuning
    best_params = tuner.run_tuning(n_trials=args.n_trials)
    print(f"\nBest parameters found:\n{best_params}")

if __name__ == "__main__":
    main() 
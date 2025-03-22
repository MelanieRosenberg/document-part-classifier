import argparse
import os
import json
import logging
import time
from typing import Dict, List, Tuple
from datetime import datetime

import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
from tqdm import tqdm

# Import our classifier (now without CRF)
from models.transformer_classifier import DocumentPartClassifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Target tags we need to achieve 90% F1 score on
TARGET_TAGS = ['TEXT', 'TABLE', 'FORM']

def evaluate_model(
    model: DocumentPartClassifier,
    val_lines: List[str],
    val_tags: List[str],
    device: torch.device,
) -> Tuple[float, Dict]:
    """
    Evaluate model on validation dataset.
    
    Args:
        model: The model to evaluate
        val_lines: List of validation text lines
        val_tags: List of validation tags
        device: Device to run evaluation on
        
    Returns:
        Tuple of (average loss, metrics dict)
    """
    # Prepare data
    encodings, labels = model.prepare_data(val_lines, val_tags)
    val_loader = model.create_data_loader(encodings, labels, shuffle=False)
    
    # Evaluate with the standard classifier
    val_f1, val_report = model.evaluate(val_loader, return_report=True)
    
    # Convert string report to dict if needed
    if isinstance(val_report, str):
        try:
            # Parse the report string into a dictionary
            lines = val_report.strip().split('\n')
            report_dict = {}
            
            # Parse the report string into a dictionary
            for line in lines[2:-3]:  # Skip header and footer lines
                parts = line.strip().split()
                if len(parts) >= 5:
                    tag = parts[0]
                    precision = float(parts[1])
                    recall = float(parts[2])
                    f1 = float(parts[3])
                    
                    report_dict[tag] = {
                        'precision': precision,
                        'recall': recall,
                        'f1-score': f1
                    }
            val_report = report_dict
        except Exception as e:
            logger.warning(f"Error parsing classification report: {e}")
            # Create a simple dict with the overall F1 score
            val_report = {tag: {'f1-score': val_f1} for tag in TARGET_TAGS}
    
    # Get loss
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            
            # For standard classification, forward pass returns loss when labels are provided
            loss = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / max(1, num_batches)
    
    return avg_loss, val_report

def compute_class_weights(
    labels: List[str],
    label_map: Dict[str, int],
    primary_weight_multiplier: float = 2.0
) -> torch.Tensor:
    """
    Compute class weights for weighted loss.
    
    Args:
        labels: List of label strings
        label_map: Mapping from label strings to indices
        primary_weight_multiplier: Multiplier for weights of primary tags
        
    Returns:
        Tensor of class weights
    """
    label_ids = [label_map[label] for label in labels]
    class_counts = np.bincount([label_map[l] for l in labels])
    
    # Ensure we have counts for all classes
    if len(class_counts) < len(label_map):
        temp_counts = np.zeros(len(label_map), dtype=np.int64)
        temp_counts[:len(class_counts)] = class_counts
        class_counts = temp_counts
    
    # Compute inverse frequency weights
    total_samples = len(labels)
    weights = torch.tensor([total_samples / max(1, count) for count in class_counts], 
                          dtype=torch.float)
    
    # Normalize weights
    weights = weights / weights.sum() * len(weights)
    
    # Apply multiplier for target tags
    for tag in TARGET_TAGS:
        if tag in label_map:
            tag_idx = label_map[tag]
            weights[tag_idx] *= primary_weight_multiplier
    
    return weights

def load_data_from_files(lines_file: str, tags_file: str) -> Tuple[List[str], List[str]]:
    """
    Load lines and tags from files.
    
    Args:
        lines_file: Path to file with text lines
        tags_file: Path to file with tags
        
    Returns:
        Tuple of (lines, tags)
    """
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
    
    return lines, tags

def create_training_plots(training_metrics: Dict[str, List[float]], output_dir: str) -> None:
    """
    Create and save training plots.
    
    Args:
        training_metrics: Dictionary containing training metrics
        output_dir: Directory to save the plots
    """
    try:
        import matplotlib.pyplot as plt
        
        # Plot loss and F1 curves
        plt.figure(figsize=(10, 6))
        plt.plot(training_metrics['epochs'], training_metrics['train_loss'], label='Train Loss')
        plt.plot(training_metrics['epochs'], training_metrics['val_f1'], label='Validation F1')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Training Loss and Validation F1')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "training_curves.png"))
        plt.close()
        
        logger.info(f"Training plots saved to {output_dir}")
    except ImportError:
        logger.warning("matplotlib not installed, skipping plot creation")

def train(args):
    """Train the model with the given arguments."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create log file
    log_file = os.path.join(output_dir, "training_log.txt")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Save training arguments
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Initialize classifier
    classifier = DocumentPartClassifier(
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        context_window=args.context_window
    )
    
    # Training metrics
    training_metrics = {
        'epochs': [],
        'train_loss': [],
        'val_f1': []
    }
    
    # Train the model
    history = classifier.train(
        train_lines_file=os.path.join(args.train_data_dir, 'lines.txt'),
        train_tags_file=os.path.join(args.train_data_dir, 'tags.txt'),
        val_lines_file=os.path.join(args.val_data_dir, 'lines.txt'),
        val_tags_file=os.path.join(args.val_data_dir, 'tags.txt'),
        model_save_dir=output_dir,
        early_stopping_patience=args.patience
    )
    
    # Record metrics - ensure consistent lengths
    num_epochs = len(history['train_loss'])
    training_metrics['epochs'] = list(range(1, num_epochs + 1))
    training_metrics['train_loss'] = history['train_loss']
    training_metrics['val_f1'] = history['val_f1']
    
    # Save training metrics
    metrics_path = os.path.join(output_dir, 'training_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(training_metrics, f, indent=2)
    
    # Create plots if requested
    if args.create_plots:
        create_training_plots(training_metrics, output_dir)
    
    # Check if target F1 scores were achieved
    target_f1_scores = {
        'FORM': 0.85,
        'TABLE': 0.85,
        'TEXT': 0.85
    }
    
    # Get final validation metrics
    val_loader = classifier.create_data_loader(
        classifier.prepare_data(
            classifier.load_data(
                os.path.join(args.val_data_dir, 'lines.txt'),
                os.path.join(args.val_data_dir, 'tags.txt')
            )[0]
        )[0],
        shuffle=False
    )
    _, val_report = classifier.evaluate(val_loader, return_report=True)
    
    # Parse the classification report to get F1 scores
    report_lines = val_report.split('\n')
    f1_scores = {}
    for line in report_lines:
        if any(tag in line for tag in target_f1_scores.keys()):
            parts = line.split()
            if len(parts) >= 5:
                tag = parts[0]
                f1 = float(parts[4])
                f1_scores[tag] = f1
    
    # Check if all target scores were achieved
    all_targets_achieved = all(
        f1_scores.get(tag, 0) >= target
        for tag, target in target_f1_scores.items()
    )
    
    if not all_targets_achieved:
        logger.info("\nTraining completed without achieving target F1 scores")
        logger.info("Final F1 scores:")
        for tag, target in target_f1_scores.items():
            current = f1_scores.get(tag, 0)
            logger.info(f"{tag}: {current:.4f} (target: {target:.4f})")
    else:
        logger.info("\nTraining completed successfully with all target F1 scores achieved!")
        logger.info("Final F1 scores:")
        for tag, score in f1_scores.items():
            logger.info(f"{tag}: {score:.4f}")
    
    return all_targets_achieved

def main():
    parser = argparse.ArgumentParser(description="Train document part classifier")
    
    # Data arguments
    parser.add_argument("--train_data_dir", type=str, required=True, help="Directory containing training data")
    parser.add_argument("--val_data_dir", type=str, required=True, help="Directory containing validation data")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save model and results")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-large", help="Name of pretrained model")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--context_window", type=int, default=2, help="Number of lines before/after to use as context")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Ratio of warmup steps")
    parser.add_argument("--patience", type=int, default=3, help="Number of epochs to wait before early stopping")
    
    # Class weight arguments
    parser.add_argument("--primary_weight_multiplier", type=float, default=2.0, help="Multiplier for primary tag weights")
    
    # System arguments
    parser.add_argument("--logging_steps", type=int, default=100, help="Number of steps between logging")
    parser.add_argument("--create_plots", action="store_true", help="Create and save training plots")
    
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
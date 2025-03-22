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

# Import our improved classifier
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
    
    # Evaluate
    val_f1, val_report = model.evaluate(val_loader, return_report=True)
    
    # Convert string report to dict if needed
    if isinstance(val_report, str):
        lines = val_report.strip().split('\n')
        report_dict = {}
        
        # Parse the report string into a dictionary
        # This is a simple parser and might need adjustment based on the exact format
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
    
    # Get loss
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            
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

def train(args):
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
    
    # Load training and validation data
    logger.info("Loading datasets...")
    
    train_lines_file = os.path.join(args.train_data_dir, "lines.txt")
    train_tags_file = os.path.join(args.train_data_dir, "tags.txt")
    val_lines_file = os.path.join(args.val_data_dir, "lines.txt")
    val_tags_file = os.path.join(args.val_data_dir, "tags.txt")
    
    # Validate files exist
    for file_path in [train_lines_file, train_tags_file, val_lines_file, val_tags_file]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")
    
    train_lines, train_tags = load_data_from_files(train_lines_file, train_tags_file)
    val_lines, val_tags = load_data_from_files(val_lines_file, val_tags_file)
    
    logger.info(f"Number of training examples: {len(train_lines)}")
    logger.info(f"Number of validation examples: {len(val_lines)}")
    
    # Initialize classifier
    classifier = DocumentPartClassifier(
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        context_window=args.context_window,
        warmup_ratio=args.warmup_ratio
    )
    
    # Update label map if needed
    unique_tags = set(train_tags + val_tags)
    if not all(tag in classifier.label_map for tag in unique_tags):
        logger.warning("Updating label map for custom tags...")
        custom_label_map = {tag: i for i, tag in enumerate(sorted(unique_tags))}
        classifier.label_map = custom_label_map
        classifier.reverse_label_map = {v: k for k, v in custom_label_map.items()}
        
        # Recreate model with the right number of labels
        classifier.model = type(classifier.model)(num_labels=len(custom_label_map))
        classifier.model.to(device)
    
    # Compute class weights
    class_weights = compute_class_weights(
        train_tags,
        classifier.label_map,
        primary_weight_multiplier=args.primary_weight_multiplier
    ).to(device)
    
    logger.info("\nClass weights:")
    for tag, weight in zip(classifier.label_map.keys(), class_weights.cpu().numpy()):
        logger.info(f"{tag}: {weight:.4f}")
        if tag in TARGET_TAGS:
            logger.info(f"  ^ Primary tag, weight multiplied by {args.primary_weight_multiplier}")
    
    # Prepare data
    train_encodings, train_labels = classifier.prepare_data(train_lines, train_tags)
    
    # Create dataloader
    train_loader = classifier.create_data_loader(train_encodings, train_labels, shuffle=True)
    
    # Initialize optimizer with weight decay
    optimizer = torch.optim.AdamW(
        classifier.model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Calculate training steps
    num_training_steps = len(train_loader) * args.num_epochs
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    
    # Create scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    logger.info(f"Total training steps: {num_training_steps}")
    logger.info(f"Warmup steps: {num_warmup_steps}")
    
    # Training loop
    best_val_loss = float('inf')
    best_metrics = None
    patience_counter = 0
    target_achieved = False
    
    # Track metrics over time
    training_metrics = {
        'epochs': [],
        'train_loss': [],
        'val_loss': [],
        'target_f1_scores': []
    }
    
    for epoch in range(args.num_epochs):
        # Training phase
        classifier.model.train()
        total_train_loss = 0
        train_steps = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            
            # Forward pass
            loss = classifier.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(classifier.model.parameters(), args.max_grad_norm)
            
            # Optimizer step
            optimizer.step()
            scheduler.step()
            
            # Update progress bar
            total_train_loss += loss.item()
            train_steps += 1
            progress_bar.set_postfix({
                'loss': f"{total_train_loss/train_steps:.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Log metrics periodically
            if train_steps % args.logging_steps == 0:
                logger.info(f"Step {train_steps}/{len(train_loader)}: "
                           f"Train Loss: {total_train_loss/train_steps:.4f}, "
                           f"LR: {scheduler.get_last_lr()[0]:.2e}")
        
        avg_train_loss = total_train_loss / train_steps
        logger.info(f"Epoch {epoch+1}/{args.num_epochs} - Train Loss: {avg_train_loss:.4f}")
        
        # Validation phase
        val_loss, val_metrics = evaluate_model(classifier, val_lines, val_tags, device)
        logger.info(f"Epoch {epoch+1}/{args.num_epochs} - Validation Loss: {val_loss:.4f}")
        
        # Update metrics
        training_metrics['epochs'].append(epoch + 1)
        training_metrics['train_loss'].append(avg_train_loss)
        training_metrics['val_loss'].append(val_loss)
        
        # Track target tag F1 scores
        target_f1 = {tag: val_metrics.get(tag, {}).get('f1-score', 0.0) for tag in TARGET_TAGS}
        training_metrics['target_f1_scores'].append(target_f1)
        
        # Log per-class metrics
        for tag in classifier.label_map.keys():
            tag_metrics = val_metrics.get(tag, {})
            precision = tag_metrics.get('precision', 0.0)
            recall = tag_metrics.get('recall', 0.0)
            f1 = tag_metrics.get('f1-score', 0.0)
            
            logger.info(f"{tag}: "
                       f"Precision: {precision:.4f}, "
                       f"Recall: {recall:.4f}, "
                       f"F1: {f1:.4f}")
        
        # Check if we've achieved target F1 scores
        target_f1_scores = {tag: val_metrics.get(tag, {}).get('f1-score', 0.0) for tag in TARGET_TAGS}
        target_achieved = all(score >= 0.9 for score in target_f1_scores.values())
        
        if target_achieved:
            logger.info("\nTarget F1 scores achieved! Saving model...")
            model_save_path = os.path.join(output_dir, "model.pt")
            
            # Save complete model state
            torch.save({
                'model_state_dict': classifier.model.state_dict(),
                'tokenizer_name': classifier.model_name,
                'label_map': classifier.label_map,
                'reverse_label_map': classifier.reverse_label_map,
                'context_window': classifier.context_window,
                'max_length': classifier.max_length
            }, model_save_path)
            
            with open(os.path.join(output_dir, "val_metrics.json"), "w") as f:
                json.dump(val_metrics, f, indent=2)
            break
        
        # Early stopping based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = val_metrics
            patience_counter = 0
            logger.info(f"New best validation loss: {best_val_loss:.4f}! Saving model...")
            
            model_save_path = os.path.join(output_dir, "model.pt")
            
            # Save complete model state
            torch.save({
                'model_state_dict': classifier.model.state_dict(),
                'tokenizer_name': classifier.model_name,
                'label_map': classifier.label_map,
                'reverse_label_map': classifier.reverse_label_map,
                'context_window': classifier.context_window,
                'max_length': classifier.max_length
            }, model_save_path)
            
            with open(os.path.join(output_dir, "val_metrics.json"), "w") as f:
                json.dump(val_metrics, f, indent=2)
        else:
            patience_counter += 1
            logger.info(f"Validation loss did not improve. Patience: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                logger.info(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
    
    # Save final model if target not achieved
    if not target_achieved:
        logger.info("\nTraining completed without achieving target F1 scores")
        model_save_path = os.path.join(output_dir, "model.pt")
        
        # Save complete model state
        torch.save({
            'model_state_dict': classifier.model.state_dict(),
            'tokenizer_name': classifier.model_name,
            'label_map': classifier.label_map,
            'reverse_label_map': classifier.reverse_label_map,
            'context_window': classifier.context_window,
            'max_length': classifier.max_length
        }, model_save_path)
        
        with open(os.path.join(output_dir, "val_metrics.json"), "w") as f:
            json.dump(val_metrics, f, indent=2)
    
    # Save training metrics
    with open(os.path.join(output_dir, "training_metrics.json"), "w") as f:
        json.dump(training_metrics, f, indent=2)
    
    # Create simple plots
    if args.create_plots:
        try:
            import matplotlib.pyplot as plt
            
            # Plot loss curves
            plt.figure(figsize=(10, 6))
            plt.plot(training_metrics['epochs'], training_metrics['train_loss'], label='Train Loss')
            plt.plot(training_metrics['epochs'], training_metrics['val_loss'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, "loss_curves.png"))
            
            # Plot F1 scores for target tags
            plt.figure(figsize=(10, 6))
            for tag in TARGET_TAGS:
                f1_scores = [epoch_f1[tag] for epoch_f1 in training_metrics['target_f1_scores']]
                plt.plot(training_metrics['epochs'], f1_scores, label=f'{tag} F1')
            plt.axhline(y=0.9, color='r', linestyle='--', label='Target F1 (0.9)')
            plt.xlabel('Epoch')
            plt.ylabel('F1 Score')
            plt.title('F1 Scores for Target Tags')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, "f1_scores.png"))
            
            logger.info(f"Training plots saved to {output_dir}")
        except ImportError:
            logger.warning("matplotlib not installed, skipping plot creation")

def main():
    parser = argparse.ArgumentParser(description="Train document part classifier")
    
    # Data arguments
    parser.add_argument("--train_data_dir", type=str, required=True, help="Directory containing training data")
    parser.add_argument("--val_data_dir", type=str, required=True, help="Directory containing validation data")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save model and results")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="roberta-large", help="Name of pretrained model")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--context_window", type=int, default=2, help="Number of lines before/after to use as context")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Ratio of warmup steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--patience", type=int, default=3, help="Number of epochs to wait before early stopping")
    
    # Class weight arguments
    parser.add_argument("--primary_weight_multiplier", type=float, default=2.0, help="Multiplier for primary tag weights")
    
    # System arguments
    parser.add_argument("--logging_steps", type=int, default=100, help="Number of steps between logging")
    parser.add_argument("--create_plots", action="store_true", help="Create and save training plots")
    
    args = parser.parse_args()
    
    # Load training and validation data
    train_lines_file = os.path.join(args.train_data_dir, "lines.txt")
    train_tags_file = os.path.join(args.train_data_dir, "tags.txt")
    val_lines_file = os.path.join(args.val_data_dir, "lines.txt")
    val_tags_file = os.path.join(args.val_data_dir, "tags.txt")
    
    # Validate files exist
    for file_path in [train_lines_file, train_tags_file, val_lines_file, val_tags_file]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")
    
    train(args)

if __name__ == "__main__":
    main()
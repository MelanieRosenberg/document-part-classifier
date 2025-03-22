import argparse
import os
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from src.data.dataset import DocumentPartDataset
from src.models.enhanced_model import EnhancedDocumentClassifier
import logging
from tqdm import tqdm
import psutil
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import classification_report
import json
from datetime import datetime
import wandb
from torch.cuda.amp import autocast, GradScaler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Target tags we need to achieve 90% F1 score on
TARGET_TAGS = ['TEXT', 'TABLE', 'FORM']

def evaluate_model(
    model: EnhancedDocumentClassifier,
    dataloader: DataLoader,
    device: torch.device,
    label_names: List[str]
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader for evaluation data
        device: Device to run evaluation on
        label_names: List of label names
        
    Returns:
        Tuple of (average loss, metrics dict)
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            # Forward pass
            outputs = model(**batch)
            
            # Accumulate loss and predictions
            total_loss += outputs['loss'].item()
            predictions = outputs['predictions'].cpu().numpy()
            labels = batch['labels'].cpu().numpy()
            
            # Apply smoothing to predictions
            smoothed_predictions = model.smooth_predictions(
                batch['input_texts'],
                [label_names[p] for p in predictions]
            )
            smoothed_predictions = [label_names.index(p) for p in smoothed_predictions]
            
            all_predictions.extend(smoothed_predictions)
            all_labels.extend(labels)
    
    # Compute average loss
    avg_loss = total_loss / len(dataloader)
    
    # Compute metrics
    report = classification_report(
        all_labels,
        all_predictions,
        target_names=label_names,
        output_dict=True
    )
    
    # Log metrics for target tags
    logger.info("\nMetrics for target tags:")
    target_f1_scores = {}
    for tag in TARGET_TAGS:
        tag_metrics = report[tag]
        target_f1_scores[tag] = tag_metrics['f1-score']
        logger.info(f"{tag}:")
        logger.info(f"  Precision: {tag_metrics['precision']:.4f}")
        logger.info(f"  Recall: {tag_metrics['recall']:.4f}")
        logger.info(f"  F1: {tag_metrics['f1-score']:.4f}")
    
    # Check if we've achieved target F1 scores
    target_achieved = all(score >= 0.9 for score in target_f1_scores.values())
    if target_achieved:
        logger.info("\nTarget F1 scores achieved for all target tags!")
    
    return avg_loss, report

def train(args):
    # Initialize wandb
    wandb.init(
        project="document-part-classifier",
        config=vars(args),
        name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save training arguments
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Create datasets
    logger.info("Loading datasets...")
    train_dataset, val_dataset = DocumentPartDataset.create_train_val_datasets(
        train_dir=os.path.join(args.data_dir, "train"),
        val_dir=os.path.join(args.data_dir, "val"),
        tokenizer_name=args.model_name,
        max_length=args.max_length,
        context_window=args.context_window
    )
    
    logger.info(f"Number of training examples: {len(train_dataset)}")
    logger.info(f"Number of validation examples: {len(val_dataset)}")
    logger.info(f"Number of labels: {train_dataset.num_labels}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Get label names and compute class weights
    label_names = list(train_dataset.label_map.keys())
    train_labels = [train_dataset.label_map[tag] for _, tag in train_dataset.examples]
    class_weights = EnhancedDocumentClassifier.compute_class_weights(
        labels=np.array(train_labels),
        label_names=label_names,
        primary_weight_multiplier=args.primary_weight_multiplier
    ).to(device)
    
    logger.info("\nClass weights:")
    for tag, weight in zip(label_names, class_weights.cpu().numpy()):
        logger.info(f"{tag}: {weight:.4f}")
        if tag in TARGET_TAGS:
            logger.info(f"  ^ Primary tag, weight multiplied by {args.primary_weight_multiplier}")
    
    # Initialize model
    model = EnhancedDocumentClassifier(
        model_name=args.model_name,
        num_labels=train_dataset.num_labels,
        class_weights=class_weights,
        label_names=label_names,
        use_lora=True,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        context_window=args.context_window
    ).to(device)
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    num_training_steps = len(train_loader) * args.num_epochs
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()
    
    logger.info(f"Total training steps: {num_training_steps}")
    logger.info(f"Warmup steps: {num_warmup_steps}")
    
    # Training loop
    best_val_loss = float('inf')
    best_metrics = None
    patience = args.patience
    patience_counter = 0
    target_achieved = False
    
    for epoch in range(args.num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        train_steps = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            # Forward pass with mixed precision
            with autocast():
                outputs = model(**batch)
                loss = outputs['loss']
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            # Optimizer step with gradient scaling
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            
            # Update progress bar
            total_train_loss += loss.item()
            train_steps += 1
            progress_bar.set_postfix({
                'loss': f"{total_train_loss/train_steps:.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Log metrics to wandb
            if train_steps % args.logging_steps == 0:
                wandb.log({
                    'train/loss': total_train_loss/train_steps,
                    'train/learning_rate': scheduler.get_last_lr()[0],
                    'train/epoch': epoch
                })
        
        avg_train_loss = total_train_loss / train_steps
        
        # Validation phase
        val_loss, val_metrics = evaluate_model(model, val_loader, device, label_names)
        
        # Log validation metrics to wandb
        wandb.log({
            'val/loss': val_loss,
            'val/epoch': epoch
        })
        
        # Log per-class metrics
        for tag in label_names:
            wandb.log({
                f'val/{tag}/precision': val_metrics[tag]['precision'],
                f'val/{tag}/recall': val_metrics[tag]['recall'],
                f'val/{tag}/f1': val_metrics[tag]['f1-score']
            })
        
        # Check if we've achieved target F1 scores
        target_f1_scores = {tag: val_metrics[tag]['f1-score'] for tag in TARGET_TAGS}
        target_achieved = all(score >= 0.9 for score in target_f1_scores.values())
        
        if target_achieved:
            logger.info("\nTarget F1 scores achieved! Saving model...")
            model.save_pretrained(output_dir)
            with open(os.path.join(output_dir, "val_metrics.json"), "w") as f:
                json.dump(val_metrics, f, indent=2)
            break
        
        # Early stopping based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = val_metrics
            patience_counter = 0
            model.save_pretrained(output_dir)
            with open(os.path.join(output_dir, "val_metrics.json"), "w") as f:
                json.dump(val_metrics, f, indent=2)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
    
    # Save final model if target not achieved
    if not target_achieved:
        logger.info("\nTraining completed without achieving target F1 scores")
        model.save_pretrained(output_dir)
        with open(os.path.join(output_dir, "val_metrics.json"), "w") as f:
            json.dump(val_metrics, f, indent=2)
    
    # Close wandb run
    wandb.finish()

def main():
    parser = argparse.ArgumentParser(description="Train enhanced document part classifier")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing train/val/test data")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save model and results")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base", help="Name of pretrained model")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--context_window", type=int, default=2, help="Number of lines before/after to use as context")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Ratio of warmup steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--patience", type=int, default=3, help="Number of epochs to wait before early stopping")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=16, help="Rank dimension for LoRA")
    parser.add_argument("--lora_alpha", type=int, default=32, help="Alpha parameter for LoRA scaling")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="Dropout probability for LoRA layers")
    
    # Class weight arguments
    parser.add_argument("--primary_weight_multiplier", type=float, default=2.0, help="Multiplier for primary tag weights")
    
    # System arguments
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--logging_steps", type=int, default=100, help="Number of steps between logging")
    
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main() 
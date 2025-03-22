import argparse
import os
import logging
from datetime import datetime
import json
from src.models.transformer_classifier import DocumentPartClassifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Train document part classifier')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing train and val data')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save model and logs')
    parser.add_argument('--model_name', type=str, default='roberta-base', help='Name of the base model to use')
    parser.add_argument('--max_length', type=int, default=256, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--context_window', type=int, default=2, help='Number of lines to include before and after')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Ratio of training steps to use for warmup')
    parser.add_argument('--early_stopping_patience', type=int, default=3, help='Number of epochs to wait for improvement')
    
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
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
        context_window=args.context_window,
        warmup_ratio=args.warmup_ratio
    )
    
    # Train the model
    history = classifier.train(
        train_lines_file=os.path.join(args.data_dir, "train", "lines.txt"),
        train_tags_file=os.path.join(args.data_dir, "train", "tags.txt"),
        val_lines_file=os.path.join(args.data_dir, "val", "lines.txt"),
        val_tags_file=os.path.join(args.data_dir, "val", "tags.txt"),
        model_save_dir=output_dir,
        early_stopping_patience=args.early_stopping_patience
    )
    
    # Save training history
    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"Training completed. Model and logs saved to {output_dir}")

if __name__ == "__main__":
    main() 
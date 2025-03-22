import argparse
import os
import logging
from datetime import datetime
import json
from typing import List, Dict, Tuple
import numpy as np
from src.models.tf_model import BiLSTMCRFClassifier
from src.data.dataset import DocumentPartDataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Target tags that need 90% F1 score
TARGET_TAGS = ['TEXT', 'TABLE', 'FORM']

def load_documents(data_dir: str) -> Tuple[List[str], List[List[str]]]:
    """
    Load documents and their labels from the data directory.
    
    Args:
        data_dir: Directory containing train/val/test data
        
    Returns:
        Tuple of (documents, labels)
    """
    documents = []
    labels = []
    
    # Load lines and tags
    with open(os.path.join(data_dir, "lines.txt"), 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f]
    
    with open(os.path.join(data_dir, "tags.txt"), 'r', encoding='utf-8') as f:
        tags = [line.strip() for line in f]
    
    # Group lines and tags by document
    current_doc = []
    current_labels = []
    
    for line, tag in zip(lines, tags):
        if line == "":  # Document separator
            if current_doc:
                documents.append("\n".join(current_doc))
                labels.append(current_labels)
                current_doc = []
                current_labels = []
        else:
            current_doc.append(line)
            current_labels.append(tag)
    
    # Add final document if exists
    if current_doc:
        documents.append("\n".join(current_doc))
        labels.append(current_labels)
    
    return documents, labels

def evaluate_predictions(
    predictions: List[str],
    true_labels: List[List[str]],
    label_names: List[str]
) -> Dict:
    """
    Evaluate model predictions.
    
    Args:
        predictions: List of predicted XML documents
        true_labels: List of true label sequences
        label_names: List of label names
        
    Returns:
        Dictionary containing evaluation metrics
    """
    from sklearn.metrics import classification_report
    
    # Extract predicted tags
    pred_tags = []
    for pred in predictions:
        # Parse XML-like tags
        import re
        tags = re.findall(r'<([A-Z_]+)>(.*?)</\1>', pred, re.DOTALL)
        doc_tags = []
        for tag, content in tags:
            content_lines = content.split('\n')
            doc_tags.extend([tag] * len(content_lines))
        pred_tags.append(doc_tags)
    
    # Flatten predictions and labels
    flat_preds = [tag for doc_tags in pred_tags for tag in doc_tags]
    flat_labels = [tag for doc_tags in true_labels for tag in doc_tags]
    
    # Compute metrics
    report = classification_report(
        flat_labels,
        flat_preds,
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
    
    return report

def train(args):
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save training arguments
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Load datasets
    logger.info("Loading datasets...")
    train_docs, train_labels = load_documents(os.path.join(args.data_dir, "train"))
    val_docs, val_labels = load_documents(os.path.join(args.data_dir, "val"))
    
    logger.info(f"Number of training documents: {len(train_docs)}")
    logger.info(f"Number of validation documents: {len(val_docs)}")
    
    # Get unique labels
    all_labels = [label for doc_labels in train_labels for label in doc_labels]
    label_names = sorted(set(all_labels))
    num_classes = len(label_names)
    
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Classes: {label_names}")
    
    # Initialize model
    model = BiLSTMCRFClassifier(
        input_dim=19,  # Number of features in DocumentPreprocessor
        num_classes=num_classes,
        window_size=args.window_size,
        lstm_units=args.lstm_units,
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate
    )
    
    # Train model
    logger.info("\nStarting training...")
    history = model.train(
        train_docs=train_docs,
        train_labels=train_labels,
        val_docs=val_docs,
        val_labels=val_labels,
        epochs=args.num_epochs,
        batch_size=args.batch_size,
        early_stopping_patience=args.patience
    )
    
    # Save model
    logger.info("\nSaving model...")
    model.save(os.path.join(output_dir, "model"))
    
    # Save label names
    with open(os.path.join(output_dir, "label_names.json"), "w") as f:
        json.dump(label_names, f, indent=2)
    
    # Evaluate on validation set
    logger.info("\nEvaluating on validation set...")
    val_predictions = model.predict(val_docs)
    val_metrics = evaluate_predictions(val_predictions, val_labels, label_names)
    
    # Save validation metrics
    with open(os.path.join(output_dir, "val_metrics.json"), "w") as f:
        json.dump(val_metrics, f, indent=2)
    
    # Save training history
    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Train BiLSTM-CRF model for document part classification")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing train/val/test data")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save model and results")
    
    # Model arguments
    parser.add_argument("--window_size", type=int, default=10, help="Size of context window")
    parser.add_argument("--lstm_units", type=int, nargs=2, default=[128, 64], help="Number of units in LSTM layers")
    parser.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate for regularization")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--patience", type=int, default=3, help="Number of epochs to wait before early stopping")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main() 
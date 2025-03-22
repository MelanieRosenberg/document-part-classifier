import argparse
import os
import json
import logging
from typing import Dict, List, Tuple
from datetime import datetime

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from models.transformer_classifier import DocumentPartClassifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_evaluation_plots(metrics: Dict, true_labels: List[str], pred_labels: List[str], output_dir: str):
    """
    Create and save evaluation plots.
    
    Args:
        metrics: Dictionary containing evaluation metrics
        true_labels: List of true labels
        pred_labels: List of predicted labels
        output_dir: Directory to save plots
    """
    try:
        # Create confusion matrix plot
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(true_labels, pred_labels)
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d',
            cmap='Blues',
            xticklabels=sorted(set(true_labels)),
            yticklabels=sorted(set(true_labels))
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()
        
        # Create bar plot of F1 scores
        plt.figure(figsize=(10, 6))
        tags = []
        f1_scores = []
        for tag, tag_metrics in metrics.items():
            tags.append(tag)
            f1_scores.append(tag_metrics['f1-score'])
        
        plt.bar(tags, f1_scores)
        plt.title('F1 Scores by Tag')
        plt.xlabel('Tag')
        plt.ylabel('F1 Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'f1_scores.png'))
        plt.close()
        
        logger.info(f"Evaluation plots saved to {output_dir}")
    except Exception as e:
        logger.warning(f"Failed to create plots: {e}")

def evaluate_test_set(
    model_path: str,
    test_data_dir: str,
    output_dir: str,
    batch_size: int = 16,
    max_length: int = 512,
    context_window: int = 2
) -> Dict:
    """
    Evaluate the trained model on the test set.
    
    Args:
        model_path: Path to the trained model checkpoint
        test_data_dir: Directory containing test data
        output_dir: Directory to save evaluation results
        batch_size: Batch size for evaluation
        max_length: Maximum sequence length
        context_window: Number of lines before/after to use as context
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = os.path.join(output_dir, f"eval_{timestamp}")
    os.makedirs(eval_dir, exist_ok=True)
    
    # Create log file
    log_file = os.path.join(eval_dir, "evaluation_log.txt")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Initialize classifier with the same parameters as training
    classifier = DocumentPartClassifier(
        model_name="microsoft/deberta-v3-large",  # Same as training
        max_length=max_length,
        batch_size=batch_size,
        learning_rate=1e-5,  # Not used for evaluation
        num_epochs=1,  # Not used for evaluation
        context_window=context_window
    )
    
    # Load the trained model
    logger.info(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path)
    classifier.model.load_state_dict(checkpoint['model_state_dict'])
    classifier.model.eval()
    
    # Load test data
    test_lines_file = os.path.join(test_data_dir, 'lines.txt')
    test_tags_file = os.path.join(test_data_dir, 'tags.txt')
    
    logger.info("Loading test data...")
    test_lines, test_tags = classifier.load_data(test_lines_file, test_tags_file)
    
    # Prepare test data
    logger.info("Preparing test data...")
    encodings, labels = classifier.prepare_data(test_lines, test_tags)
    test_loader = classifier.create_data_loader(encodings, labels, shuffle=False)
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch[0].to(classifier.device)
            attention_mask = batch[1].to(classifier.device)
            labels = batch[2].to(classifier.device)
            
            # Get predictions
            logits = classifier.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())
    
    # Convert numeric labels to strings
    pred_labels = [classifier.reverse_label_map[pred] for pred in all_preds]
    true_labels = [classifier.reverse_label_map[label] for label in all_labels]
    
    # Calculate metrics
    test_report = classification_report(true_labels, pred_labels)
    
    # Parse the classification report
    report_lines = test_report.strip().split('\n')
    metrics = {}
    
    for line in report_lines[2:-3]:  # Skip header and footer lines
        parts = line.strip().split()
        if len(parts) >= 5:
            tag = parts[0]
            precision = float(parts[1])
            recall = float(parts[2])
            f1 = float(parts[3])
            
            metrics[tag] = {
                'precision': precision,
                'recall': recall,
                'f1-score': f1
            }
    
    # Save evaluation metrics
    metrics_path = os.path.join(eval_dir, 'test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Create evaluation plots
    create_evaluation_plots(metrics, true_labels, pred_labels, eval_dir)
    
    # Print evaluation results
    logger.info("\nTest Set Evaluation Results:")
    logger.info("-" * 40)
    logger.info(f"Overall F1 Score: {np.mean([m['f1-score'] for m in metrics.values()]):.4f}")
    logger.info("\nPer-Tag Metrics:")
    for tag, tag_metrics in metrics.items():
        logger.info(f"\n{tag}:")
        logger.info(f"  Precision: {tag_metrics['precision']:.4f}")
        logger.info(f"  Recall: {tag_metrics['recall']:.4f}")
        logger.info(f"  F1 Score: {tag_metrics['f1-score']:.4f}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate document part classifier on test set")
    
    # Required arguments
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--test_data_dir", type=str, required=True, help="Directory containing test data")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save evaluation results")
    
    # Optional arguments
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--context_window", type=int, default=2, help="Number of lines before/after to use as context")
    
    args = parser.parse_args()
    evaluate_test_set(
        model_path=args.model_path,
        test_data_dir=args.test_data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        context_window=args.context_window
    )

if __name__ == "__main__":
    main() 
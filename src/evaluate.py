import argparse
import os
import json
import logging
from typing import Dict, List, Tuple
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from tqdm import tqdm

from models.transformer_classifier import DocumentPartClassifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    test_f1, test_report = classifier.evaluate(test_loader, return_report=True)
    
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
    
    # Print evaluation results
    logger.info("\nTest Set Evaluation Results:")
    logger.info("-" * 40)
    logger.info(f"Overall F1 Score: {test_f1:.4f}")
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
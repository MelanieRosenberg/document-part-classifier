from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import torch
from seaborn import heatmap
import matplotlib.pyplot as plt
from ..data.preprocessing import DocumentPreprocessor

class MetricsCalculator:
    """Calculate and visualize metrics for document part classification."""
    
    def __init__(self, preprocessor: DocumentPreprocessor):
        self.preprocessor = preprocessor
        self.label_names = preprocessor.tag_types
    
    def calculate_metrics(
        self,
        true_labels: List[int],
        pred_labels: List[int],
        prefix: str = ""
    ) -> Dict[str, float]:
        """Calculate precision, recall, and F1 score."""
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels,
            pred_labels,
            labels=range(len(self.label_names)),
            average=None
        )
        
        # Per-class metrics
        class_metrics = {}
        for i, label in enumerate(self.label_names):
            class_metrics.update({
                f"{prefix}{label}_precision": precision[i],
                f"{prefix}{label}_recall": recall[i],
                f"{prefix}{label}_f1": f1[i],
                f"{prefix}{label}_support": support[i]
            })
        
        # Weighted averages
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            true_labels,
            pred_labels,
            average="weighted"
        )
        
        # Macro averages
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            true_labels,
            pred_labels,
            average="macro"
        )
        
        metrics = {
            f"{prefix}weighted_precision": weighted_precision,
            f"{prefix}weighted_recall": weighted_recall,
            f"{prefix}weighted_f1": weighted_f1,
            f"{prefix}macro_precision": macro_precision,
            f"{prefix}macro_recall": macro_recall,
            f"{prefix}macro_f1": macro_f1
        }
        
        metrics.update(class_metrics)
        return metrics
    
    def plot_confusion_matrix(
        self,
        true_labels: List[int],
        pred_labels: List[int],
        normalize: bool = True,
        save_path: Optional[str] = None
    ) -> None:
        """Plot confusion matrix as a heatmap."""
        cm = confusion_matrix(
            true_labels,
            pred_labels,
            labels=range(len(self.label_names))
        )
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(10, 8))
        heatmap(
            cm,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=self.label_names,
            yticklabels=self.label_names
        )
        plt.title('Normalized Confusion Matrix' if normalize else 'Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def print_classification_report(
        self,
        true_labels: List[int],
        pred_labels: List[int]
    ) -> None:
        """Print detailed classification report."""
        print("\nClassification Report:")
        print(classification_report(
            true_labels,
            pred_labels,
            labels=range(len(self.label_names)),
            target_names=self.label_names,
            digits=4
        ))
    
    def analyze_errors(
        self,
        true_labels: List[int],
        pred_labels: List[int],
        texts: List[str],
        n_examples: int = 5
    ) -> Dict[Tuple[str, str], List[str]]:
        """Analyze misclassification examples."""
        error_examples = {}
        
        for true_label, pred_label, text in zip(true_labels, pred_labels, texts):
            if true_label != pred_label:
                true_name = self.label_names[true_label]
                pred_name = self.label_names[pred_label]
                key = (true_name, pred_name)
                
                if key not in error_examples:
                    error_examples[key] = []
                
                if len(error_examples[key]) < n_examples:
                    error_examples[key].append(text)
        
        return error_examples
    
    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        save_path: Optional[str] = None
    ) -> None:
        """Plot training metrics over time."""
        plt.figure(figsize=(15, 5))
        
        # Loss plot
        plt.subplot(1, 3, 1)
        plt.plot(history['train_loss'], label='Train')
        if 'val_loss' in history:
            plt.plot(history['val_loss'], label='Validation')
        plt.title('Loss over time')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.legend()
        
        # F1 score plot
        plt.subplot(1, 3, 2)
        plt.plot(history['train_f1'], label='Train')
        if 'val_f1' in history:
            plt.plot(history['val_f1'], label='Validation')
        plt.title('F1 Score over time')
        plt.xlabel('Step')
        plt.ylabel('F1 Score')
        plt.legend()
        
        # Memory usage plot
        if 'memory_usage' in history:
            plt.subplot(1, 3, 3)
            plt.plot(history['memory_usage'])
            plt.title('Memory Usage over time')
            plt.xlabel('Step')
            plt.ylabel('Memory (MB)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def get_attention_analysis(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tokenizer
    ) -> Dict[str, np.ndarray]:
        """Analyze attention patterns in the model."""
        # Get attention weights
        with torch.no_grad():
            attention = model.get_attention_weights(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Convert to numpy
        attention = [layer.cpu().numpy() for layer in attention]
        
        # Average across heads and layers
        avg_attention = np.mean([layer.mean(axis=1) for layer in attention], axis=0)
        
        # Get tokens
        tokens = [tokenizer.decode([token_id]) for token_id in input_ids[0]]
        
        return {
            "attention_weights": avg_attention,
            "tokens": tokens
        } 
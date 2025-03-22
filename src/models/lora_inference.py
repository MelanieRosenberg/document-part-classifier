import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, PeftConfig
import logging
from typing import List, Dict, Union
import os

logger = logging.getLogger(__name__)

class LoRAInferenceModel:
    def __init__(
        self,
        model_path: str,
        context_window: int = 3,
        max_length: int = 512,
        device: str = None
    ):
        """
        Initialize LoRA inference model.
        
        Args:
            model_path: Path to the saved LoRA model
            context_window: Size of context window to use
            max_length: Maximum sequence length
            device: Device to run inference on (default: auto-detect)
        """
        self.model_path = model_path
        self.context_window = context_window
        self.max_length = max_length
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load configuration
        logger.info(f"Loading model configuration from {model_path}")
        self.config = PeftConfig.from_pretrained(model_path)
        
        # Setup label mappings
        self.id2label = {0: "FORM", 1: "TABLE", 2: "TEXT"}
        self.label2id = {"FORM": 0, "TABLE": 1, "TEXT": 2}
        
        # Load tokenizer and model
        logger.info(f"Loading tokenizer from {self.config.base_model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name_or_path)
        
        logger.info("Loading base model and LoRA weights")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.config.base_model_name_or_path,
            num_labels=len(self.id2label),
            id2label=self.id2label,
            label2id=self.label2id
        )
        
        # Load and merge LoRA weights
        self.model = PeftModel.from_pretrained(base_model, model_path)
        logger.info("Merging LoRA weights with base model for faster inference")
        self.model = self.model.merge_and_unload()
        
        # Move model to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded and ready for inference on {self.device}")
    
    def preprocess_text(self, texts: List[str]) -> List[str]:
        """
        Preprocess texts with context window.
        
        Args:
            texts: List of texts to process
            
        Returns:
            List of processed texts with context
        """
        processed_texts = []
        for i in range(len(texts)):
            # Get context window
            start_idx = max(0, i - self.context_window)
            end_idx = min(len(texts), i + self.context_window + 1)
            
            # Combine text with context
            context = texts[start_idx:end_idx]
            text = " [SEP] ".join(context)
            processed_texts.append(text)
        
        return processed_texts
    
    def predict(
        self,
        texts: List[str],
        return_confidence: bool = False,
        batch_size: int = 32
    ) -> Union[pd.DataFrame, Dict]:
        """
        Make predictions on new text samples.
        
        Args:
            texts: List of texts to classify
            return_confidence: Whether to return confidence scores
            batch_size: Batch size for inference
            
        Returns:
            DataFrame with predictions or dict with predictions and confidences
        """
        # Process texts with context window
        processed_texts = self.preprocess_text(texts)
        
        all_predictions = []
        all_confidences = []
        
        # Process in batches
        for i in range(0, len(processed_texts), batch_size):
            batch_texts = processed_texts[i:i + batch_size]
            
            # Tokenize inputs
            inputs = self.tokenizer(
                batch_texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Make predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
            
            # Convert predictions to labels
            batch_predictions = [self.id2label[pred.item()] for pred in predictions]
            all_predictions.extend(batch_predictions)
            
            if return_confidence:
                # Get confidence scores for all classes
                batch_confidences = probs.cpu().numpy()
                all_confidences.extend(batch_confidences)
        
        # Create results dataframe
        results = pd.DataFrame({
            "text": texts,
            "predicted_label": all_predictions
        })
        
        if return_confidence:
            # Add confidence scores for each class
            confidences = np.array(all_confidences)
            for i, label in self.id2label.items():
                results[f"{label.lower()}_confidence"] = confidences[:, i]
            
            # Find examples near decision boundaries
            for label in self.id2label.values():
                confidence_col = f"{label.lower()}_confidence"
                boundary_mask = (results[confidence_col] > 0.4) & (results[confidence_col] < 0.6)
                if boundary_mask.any():
                    logger.info(f"\nFound {boundary_mask.sum()} {label} examples near decision boundary:")
                    logger.info(results[boundary_mask][["text", "predicted_label", confidence_col]])
            
            return {
                "predictions": results,
                "confidences": confidences
            }
        
        return results

def main():
    """Example usage of the inference model."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run inference with LoRA model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved LoRA model")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input text file (one text per line)")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save predictions")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--context_window", type=int, default=3, help="Size of context window")
    parser.add_argument("--return_confidence", action="store_true", help="Return confidence scores")
    
    args = parser.parse_args()
    
    # Load texts
    with open(args.input_file, 'r') as f:
        texts = [line.strip() for line in f]
    
    # Initialize model
    model = LoRAInferenceModel(
        model_path=args.model_path,
        context_window=args.context_window
    )
    
    # Make predictions
    results = model.predict(
        texts,
        return_confidence=args.return_confidence,
        batch_size=args.batch_size
    )
    
    # Save results
    if isinstance(results, dict):
        results["predictions"].to_csv(args.output_file, index=False)
        np.save(args.output_file + "_confidences.npy", results["confidences"])
    else:
        results.to_csv(args.output_file, index=False)

if __name__ == "__main__":
    main() 
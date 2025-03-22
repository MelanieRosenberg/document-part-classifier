import argparse
import os
import torch
from torch.utils.data import DataLoader
from src.data.dataset import DocumentPartDataset
from src.models.enhanced_model import EnhancedDocumentClassifier
import logging
from tqdm import tqdm
import json
from typing import List, Dict
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def predict_document(
    model: EnhancedDocumentClassifier,
    lines: List[str],
    device: torch.device,
    label_names: List[str],
    batch_size: int = 32,
    context_window: int = 2
) -> List[str]:
    """
    Predict tags for a document.
    
    Args:
        model: The trained model
        lines: List of lines in the document
        device: Device to run inference on
        label_names: List of label names
        batch_size: Batch size for inference
        context_window: Number of lines before/after to use as context
        
    Returns:
        List of predicted tags
    """
    model.eval()
    
    # Create dataset
    dataset = DocumentPartDataset(
        data_dir=None,  # Not needed for inference
        tokenizer_name=model.config.name_or_path,
        max_length=model.config.max_position_embeddings,
        label_map={name: i for i, name in enumerate(label_names)},
        context_window=context_window
    )
    
    # Add document to dataset
    dataset.examples = [(line, label_names[0]) for line in lines]  # Dummy labels
    dataset.example_positions = [(0, i) for i in range(len(lines))]
    dataset.doc_lines = {0: lines}
    dataset.doc_tags = {0: [label_names[0]] * len(lines)}
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Run inference
    all_predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            # Forward pass
            outputs = model(**batch)
            predictions = outputs['predictions'].cpu().numpy()
            
            # Apply smoothing
            smoothed_predictions = model.smooth_predictions(
                batch['input_texts'],
                [label_names[p] for p in predictions]
            )
            
            all_predictions.extend(smoothed_predictions)
    
    return all_predictions

def process_file(
    input_file: str,
    output_file: str,
    model: EnhancedDocumentClassifier,
    device: torch.device,
    label_names: List[str],
    batch_size: int = 32,
    context_window: int = 2
):
    """
    Process a single file.
    
    Args:
        input_file: Path to input file
        output_file: Path to output file
        model: The trained model
        device: Device to run inference on
        label_names: List of label names
        batch_size: Batch size for inference
        context_window: Number of lines before/after to use as context
    """
    # Read input file
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # Skip empty files
    if not lines:
        logger.warning(f"Skipping empty file: {input_file}")
        return
    
    # Predict tags
    predictions = predict_document(
        model,
        lines,
        device,
        label_names,
        batch_size,
        context_window
    )
    
    # Reconstruct document with tags
    tagged_doc = model.reconstruct_document(lines, predictions)
    
    # Write output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(tagged_doc)
    
    logger.info(f"Processed {input_file} -> {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Predict document part tags")
    
    # Model arguments
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing trained model")
    parser.add_argument("--label_map", type=str, required=True, help="Path to label map JSON file")
    
    # Data arguments
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save tagged files")
    
    # Inference arguments
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--context_window", type=int, default=2, help="Number of lines before/after to use as context")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load label map
    with open(args.label_map, 'r') as f:
        label_map = json.load(f)
    label_names = list(label_map.keys())
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    model = EnhancedDocumentClassifier.from_pretrained(
        args.model_dir,
        num_labels=len(label_names),
        label_names=label_names
    ).to(device)
    
    # Process files
    for fname in os.listdir(args.input_dir):
        if not fname.endswith('.txt'):
            continue
            
        input_file = os.path.join(args.input_dir, fname)
        output_file = os.path.join(args.output_dir, fname)
        
        try:
            process_file(
                input_file,
                output_file,
                model,
                device,
                label_names,
                args.batch_size,
                args.context_window
            )
        except Exception as e:
            logger.error(f"Error processing {fname}: {str(e)}")
            continue

if __name__ == "__main__":
    main() 
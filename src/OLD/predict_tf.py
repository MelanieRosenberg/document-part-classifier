import argparse
import os
import logging
import json
from typing import List
from src.models.tf_model import BiLSTMCRFClassifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_file(
    input_file: str,
    output_file: str,
    model: BiLSTMCRFClassifier
):
    """
    Process a single file.
    
    Args:
        input_file: Path to input file
        output_file: Path to output file
        model: The trained model
    """
    # Read input file
    with open(input_file, 'r', encoding='utf-8') as f:
        doc = f.read()
    
    # Skip empty files
    if not doc.strip():
        logger.warning(f"Skipping empty file: {input_file}")
        return
    
    # Predict tags
    predictions = model.predict([doc])
    tagged_doc = predictions[0]
    
    # Write output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(tagged_doc)
    
    logger.info(f"Processed {input_file} -> {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Predict document part tags using BiLSTM-CRF model")
    
    # Model arguments
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing trained model")
    
    # Data arguments
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save tagged files")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    logger.info("Loading model...")
    model = BiLSTMCRFClassifier.load(args.model_dir)
    
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
                model
            )
        except Exception as e:
            logger.error(f"Error processing {fname}: {str(e)}")
            continue

if __name__ == "__main__":
    main() 
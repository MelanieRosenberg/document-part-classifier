import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import time
from typing import List, Dict
import logging
import xml.etree.ElementTree as ET
import os
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobertaClassifier:
    def __init__(
        self,
        model_path: str = "../../models/full_run/run_20250321_175632/model.pt",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        logger.info("Loading model and tokenizer...")
        
        # Load trained weights first to get label mapping
        logger.info(f"Loading trained weights from {model_path}")
        state_dict = torch.load(model_path, map_location=self.device)
        # Extract metadata
        self.tokenizer_name = state_dict['tokenizer_name']
        self.label_map = state_dict['label_map']
        self.reverse_label_map = state_dict['reverse_label_map']
        self.context_window = state_dict['context_window']
        self.max_length = state_dict['max_length']
        
        # Initialize model and tokenizer with the correct version
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "roberta-base",
            num_labels=len(self.label_map),  # Number of labels from label map
            id2label=self.reverse_label_map,  # Use the reverse label map
            label2id=self.label_map,  # Use the label map
            ignore_mismatched_sizes=True,  # Allow size mismatches
            vocab_size=50267  # Match the vocabulary size from the trained model
        )
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        
        # Load the model state dict
        self.model.load_state_dict(state_dict['model_state_dict'], strict=False)
        
        # Move model to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()
    
    def preprocess_text(self, texts: List[str]) -> List[str]:
        """Preprocess texts with context window."""
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
    
    def extract_text_from_xml(self, xml_content: str) -> List[str]:
        """Extract text content from XML or plain text."""
        try:
            # Try to parse as XML first
            root = ET.fromstring(xml_content)
            # Extract text from all elements
            texts = []
            for elem in root.iter():
                if elem.text and elem.text.strip():
                    texts.append(elem.text.strip())
            return texts
        except ET.ParseError:
            # If XML parsing fails, try to extract text content directly
            # Split into lines and filter out empty ones
            lines = []
            # Remove XML declaration if present
            content = xml_content.replace('<?xml version="1.0" encoding="UTF-8"?>', '')
            # Split by line breaks and remove XML tags
            for line in content.split('\n'):
                # Remove XML tags
                text = ''
                in_tag = False
                for char in line:
                    if char == '<':
                        in_tag = True
                    elif char == '>':
                        in_tag = False
                    elif not in_tag:
                        text += char
                # Clean and add non-empty lines
                text = text.strip()
                if text:
                    lines.append(text)
            return lines
    
    def predict(self, xml_file) -> Dict[str, List[str]]:
        """Predict document part types for XML file."""
        if xml_file is None:
            return {
                "predictions": ["Error: No file uploaded"],
                "inference_time": "0.00 seconds"
            }
        
        try:
            # Read XML content
            if hasattr(xml_file, 'read'):
                xml_content = xml_file.read().decode('utf-8')
            else:
                # If it's a file path, read the file
                with open(xml_file, 'r', encoding='utf-8') as f:
                    xml_content = f.read()
            
            # Extract text from XML
            lines = self.extract_text_from_xml(xml_content)
            if not lines:
                return {
                    "predictions": ["Error: No text content found in XML"],
                    "inference_time": "0.00 seconds"
                }
            
            # Process texts with context window
            processed_texts = self.preprocess_text(lines)
            
            # Tokenize inputs
            inputs = self.tokenizer(
                processed_texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Make predictions
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
            
            # Convert predictions to labels using the reverse label map
            pred_labels = [self.reverse_label_map[pred.item()] for pred in predictions]
            inference_time = time.time() - start_time
            
            # Format results
            results = []
            for line, label in zip(lines, pred_labels):
                results.append(f"{label}: {line}")
            
            return {
                "predictions": results,
                "inference_time": f"{inference_time:.2f} seconds"
            }
        except Exception as e:
            import traceback
            error_msg = f"Error processing file: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            return {
                "predictions": [error_msg],
                "inference_time": "0.00 seconds"
            }

def create_gradio_interface():
    """Create Gradio interface for document classification."""
    # Initialize classifier
    classifier = RobertaClassifier()
    
    def process_document(file):
        """Process document and return predictions."""
        results = classifier.predict(file)
        return "\n".join(results["predictions"]), results["inference_time"]
    
    # Create interface
    iface = gr.Interface(
        fn=process_document,
        inputs=gr.File(
            label="Upload XML Document",
            file_types=[".xml"]
        ),
        outputs=[
            gr.Textbox(
                label="Predictions",
                lines=10
            ),
            gr.Textbox(
                label="Inference Time"
            )
        ],
        title="RoBERTa Document Part Classifier",
        description="Upload an XML document to classify its parts into FORM, TABLE, or TEXT categories using RoBERTa model."
    )
    
    return iface

if __name__ == "__main__":
    # Create and launch interface
    iface = create_gradio_interface()
    iface.launch(share=True) 
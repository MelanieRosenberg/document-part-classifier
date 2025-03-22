import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import time
from typing import List, Dict
import logging
import xml.etree.ElementTree as ET

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
        
        # Initialize model and tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "roberta-base",
            num_labels=3  # FORM, TABLE, TEXT
        )
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        
        # Load trained weights
        logger.info(f"Loading trained weights from {model_path}")
        state_dict = torch.load(model_path, map_location=self.device)
        # Extract model state dict and metadata
        model_state_dict = state_dict['model_state_dict']
        self.tokenizer_name = state_dict['tokenizer_name']
        self.label_map = state_dict['label_map']
        self.reverse_label_map = state_dict['reverse_label_map']
        self.context_window = state_dict['context_window']
        self.max_length = state_dict['max_length']
        
        # Load the model state dict
        self.model.load_state_dict(model_state_dict)
        
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
        """Extract text content from XML without tags."""
        try:
            # Parse XML content
            root = ET.fromstring(xml_content)
            
            # Extract text from all elements
            texts = []
            for elem in root.iter():
                if elem.text and elem.text.strip():
                    texts.append(elem.text.strip())
            
            return texts
        except ET.ParseError as e:
            logger.error(f"Error parsing XML: {e}")
            return []
    
    def predict(self, xml_file) -> Dict[str, List[str]]:
        """Predict document part types for XML file."""
        if xml_file is None:
            return {
                "predictions": ["Error: No file uploaded"],
                "inference_time": "0.00 seconds"
            }
        
        # Read XML content
        xml_content = xml_file.decode('utf-8')
        
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
        
        # Convert predictions to labels
        pred_labels = [self.id2label[pred.item()] for pred in predictions]
        inference_time = time.time() - start_time
        
        # Format results
        results = []
        for line, label in zip(lines, pred_labels):
            results.append(f"{label}: {line}")
        
        return {
            "predictions": results,
            "inference_time": f"{inference_time:.2f} seconds"
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
        description="Upload an XML document to classify its parts into FORM, TABLE, or TEXT categories using RoBERTa model.",
        examples=[
            ["example.xml"]
        ]
    )
    
    return iface

if __name__ == "__main__":
    # Create and launch interface
    iface = create_gradio_interface()
    iface.launch(share=True) 
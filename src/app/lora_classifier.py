import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
import time
from typing import List, Dict
import logging
import xml.etree.ElementTree as ET

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LlamaClassifier:
    def __init__(
        self,
        base_model_path: str = "/home/azureuser/llama_models/llama-3-2-1b",
        lora_weights_path: str = "models/llama_1b_lora/run_20250322_202241/final_model",
        context_window: int = 3,
        max_length: int = 512,
        device: str = None
    ):
        """Initialize the Llama classifier."""
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.context_window = context_window
        self.max_length = max_length
        
        # Label mappings
        self.id2label = {0: "FORM", 1: "TABLE", 2: "TEXT"}
        self.label2id = {"FORM": 0, "TABLE": 1, "TEXT": 2}
        
        # Load model and tokenizer
        logger.info("Loading model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_path,
            num_labels=len(self.id2label),
            id2label=self.id2label,
            label2id=self.label2id,
            trust_remote_code=True
        )
        
        # Load and merge LoRA weights
        logger.info("Loading LoRA weights...")
        self.model = PeftModel.from_pretrained(
            self.base_model,
            lora_weights_path
        )
        logger.info("Merging LoRA weights with base model for faster inference")
        self.model = self.model.merge_and_unload()
        
        # Move model to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded and ready for inference on {self.device}")
    
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
        
        try:
            # Read XML content
            if hasattr(xml_file, 'read'):
                xml_content = xml_file.read().decode('utf-8')
            else:
                xml_content = xml_file
            
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
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return {
                "predictions": [f"Error during prediction: {str(e)}"],
                "inference_time": "0.00 seconds"
            }

def create_gradio_interface():
    """Create Gradio interface for document classification."""
    # Initialize classifier
    classifier = LlamaClassifier()
    
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
        title="Llama Document Part Classifier",
        description="Upload an XML document to classify its parts into FORM, TABLE, or TEXT categories using Llama model.",
        examples=[
            ["example.xml"]
        ]
    )
    
    return iface

if __name__ == "__main__":
    # Create and launch interface
    iface = create_gradio_interface()
    iface.launch(share=True) 
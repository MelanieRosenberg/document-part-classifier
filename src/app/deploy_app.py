import gradio as gr
import logging
import argparse
from typing import Literal

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def deploy_app(model_type: Literal["deberta", "roberta", "llama"], share: bool = True):
    """Deploy Gradio app for document classification.
    
    Args:
        model_type: Type of model to use ("deberta", "roberta", or "llama")
        share: Whether to share the app publicly
    """
    try:
        if model_type == "deberta":
            logger.info("Deploying DeBERTa classifier app...")
            from deberta_classifier import create_gradio_interface
        elif model_type == "roberta":
            logger.info("Deploying RoBERTa classifier app...")
            from roberta_classifier import create_gradio_interface
        elif model_type == "llama":
            logger.info("Deploying Llama classifier app...")
            from lora_classifier import create_gradio_interface
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Create and launch interface
        iface = create_gradio_interface()
        iface.launch(share=share)
        
    except Exception as e:
        logger.error(f"Error deploying app: {e}")
        raise

def main():
    """Run deployment from command line."""
    parser = argparse.ArgumentParser(description="Deploy document classification app")
    parser.add_argument("--model", type=str, choices=["deberta", "roberta", "llama"],
                      required=True, help="Model type to use")
    parser.add_argument("--no-share", action="store_true",
                      help="Don't share the app publicly")
    
    args = parser.parse_args()
    deploy_app(args.model, share=not args.no_share)

if __name__ == "__main__":
    main() 
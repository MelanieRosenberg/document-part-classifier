# Document Part Classifier

A machine learning model for classifying different parts of documents (TEXT, TABLE, FORM) using either DeBERTa or LLaMA with LoRA fine-tuning.

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd document-part-classifier
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
# OR
.\venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Processing

The data processing pipeline consists of several steps:

1. Raw Data Analysis:
```bash
python src/data/process_raw_data.py
```
This script will:
- Analyze raw XML files in `data/raw/`
- Generate statistics about tag distribution
- Create initial data files

2. Data Splitting and Balancing:
```bash
cd src/data
python document_aware_train_val_test_splitter.py  # Creates train/val/test splits
python document_aware_balanced_sampler.py         # Balances the training data
```

Alternatively, you can use the provided notebooks in the `src/data` directory:
- `raw_document_analysis.ipynb`: Analyzes raw XML files
- `document_aware_train_val_test_splitter.ipynb`: Creates train/val/test splits
- `document_aware_balanced_sampler.ipynb`: Balances the training data

The processed data will be organized as follows:
```
data/
├── raw/                    # Original XML files
├── train/                  # Training data
│   ├── lines.txt          # Input text lines
│   └── labels.txt         # Corresponding labels
├── val/                    # Validation data
│   ├── lines.txt
│   └── labels.txt
└── test/                  # Test data
    ├── lines.txt
    └── labels.txt
```

## Training

### DeBERTa Model

Train the DeBERTa model with:
```bash
python src/training/train_deberta.py \
    --model_name microsoft/deberta-v3-base \
    --train_data_dir data/train \
    --val_data_dir data/val \
    --output_dir models/deberta \
    --batch_size 16 \
    --num_epochs 10 \
    --learning_rate 2e-5 \
    --context_window 2
```

### LLaMA with LoRA

Train LLaMA with LoRA adapters:
```bash
python src/training/train_llama_lora.py \
    --model_name llama-3.2-1b \
    --train_data_dir data/train \
    --val_data_dir data/val \
    --output_dir models/llama_lora \
    --batch_size 8 \
    --num_epochs 5 \
    --learning_rate 1e-4 \
    --lora_r 8 \
    --lora_alpha 32
```

Note: For LLaMA training, you need:
- Sufficient GPU memory (at least 8GB recommended)
- HuggingFace token with appropriate permissions

## Evaluation

Evaluate model performance with:
```bash
python src/training/evaluate.py \
    --model_path models/your_model \
    --test_data_dir data/test \
    --output_dir evaluation_results
```

This will generate:
- Confusion matrix
- Per-tag metrics (precision, recall, F1)
- Overall accuracy statistics
- Error analysis

## Gradio App (Beta)

A Gradio web interface is available for quick testing, but it's currently in beta and not intended for production use:
```bash
python src/app/deploy_app.py
```

Note: The app provides basic functionality for testing the model but may not handle all edge cases or provide optimal performance.

## Performance Metrics

Target metrics for both models:
- F1 score > 0.90 for TEXT, TABLE, FORM tags
- Balanced precision and recall
- Robust handling of document context
- Less than 2 seconds for inference


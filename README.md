# Document Part Classifier

A machine learning model for classifying different parts of documents (TEXT, TABLE, FORM) with high accuracy.

## Features

- Two model implementations:
  1. DeBERTa-v3 with LoRA adapters (PyTorch)
  2. BiLSTM-CRF with rich feature engineering (TensorFlow)
- Rich feature engineering for document structure
- Context-aware predictions using surrounding lines
- Post-processing rules for smooth tag transitions
- Handles both tagged and untagged documents
- Focuses on achieving 90% F1 score for primary tags (TEXT, TABLE, FORM)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd document-part-classifier
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

#### DeBERTa-v3 Model (PyTorch)
```bash
python train_enhanced.py \
    --data_dir /path/to/data \
    --output_dir /path/to/output \
    --model_name microsoft/deberta-v3-base \
    --batch_size 16 \
    --num_epochs 10 \
    --learning_rate 2e-5 \
    --context_window 2
```

#### BiLSTM-CRF Model (TensorFlow)
```bash
python train_tf.py \
    --data_dir /path/to/data \
    --output_dir /path/to/output \
    --window_size 10 \
    --lstm_units 128 64 \
    --dropout_rate 0.2 \
    --learning_rate 0.001 \
    --batch_size 32 \
    --num_epochs 15
```

Key arguments for TensorFlow model:
- `--data_dir`: Directory containing train/val/test data
- `--output_dir`: Directory to save model and results
- `--window_size`: Size of context window (default: 10)
- `--lstm_units`: Number of units in LSTM layers (default: [128, 64])
- `--dropout_rate`: Dropout rate for regularization (default: 0.2)
- `--learning_rate`: Learning rate (default: 0.001)
- `--batch_size`: Batch size for training (default: 32)
- `--num_epochs`: Number of epochs (default: 15)
- `--patience`: Number of epochs to wait before early stopping (default: 3)
- `--seed`: Random seed for reproducibility (default: 42)

### Inference

#### DeBERTa-v3 Model
```bash
python predict.py \
    --model_dir /path/to/trained/model \
    --label_map /path/to/label_map.json \
    --input_dir /path/to/input/files \
    --output_dir /path/to/output/files \
    --batch_size 32
```

#### BiLSTM-CRF Model
```bash
python predict_tf.py \
    --model_dir /path/to/trained/model \
    --input_dir /path/to/input/files \
    --output_dir /path/to/output/files
```

## Model Architecture

### DeBERTa-v3 Model
1. Base Model:
   - DeBERTa-v3 base with LoRA adapters
   - Efficient fine-tuning with parameter-efficient adapters

2. Feature Engineering:
   - Layout features (indentation, line spacing)
   - Text features (word count, special characters)
   - Form-specific features (checkboxes, fill-in fields)
   - Table-specific features (vertical bars, horizontal lines)

3. CRF Layer:
   - Conditional Random Field for sequence labeling
   - Better handling of tag dependencies

### BiLSTM-CRF Model
1. Feature Engineering:
   - Rich document structure features
   - Spacing and indentation analysis
   - Form and table pattern detection
   - Text content analysis

2. Model Architecture:
   - Bidirectional LSTM layers
   - CRF layer for sequence labeling
   - Context window processing
   - Dropout regularization

3. Post-processing:
   - Smoothing rules for tag transitions
   - Structure validation
   - Pattern-based corrections

## Data Format

Input documents should be plain text files with one line per line of text. The model will add XML-like tags around sections:

```
<TEXT>
This is a paragraph of text.
It can span multiple lines.
</TEXT>
<TABLE>
| Header 1 | Header 2 |
|----------|----------|
| Value 1  | Value 2  |
</TABLE>
<FORM>
Name: _______________
Date: _______________
</FORM>
```

## Performance

The models aim to achieve:
- 90% F1 score for primary tags (TEXT, TABLE, FORM)
- Processing time under 2 seconds for small input files
- Robust handling of both tagged and untagged documents

## License

[Add your license information here] 
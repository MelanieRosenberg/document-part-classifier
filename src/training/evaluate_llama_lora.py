import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
from datasets import Dataset
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import os
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Define your document tags/classes
id2label = {0: "FORM", 1: "TABLE", 2: "TEXT"}
label2id = {"FORM": 0, "TABLE": 1, "TEXT": 2}

# Model configuration
model_name = "/home/azureuser/llama_models/llama-3-2-1b"  # Base model path
adapter_path = "models/llama_1b_lora/run_20250322_213248/final_model"  # Path to your trained LoRA adapter

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load base model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id,
    device_map="auto",
    local_files_only=True
)

model.config.pad_token_id = tokenizer.pad_token_id

# Load and apply the trained LoRA adapter
model = PeftModel.from_pretrained(model, adapter_path)
print("Loaded trained model")

# Function to preprocess your data for sequence classification
def preprocess_function(examples):
    # Tokenize texts
    tokenized_inputs = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Add labels
    tokenized_inputs["labels"] = examples["label"]
    
    return tokenized_inputs

# Load and prepare your datasets
def load_data():
    def load_split(split_dir):
        with open(os.path.join(split_dir, "lines.txt"), "r") as f:
            texts = [line.strip() for line in f.readlines()]
        with open(os.path.join(split_dir, "labels.txt"), "r") as f:
            labels = [label2id[tag.strip()] for tag in f.readlines()]
        return {"text": texts, "label": labels}
    
    # Load only test data
    test_data = load_split("data/test")
    test_dataset = Dataset.from_dict(test_data)
    
    return test_dataset

# Load test dataset
test_dataset = load_data()

# Preprocess test dataset
test_dataset = test_dataset.map(preprocess_function, batched=True)

# Function to evaluate the model
def evaluate_model(model, dataset, batch_size=8):
    model.eval()
    all_predictions = []
    all_labels = []
    
    # Create a PyTorch DataLoader for batching
    from torch.utils.data import DataLoader
    from transformers import default_data_collator
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=default_data_collator,
        shuffle=False
    )
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        # Get actual labels
        labels = batch["labels"].cpu().numpy()
        all_labels.extend(labels)
        
        # Move batch to device
        inputs = {k: v.to(model.device) for k, v in batch.items() if k != "labels"}
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_predictions.extend(predictions)
    
    # Calculate final metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, labels=[0, 1, 2]
    )
    
    # Overall metrics
    overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted'
    )
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[id2label[i] for i in range(len(id2label))],
                yticklabels=[id2label[i] for i in range(len(id2label))])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')

    return {
        "overall_f1": overall_f1,
        "form_f1": f1[0],
        "table_f1": f1[1],
        "text_f1": f1[2],
    }


# Run evaluation
print("Starting evaluation...")
results = evaluate_model(model, test_dataset)
print("\nTest Results:")
print(f"Overall F1: {results['overall_f1']:.4f}")
print(f"FORM F1: {results['form_f1']:.4f}")
print(f"TABLE F1: {results['table_f1']:.4f}")
print(f"TEXT F1: {results['text_f1']:.4f}")

# Example inference
if __name__ == "__main__":
    # Test examples
    test_examples = [
        "This is an example of a form field with labels and input boxes.",
        "| Name | Age | Occupation |\n|------|-----|------------|\n| John | 30  | Engineer   |",
        "This is a regular text paragraph with information about the document."
    ]
    
    # Run classification
    for text in test_examples:
        # Tokenize input
        inputs = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(model.device)
        
        # Get prediction
        try:
            with torch.no_grad():
                outputs = model(**inputs)
                prediction = torch.argmax(outputs.logits, dim=1)
                predicted_label = id2label[prediction[0].item()]
            print(f"Text: {text[:50]}...\nPrediction: {predicted_label}\n")
        except Exception as e:
            print(f"Error processing example: {e}")

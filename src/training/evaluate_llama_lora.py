import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import Dataset
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import os
from tqdm import tqdm

# Define your document tags/classes
id2label = {0: "FORM", 1: "TABLE", 2: "TEXT"}
label2id = {"FORM": 0, "TABLE": 1, "TEXT": 2}

# Model configuration
model_name = "/home/azureuser/llama_models/llama-3-2-1b"  # Base model path
adapter_path = "models/llama_1b_lora/run_20250322_202241/final_model"  # Path to your trained LoRA adapter

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    local_files_only=True
)

# Load and apply the trained LoRA adapter
model = PeftModel.from_pretrained(model, adapter_path)
print("Loaded trained model")

# Define prompt template for document classification
def create_prompt(text):
    return f"""<s>[INST] Classify the following document segment into one of these categories: FORM, TABLE, or TEXT.
The segment is delimited by triple backticks.
```
{text}
```
Classification: [/INST]"""

# Function to preprocess your data for causal LM
def preprocess_function(examples):
    # Create input prompts from text
    prompts = [create_prompt(text) for text in examples["text"]]
    
    # Tokenize prompts
    tokenized_inputs = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Prepare labels for each example
    labels = []
    for label_id in examples["label"]:
        label_text = f" {id2label[label_id]}</s>"
        tokenized_label = tokenizer(label_text, return_tensors="pt")
        labels.append(tokenized_label.input_ids[0])
    
    # Tokenize and prepare inputs with labels for causal LM
    tokenized_examples = {
        "input_ids": [],
        "attention_mask": [],
        "labels": []
    }
    
    for i, prompt_ids in enumerate(tokenized_inputs.input_ids):
        prompt_length = len(prompt_ids)
        label_ids = labels[i]
        
        # Create input_ids by combining prompt and label
        input_ids = torch.cat([prompt_ids, label_ids])
        
        # Create attention mask
        attention_mask = torch.ones_like(input_ids)
        
        # Create label_ids (set to -100 for prompt tokens to ignore them in loss)
        label_ids_with_ignore = torch.cat([
            torch.ones(prompt_length, dtype=torch.long) * -100,
            label_ids
        ])
        
        # Add to tokenized examples
        tokenized_examples["input_ids"].append(input_ids)
        tokenized_examples["attention_mask"].append(attention_mask)
        tokenized_examples["labels"].append(label_ids_with_ignore)
    
    return tokenized_examples

# Load and prepare your datasets
def load_data():
    def load_split(split_dir):
        with open(os.path.join(split_dir, "lines.txt"), "r") as f:
            texts = [line.strip() for line in f.readlines()]
        with open(os.path.join(split_dir, "tags.txt"), "r") as f:
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

# Function to clean up generated text
def clean_generated_text(text):
    # Remove special tokens
    special_tokens = ["<|begin_of_text|>", "<|end_of_text|>", "<s>", "</s>"]
    for token in special_tokens:
        text = text.replace(token, "")
    
    # Remove prompt if it was repeated
    if "Classify the following document segment" in text:
        text = text.split("Classify the following document segment")[0].strip()
    
    # Clean up any remaining whitespace
    text = text.strip()
    return text

# Function to evaluate the model
def evaluate_model(model, dataset):
    model.eval()
    all_predictions = []
    all_labels = []
    total_examples = len(dataset)
    report_interval = max(1, total_examples // 10)  # Report every 10%
    
    # Track unexpected outputs
    unexpected_outputs = []
    
    for idx, example in enumerate(tqdm(dataset, desc="Evaluating")):
        # Get actual label
        label_tokens = [id for id in example["labels"] if id != -100]
        if label_tokens:
            true_label_text = tokenizer.decode(label_tokens)
            for label_id, label_name in id2label.items():
                if label_name in true_label_text:
                    all_labels.append(label_id)
                    break
        
        # Get prediction
        input_ids = torch.tensor(example["input_ids"])
        attention_mask = torch.tensor(example["attention_mask"])
        
        inputs = {
            "input_ids": input_ids.unsqueeze(0).to(model.device),
            "attention_mask": attention_mask.unsqueeze(0).to(model.device)
        }
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True,  # Enable sampling
                temperature=0.1,  # Low temperature for more deterministic output
                top_p=0.9,  # Nucleus sampling
                num_beams=1,  # No beam search
                early_stopping=False  # Disable early stopping since we're not using beam search
            )
        
        # Decode the generated output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Extract response more robustly
        response = ""
        if "[/INST]" in generated_text:
            try:
                response = generated_text.split("[/INST]")[1].strip()
            except IndexError:
                response = generated_text
                unexpected_outputs.append({
                    "idx": idx,
                    "text": example["text"][:100],
                    "generated": generated_text,
                    "reason": "split_error"
                })
        else:
            response = generated_text
            unexpected_outputs.append({
                "idx": idx,
                "text": example["text"][:100],
                "generated": generated_text,
                "reason": "no_inst_marker"
            })
        
        # Clean up the response
        response = clean_generated_text(response)
        
        # Extract the predicted class
        predicted_label = None
        for label_name in ["FORM", "TABLE", "TEXT"]:
            if label_name in response:
                predicted_label = label2id[label_name]
                break
        
        # If no class is clearly identified, default to TEXT
        if predicted_label is None:
            predicted_label = 2
            unexpected_outputs.append({
                "idx": idx,
                "text": example["text"][:100],
                "generated": response,
                "reason": "no_class_found"
            })
            
        all_predictions.append(predicted_label)
        
        # Report intermediate results every 10%
        if (idx + 1) % report_interval == 0:
            # Calculate current metrics
            current_precision, current_recall, current_f1, _ = precision_recall_fscore_support(
                all_labels, all_predictions, average=None, labels=[0, 1, 2]
            )
            current_overall_f1 = precision_recall_fscore_support(
                all_labels, all_predictions, average='weighted'
            )[2]
            
            print(f"\nIntermediate Results at {(idx + 1) / total_examples * 100:.1f}%:")
            print(f"Overall F1: {current_overall_f1:.4f}")
            print(f"FORM F1: {current_f1[0]:.4f}")
            print(f"TABLE F1: {current_f1[1]:.4f}")
            print(f"TEXT F1: {current_f1[2]:.4f}")
            
            # Print some example outputs if we have unexpected ones
            if unexpected_outputs:
                print("\nExample unexpected outputs:")
                for i, unexpected in enumerate(unexpected_outputs[:3]):  # Show up to 3 examples
                    print(f"\nExample {i+1}:")
                    print(f"Input text: {unexpected['text']}")
                    print(f"Generated: {unexpected['generated']}")
                    if 'reason' in unexpected:
                        print(f"Reason: {unexpected['reason']}")
    
    # Calculate final metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, labels=[0, 1, 2]
    )
    
    # Overall metrics
    overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted'
    )
    
    # Print summary of unexpected outputs
    if unexpected_outputs:
        print(f"\nTotal unexpected outputs: {len(unexpected_outputs)}")
        print(f"Percentage of unexpected outputs: {(len(unexpected_outputs) / total_examples) * 100:.2f}%")
    
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
        prompt = create_prompt(text)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                num_return_sequences=1,
            )
        
        # Decode the generated output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        response = generated_text.split("[/INST]")[1].strip()
        
        # Extract the predicted class
        prediction = "TEXT"  # default
        for label_name in ["FORM", "TABLE", "TEXT"]:
            if label_name in response:
                prediction = label_name
                break
        
        print(f"Text: {text[:50]}...\nPrediction: {prediction}\n")
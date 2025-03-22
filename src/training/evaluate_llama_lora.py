import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from datasets import Dataset
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import os
from tqdm import tqdm

# Define your document tags/classes
id2label = {0: "FORM", 1: "TABLE", 2: "TEXT"}
label2id = {"FORM": 0, "TABLE": 1, "TEXT": 2}

# Model configuration
model_name = "/home/azureuser/llama_models/llama-3-2-1b"  # Full path to the directory containing model.safetensors
use_4bit = True  # Use 4-bit quantization to save memory

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set padding token

# Load model with quantization for memory efficiency
if use_4bit:
    from transformers import BitsAndBytesConfig
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=True  # Only use local files
    )
    model = prepare_model_for_kbit_training(model)
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        local_files_only=True  # Only use local files
    )

# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# Apply LoRA configuration to model
model = get_peft_model(model, lora_config)
print(f"Trainable parameters: {model.print_trainable_parameters()}")

# Define prompt template for document classification
def create_prompt(text):
    return f"""<s>[INST] Classify the following document segment into one of these categories: FORM, TABLE, or TEXT.
The segment is delimited by triple backticks.
```
{text}
```
Classification: [/INST]"""

# Function to preprocess your data for causal LM training
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
    
    # Load data from the appropriate directories
    train_data = load_split("data/train")
    val_data = load_split("data/val")
    test_data = load_split("data/test")
    
    # Convert to datasets
    train_dataset = Dataset.from_dict(train_data)
    val_dataset = Dataset.from_dict(val_data)
    test_dataset = Dataset.from_dict(test_data)
    
    return train_dataset, val_dataset, test_dataset

train_dataset, val_dataset, test_dataset = load_data()

# Preprocess datasets
train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="llama_lora_document_classifier",
    learning_rate=2e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    logging_dir="./logs",
    fp16=True,  # Use mixed precision
)

# Custom Trainer for evaluation
class DocumentClassificationTrainer(Trainer):
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # Run standard evaluation
        output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Custom evaluation focusing on class predictions
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
            
        # Get predictions
        predictions = self.predict(eval_dataset)
        
        # Process predictions to extract class labels
        predicted_labels = []
        true_labels = []
        
        for i, example in enumerate(eval_dataset):
            # Get actual label
            label_tokens = [j for j, id in enumerate(example["labels"]) if id != -100]
            if label_tokens:
                true_label_text = tokenizer.decode(example["labels"][label_tokens])
                for label_id, label_name in id2label.items():
                    if label_name in true_label_text:
                        true_labels.append(label_id)
                        break
            
            # Get predicted label from model output
            logits = predictions.predictions[i]
            prompt_length = sum(1 for id in example["labels"] if id == -100)
            
            # Get logits for the first token of the answer
            next_token_logits = logits[prompt_length - 1]
            
            # Find most likely token
            predicted_token_id = np.argmax(next_token_logits)
            predicted_token = tokenizer.decode(predicted_token_id)
            
            # Map to class
            predicted_label = None
            for label_id, label_name in id2label.items():
                if label_name in predicted_token or predicted_token in label_name:
                    predicted_label = label_id
                    break
            
            # If no matching class, get full prediction and try matching
            if predicted_label is None:
                # Get top 5 tokens
                top_token_ids = np.argsort(next_token_logits)[-5:]
                for token_id in top_token_ids:
                    token = tokenizer.decode(token_id)
                    for label_id, label_name in id2label.items():
                        if label_name in token or token in label_name:
                            predicted_label = label_id
                            break
                    if predicted_label is not None:
                        break
            
            # If still no match, default to most common class
            if predicted_label is None:
                predicted_label = 2  # TEXT class as default
                
            predicted_labels.append(predicted_label)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predicted_labels, average=None, labels=[0, 1, 2]
        )
        
        # Overall metrics
        overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
            true_labels, predicted_labels, average='weighted'
        )
        
        # Add metrics to output
        output.update({
            f"{metric_key_prefix}_overall_f1": overall_f1,
            f"{metric_key_prefix}_form_f1": f1[0],
            f"{metric_key_prefix}_table_f1": f1[1],
            f"{metric_key_prefix}_text_f1": f1[2],
        })
        
        return output

# Initialize Trainer
trainer = DocumentClassificationTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Evaluate on test set
test_results = trainer.evaluate(test_dataset)
print("Test Results:", test_results)

# Save the model
model.save_pretrained("./llama_lora_document_classifier_final")

# Function for inference on new data
def classify_document(text, model, tokenizer):
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
    for label_name in ["FORM", "TABLE", "TEXT"]:
        if label_name in response:
            return label_name
    
    # Default if no class is clearly identified
    return "TEXT"

# Example inference
if __name__ == "__main__":
    # Load the saved model for inference
    inference_model = AutoModelForCausalLM.from_pretrained(
        "./llama_lora_document_classifier_final",
        device_map="auto"
    )
    
    # Test examples
    test_examples = [
        "This is an example of a form field with labels and input boxes.",
        "| Name | Age | Occupation |\n|------|-----|------------|\n| John | 30  | Engineer   |",
        "This is a regular text paragraph with information about the document."
    ]
    
    # Run classification
    for text in test_examples:
        prediction = classify_document(text, inference_model, tokenizer)
        print(f"Text: {text[:50]}...\nPrediction: {prediction}\n")
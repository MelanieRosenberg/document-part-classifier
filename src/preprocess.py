import os
import logging
from pathlib import Path
import random
from typing import List, Tuple
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
import re
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_xml_content(content: str) -> str:
    """Clean XML content to handle common formatting issues."""
    # Remove XML declaration if present
    if content.startswith('<?xml'):
        content = content.split('?>', 1)[1]
    
    # Remove BOM if present
    if content.startswith('\ufeff'):
        content = content[1:]
    
    # Clean whitespace
    content = content.strip()
    
    # Normalize line endings
    content = content.replace('\r\n', '\n')
    
    # Fix common XML issues
    content = re.sub(r'&(?![a-zA-Z]+;)', '&amp;', content)  # Fix unescaped ampersands
    content = re.sub(r'<(?![a-zA-Z/])', '&lt;', content)   # Fix unescaped <
    content = re.sub(r'(?<![a-zA-Z])>', '&gt;', content)   # Fix unescaped >
    
    return content

def extract_content_between_tags(content: str, tag: str) -> List[str]:
    """Extract content between opening and closing tags using regex."""
    pattern = f'<{tag}>(.*?)</{tag}>'
    matches = re.finditer(pattern, content, re.DOTALL)
    contents = []
    
    for match in matches:
        text = match.group(1).strip()
        # Split into lines and filter empty ones
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        contents.extend(lines)
    
    return contents

def process_file(file_path: Path) -> List[Tuple[str, str]]:
    """Process a single XML file and return list of (tag, line) tuples."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Clean the content
        content = clean_xml_content(content)
        results = []
        target_tags = ['TABLE', 'FORM', 'TEXT']
        
        # Try both XML parsing and regex approaches, combine results
        for tag in target_tags:
            # Try regex extraction first
            contents = extract_content_between_tags(content, tag)
            
            # Add each non-empty line
            for text in contents:
                if text.strip():
                    results.append((tag, text.strip()))
            
            # Also try XML parsing as backup
            try:
                if not content.strip().startswith('<document>'):
                    content = f'<document>{content}</document>'
                root = ET.fromstring(content)
                elements = root.findall(f'.//{tag}')
                for elem in elements:
                    if elem.text and elem.text.strip():
                        results.append((tag, elem.text.strip()))
            except ET.ParseError:
                logger.warning(f"XML parsing failed for {file_path} with tag {tag}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_results = []
        for item in results:
            if item not in seen:
                seen.add(item)
                unique_results.append(item)
        
        return unique_results
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return []

def oversample_minority_class(data: List[Tuple[str, str]], target_size: int = None) -> List[Tuple[str, str]]:
    """Oversample minority classes to match the size of the majority class.
    
    Args:
        data: List of (tag, text) tuples
        target_size: Optional target size for each class (if None, uses max class size)
        
    Returns:
        List of (tag, text) tuples with balanced classes
    """
    # Group by tag
    tag_groups = defaultdict(list)
    for tag, text in data:  # Correct order: (tag, text)
        tag_groups[tag].append((tag, text))
    
    # Find the size of the largest class
    max_size = max(len(examples) for examples in tag_groups.values())
    if target_size is not None:
        max_size = max(max_size, target_size)
    
    logger.info(f"Target size for oversampling: {max_size}")
    
    # Oversample each class to match the largest class
    balanced_data = []
    for tag, examples in tag_groups.items():
        current_size = len(examples)
        logger.info(f"Class {tag} before oversampling: {current_size} examples")
        
        if current_size < max_size:
            # Oversample with replacement
            num_samples_needed = max_size - current_size
            oversampled = random.choices(examples, k=num_samples_needed)
            balanced_class = examples + oversampled
            balanced_data.extend(balanced_class)
            logger.info(f"Class {tag} after oversampling: {len(balanced_class)} examples")
        else:
            balanced_data.extend(examples)
            logger.info(f"Class {tag} unchanged: {current_size} examples")
    
    # Shuffle the balanced data
    random.shuffle(balanced_data)
    return balanced_data

def split_data(data: List[Tuple[str, str]], train_ratio: float = 0.7, val_ratio: float = 0.15, min_samples: int = 300) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Split data into train, validation, and test sets with stratification and minimum samples.
    
    Args:
        data: List of (tag, text) tuples
        train_ratio: Proportion of data to use for training
        val_ratio: Proportion of data to use for validation
        min_samples: Minimum number of samples for each class in val and test sets
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    # Group data by tag
    tag_groups = defaultdict(list)
    for tag, text in data:
        tag_groups[tag].append((tag, text))
    
    train_data, val_data, test_data = [], [], []
    
    for tag, examples in tag_groups.items():
        total_samples = len(examples)
        logger.info(f"Processing split for {tag} with {total_samples} total samples")
        
        # Calculate normal split based on ratios
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)
        test_size = total_samples - train_size - val_size
        
        # Ensure minimum samples in val and test if possible
        if total_samples > 2 * min_samples:  # Only if we have enough samples
            desired_val_test = 2 * min_samples  # min_samples for each of val and test
            if train_size > (total_samples - desired_val_test):
                # Reduce train_size to ensure minimum val and test sizes
                train_size = total_samples - desired_val_test
                # Recalculate val_size and test_size with equal split of remaining
                val_size = min_samples
                test_size = min_samples
        
        # Shuffle the examples
        random.seed(42)  # For reproducibility
        random.shuffle(examples)
        
        # Split according to calculated sizes
        train_data.extend(examples[:train_size])
        val_data.extend(examples[train_size:train_size+val_size])
        test_data.extend(examples[train_size+val_size:])
        
        logger.info(f"{tag} split: {train_size} train, {val_size} val, {test_size} test")
    
    # Shuffle each set
    random.seed(42)
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    return train_data, val_data, test_data

def save_data(data: List[Tuple[str, str]], output_dir: Path):
    """Save processed data to output directory."""
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save lines and tags separately
    lines_file = output_dir / "lines.txt"
    tags_file = output_dir / "tags.txt"
    
    with open(lines_file, 'w', encoding='utf-8') as lf, open(tags_file, 'w', encoding='utf-8') as tf:
        for tag, text in data:  # The order is (tag, text)
            lf.write(f"{text}\n")
            tf.write(f"{tag}\n")

def main():
    # Initialize paths
    data_dir = Path("data")
    input_dir = data_dir / "raw"
    
    # Process all XML files
    all_data = []
    for xml_file in input_dir.glob("*.xml"):
        logger.info(f"Processing {xml_file}")
        results = process_file(xml_file)
        all_data.extend(results)
    
    # Print initial statistics
    tag_counts = {}
    for tag, _ in all_data:
        tag_counts[tag] = tag_counts.get(tag, 0) + 1
    
    logger.info("\nInitial Tag Distribution:")
    for tag, count in tag_counts.items():
        logger.info(f"{tag}: {count} lines")
    
    if not tag_counts:
        logger.error("No data was processed successfully!")
        exit(1)
    
    # Split the data with stratification and minimum samples
    # Try to ensure at least 300 samples of each class in validation and test sets
    train_data, val_data, test_data = split_data(
        all_data, 
        train_ratio=0.6,  # Slightly reduced to allow more samples for val/test
        val_ratio=0.2,    # Slightly increased 
        min_samples=300   # Target minimum samples per class in val and test
    )
    
    # Balance the training set
    balanced_train_data = oversample_minority_class(train_data)
    
    # Print balanced statistics for training set
    balanced_counts = {}
    for tag, _ in balanced_train_data:
        balanced_counts[tag] = balanced_counts.get(tag, 0) + 1
    
    logger.info("\nBalanced Training Set Distribution:")
    for tag, count in balanced_counts.items():
        logger.info(f"{tag}: {count} lines")
    
    # Print validation and test set distributions
    val_counts = {}
    test_counts = {}
    for tag, _ in val_data:
        val_counts[tag] = val_counts.get(tag, 0) + 1
    for tag, _ in test_data:
        test_counts[tag] = test_counts.get(tag, 0) + 1
    
    logger.info("\nValidation Set Distribution:")
    for tag, count in val_counts.items():
        logger.info(f"{tag}: {count} lines")
    
    logger.info("\nTest Set Distribution:")
    for tag, count in test_counts.items():
        logger.info(f"{tag}: {count} lines")
    
    # Save the datasets
    save_data(balanced_train_data, data_dir / "train")
    save_data(val_data, data_dir / "val") 
    save_data(test_data, data_dir / "test")

if __name__ == "__main__":
    main()
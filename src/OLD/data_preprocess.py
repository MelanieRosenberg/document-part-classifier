import os
from typing import List, Dict, Tuple
import shutil
from collections import defaultdict
import logging
from sklearn.model_selection import train_test_split
import re
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Primary tags we care about
PRIMARY_TAGS = {'TEXT', 'FORM', 'TABLE'}

def extract_labeled_examples(content: str) -> List[Tuple[str, str]]:
    """
    Extract text segments and their corresponding labels from XML content.
    Returns a list of (text, label) tuples.
    """
    examples = []
    
    # Pattern to match content between tags
    pattern = r'<(' + '|'.join(PRIMARY_TAGS) + r')>(.*?)</\1>'
    matches = re.finditer(pattern, content, re.DOTALL)
    
    for match in matches:
        tag = match.group(1)  # The tag name
        text = match.group(2).strip()  # The content between tags
        if text:  # Only include non-empty examples
            examples.append((text, tag))
    
    return examples

def analyze_file_content(content: str) -> Dict:
    """
    Analyze the content of a file to determine its structure.
    """
    analysis = {
        'has_xml_tags': False,
        'all_tags': [],
        'primary_tags_found': [],
        'content_structure': [],
        'content_preview': content[:200] + '...' if len(content) > 200 else content,
        'labeled_examples': []
    }
    
    # Look for all XML tags in order of appearance
    tag_pattern = r'</?([A-Za-z][^>]*)>'
    xml_tags = re.findall(tag_pattern, content)
    if xml_tags:
        analysis['has_xml_tags'] = True
        analysis['all_tags'] = list(set(xml_tags))
        analysis['primary_tags_found'] = [tag for tag in analysis['all_tags'] if tag in PRIMARY_TAGS]
        
        # Get content structure (sequence of tags)
        tag_sequence = re.finditer(tag_pattern, content)
        analysis['content_structure'] = [match.group(0) for match in tag_sequence][:10]  # First 10 tags
        
        # Extract labeled examples
        analysis['labeled_examples'] = extract_labeled_examples(content)
    
    # Look for actual content patterns
    lines = content.split('\n')
    non_empty_lines = [line.strip() for line in lines if line.strip()]
    if non_empty_lines:
        analysis['first_lines'] = non_empty_lines[:5]  # First 5 non-empty lines
        
    return analysis

def setup_data_directory(
    input_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> None:
    """Analyze input files and prepare training data."""
    logger.info("\nAnalyzing input files:")
    
    file_stats = defaultdict(int)
    all_tags_found = defaultdict(int)
    file_analyses = []
    all_examples = []
    
    for fname in os.listdir(input_dir):
        file_path = os.path.join(input_dir, fname)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            analysis = analyze_file_content(content)
            analysis['filename'] = fname
            file_analyses.append(analysis)
            
            # Update statistics
            if analysis['has_xml_tags']:
                file_stats['has_xml_tags'] += 1
                for tag in analysis['all_tags']:
                    all_tags_found[tag] += 1
                if analysis['primary_tags_found']:
                    file_stats['has_primary_tags'] += 1
                    all_examples.extend(analysis['labeled_examples'])
            
        except Exception as e:
            logger.error(f"Error processing {fname}: {str(e)}")
            continue
    
    # Log findings
    logger.info("\nFile Analysis Results:")
    logger.info(f"Total files processed: {len(file_analyses)}")
    logger.info(f"Files with XML tags: {file_stats['has_xml_tags']}")
    logger.info(f"Files with primary tags: {file_stats['has_primary_tags']}")
    logger.info(f"Total labeled examples extracted: {len(all_examples)}")
    
    # Count examples per class
    class_distribution = defaultdict(int)
    for _, label in all_examples:
        class_distribution[label] += 1
    
    logger.info("\nClass Distribution:")
    for label, count in class_distribution.items():
        logger.info(f"- {label}: {count} examples")
    
    logger.info("\nAll XML Tags Found:")
    for tag, count in sorted(all_tags_found.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"- {tag}: {count} files")
    
    # Split data into train/val/test sets
    if all_examples:
        # First split into train and temp
        train_examples, temp_examples = train_test_split(
            all_examples, 
            train_size=train_ratio,
            random_state=42,
            stratify=[label for _, label in all_examples]
        )
        
        # Then split temp into val and test
        val_size = val_ratio / (1 - train_ratio)
        val_examples, test_examples = train_test_split(
            temp_examples,
            train_size=val_size,
            random_state=42,
            stratify=[label for _, label in temp_examples]
        )
        
        # Save splits
        os.makedirs(output_dir, exist_ok=True)
        
        def save_split(examples: List[Tuple[str, str]], filename: str):
            with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
                for text, label in examples:
                    f.write(f"{label}\t{text}\n")
        
        save_split(train_examples, "train.tsv")
        save_split(val_examples, "val.tsv")
        save_split(test_examples, "test.tsv")
        
        logger.info(f"\nData splits saved to {output_dir}:")
        logger.info(f"- Train set: {len(train_examples)} examples")
        logger.info(f"- Validation set: {len(val_examples)} examples")
        logger.info(f"- Test set: {len(test_examples)} examples")
    
    # Save detailed analysis
    analysis_file = os.path.join(output_dir, "file_analysis.json")
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'file_stats': dict(file_stats),
                'all_tags_found': dict(all_tags_found),
                'class_distribution': dict(class_distribution)
            },
            'files': file_analyses
        }, f, indent=2)
    
    logger.info(f"\nDetailed analysis saved to {analysis_file}")
    
    # Show examples of files without primary tags
    no_primary_tags = [a for a in file_analyses if not a['primary_tags_found']]
    if no_primary_tags:
        logger.info(f"\nExample files without primary tags ({len(no_primary_tags)} total):")
        for example in no_primary_tags[:3]:
            logger.info(f"\nFile: {example['filename']}")
            logger.info(f"Tags found: {example['all_tags']}")
            logger.info("First few lines:")
            for line in example.get('first_lines', []):
                logger.info(line)

if __name__ == "__main__":
    # Example usage
    input_dir = "inputs/problem2"
    output_dir = "document-part-classifier/data"
    
    # Analyze input files and prepare training data
    setup_data_directory(input_dir, output_dir) 
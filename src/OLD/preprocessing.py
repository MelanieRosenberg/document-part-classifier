import re
from typing import List, Tuple, Dict
import numpy as np

class DocumentPreprocessor:
    """Handles document preprocessing for part classification."""
    
    def __init__(self, context_size: int = 3):
        self.context_size = context_size
        self.tag_types = ["TEXT", "TABLE", "FORM", "OTHER"]
        self.tag_to_id = {tag: idx for idx, tag in enumerate(self.tag_types)}
        self.id_to_tag = {idx: tag for tag, idx in self.tag_to_id.items()}
    
    def extract_line_features(self, line: str) -> Dict[str, float]:
        """Extract structural and content features from a line of text."""
        return {
            "length": len(line),
            "word_count": len(line.split()),
            "starts_with_bullet": float(line.strip().startswith(("•", "-", "*"))),
            "uppercase_ratio": sum(1 for c in line if c.isupper()) / max(len(line), 1),
            "digit_ratio": sum(1 for c in line if c.isdigit()) / max(len(line), 1),
            "special_char_ratio": sum(1 for c in line if not c.isalnum() and not c.isspace()) / max(len(line), 1),
            "indentation": len(line) - len(line.lstrip()),
            "has_form_patterns": float(bool(re.search(r'[\_\.\:]{3,}|(?i)(name|address|phone|email|date)[\s\:]+', line))),
            "has_table_patterns": float(bool(re.search(r'[\|\t]{2,}|\s{3,}', line)))
        }
    
    def add_context_to_line(self, lines: List[str], current_idx: int) -> str:
        """Add surrounding context to a single line."""
        start_idx = max(0, current_idx - self.context_size)
        end_idx = min(len(lines), current_idx + self.context_size + 1)
        
        before_context = ' '.join(lines[start_idx:current_idx])
        current_line = lines[current_idx]
        after_context = ' '.join(lines[current_idx + 1:end_idx])
        
        return f"{before_context} [SEP] {current_line} [SEP] {after_context}"
    
    def process_document(self, document: str) -> Tuple[List[str], List[Dict[str, float]]]:
        """Process a document into lines with context and features."""
        lines = document.split('\n')
        processed_lines = []
        line_features = []
        
        for i, line in enumerate(lines):
            if not line.strip():  # Skip empty lines
                continue
                
            # Get line with context
            line_with_context = self.add_context_to_line(lines, i)
            processed_lines.append(line_with_context)
            
            # Extract features
            features = self.extract_line_features(line)
            line_features.append(features)
        
        return processed_lines, line_features
    
    def extract_sections(self, document: str) -> List[Tuple[str, List[str]]]:
        """Extract sections from a document based on XML-like tags."""
        sections = []
        current_tag = None
        current_lines = []
        
        for line in document.split('\n'):
            # Check for opening tags
            if line.startswith('<') and '>' in line and not line.startswith('</'):
                if current_tag:  # Save previous section
                    sections.append((current_tag, current_lines))
                    current_lines = []
                current_tag = line.strip('<>').strip()
            
            # Check for closing tags
            elif line.startswith('</') and '>' in line:
                if current_tag:  # Save completed section
                    sections.append((current_tag, current_lines))
                    current_lines = []
                    current_tag = None
            
            # Content line
            elif current_tag and line.strip():
                current_lines.append(line)
        
        return sections
    
    def create_sliding_windows(
        self, 
        tokens: List[str], 
        labels: List[int], 
        max_length: int = 512, 
        stride: int = 256
    ) -> Tuple[List[List[str]], List[List[int]]]:
        """Create sliding windows for long sequences."""
        windows = []
        window_labels = []
        
        for i in range(0, len(tokens), stride):
            window_tokens = tokens[i:i + max_length]
            window_label = labels[i:i + max_length]
            
            if len(window_tokens) < 10:  # Skip very small windows
                continue
            
            # Pad if needed
            if len(window_tokens) < max_length:
                padding_length = max_length - len(window_tokens)
                window_tokens.extend(["[PAD]"] * padding_length)
                window_label.extend([self.tag_to_id["OTHER"]] * padding_length)
            
            windows.append(window_tokens)
            window_labels.append(window_label)
        
        return windows, window_labels
    
    def post_process_predictions(
        self, 
        lines: List[str], 
        predicted_tags: List[str]
    ) -> str:
        """Merge consecutive lines with same tag and format as XML."""
        merged_sections = []
        current_tag = None
        current_content = []
        
        for line, tag in zip(lines, predicted_tags):
            if tag != current_tag:
                # Start a new section
                if current_tag:
                    merged_sections.append((current_tag, current_content))
                    current_content = []
                current_tag = tag
            
            current_content.append(line)
        
        # Add the last section
        if current_tag and current_content:
            merged_sections.append((current_tag, current_content))
        
        # Format as XML
        xml_output = ""
        for tag, content in merged_sections:
            xml_output += f"<{tag}>\n"
            xml_output += "\n".join(content) + "\n"
            xml_output += f"</{tag}>\n"
        
        return xml_output 
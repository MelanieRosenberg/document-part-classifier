import os
from pathlib import Path
import re
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_tags(content: str) -> list:
    """Extract all XML tags from content."""
    pattern = r'<([A-Z_]+)>'
    return re.findall(pattern, content)

def analyze_tags():
    """Analyze tag distribution across all XML files."""
    raw_dir = Path("../../data/raw")
    tag_counter = Counter()
    file_counter = 0
    files_with_other_tags = set()
    
    # Standard categories
    standard_tags = {'TEXT', 'TABLE', 'FORM'}
    
    # Process all XML files
    for xml_file in raw_dir.glob("*.xml"):
        try:
            with open(xml_file, 'r', encoding='utf-8') as f:
                content = f.read()
                tags = extract_tags(content)
                tag_counter.update(tags)
                
                # Check if file has non-standard tags
                file_tags = set(tags)
                if file_tags - standard_tags:
                    files_with_other_tags.add(xml_file.name)
                
                file_counter += 1
        except Exception as e:
            logger.error(f"Error processing {xml_file}: {e}")
    
    # Print results
    print(f"\nAnalyzed {file_counter} XML files")
    print("\nTag distribution:")
    print("-" * 40)
    for tag, count in tag_counter.most_common():
        is_standard = "✓" if tag in standard_tags else "✗"
        print(f"{tag:<20} {count:>6} {is_standard}")
    
    # Print files with non-standard tags
    if files_with_other_tags:
        print(f"\nFiles with non-standard tags ({len(files_with_other_tags)}):")
        print("-" * 40)
        for filename in sorted(files_with_other_tags):
            print(filename)

if __name__ == "__main__":
    analyze_tags() 
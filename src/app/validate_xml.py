import os
import xml.etree.ElementTree as ET
from pathlib import Path
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_elements(content: str) -> list:
    """Extract all top-level XML elements from content."""
    # Find all top-level XML elements using regex
    pattern = r'<([A-Z_]+)>(.*?)</\1>'
    matches = re.finditer(pattern, content, re.DOTALL)
    elements = []
    
    for match in matches:
        tag = match.group(1)
        text = match.group(2).strip()
        elements.append((tag, text))
    
    return elements

def validate_xml_file(file_path: str) -> None:
    """Validate XML-like file and print its structure."""
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        print(f"\nAnalyzing {os.path.basename(file_path)}:")
        print("-" * 50)
        
        # Print first few lines of raw content
        print("Raw content preview:")
        lines = content.split('\n')
        print("\n".join(lines[:5]))
        print("...")
        
        # Extract and analyze elements
        elements = extract_elements(content)
        
        if elements:
            print("\nFound elements:")
            for tag, text in elements:
                print(f"\n{tag}:")
                print("-" * len(tag))
                # Print first few lines of text content
                text_lines = text.split('\n')
                for line in text_lines[:5]:
                    if line.strip():
                        print(line)
                if len(text_lines) > 5:
                    print("...")
        else:
            print("\nNo valid XML-like elements found")
            
        # Try standard XML parsing to show any issues
        try:
            # Wrap in root element to attempt parsing
            wrapped_content = f"<ROOT>{content}</ROOT>"
            root = ET.fromstring(wrapped_content)
            print("\nValid when wrapped in ROOT element")
        except ET.ParseError as e:
            print(f"\nXML Parse Error when wrapped: {str(e)}")
            print("This might be due to:")
            print("1. Malformed XML tags")
            print("2. Unclosed tags")
            print("3. Invalid characters")
            
    except Exception as e:
        print(f"Error reading file: {str(e)}")

def main():
    # Path to raw data directory
    raw_dir = Path("../../data/raw")
    
    # Get list of XML files
    xml_files = list(raw_dir.glob("*.xml"))
    
    print(f"Found {len(xml_files)} XML files")
    
    # Ask user what to do
    print("\nOptions:")
    print("1. Analyze all files")
    print("2. Analyze files with errors only")
    print("3. Analyze specific file")
    
    choice = input("\nEnter your choice (1-3): ")
    
    if choice == "1":
        for file in xml_files:
            validate_xml_file(str(file))
    elif choice == "2":
        for file in xml_files:
            try:
                ET.parse(file)
            except ET.ParseError:
                validate_xml_file(str(file))
    elif choice == "3":
        print("\nAvailable files:")
        for i, file in enumerate(xml_files):
            print(f"{i+1}. {file.name}")
        file_num = int(input("\nEnter file number: ")) - 1
        if 0 <= file_num < len(xml_files):
            validate_xml_file(str(xml_files[file_num]))
        else:
            print("Invalid file number")
    else:
        print("Invalid choice")

if __name__ == "__main__":
    # Analyze specific file
    file_path = "../../data/raw/document_part_classification_2.xml"
    validate_xml_file(file_path) 
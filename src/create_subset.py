import os
import shutil

def create_subset(input_dir, output_dir, num_lines=100):
    """Create a subset of the training data with the first num_lines."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read first num_lines from each file
    for filename in ['lines.txt', 'tags.txt']:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = [next(f) for _ in range(num_lines)]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
    
    print(f"Created subset with {num_lines} lines in {output_dir}")

if __name__ == "__main__":
    # Create subset of training data
    create_subset(
        input_dir="data/train",
        output_dir="data/train_subset",
        num_lines=100
    ) 
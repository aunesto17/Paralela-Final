import os
import json
import glob
import random
from tree_sitter import Language, Parser

# Update this path if you are running from a different directory
SO_FILE_PATH = '../../tokenizer/tokompiler/parsers/tokompiler-languages.so'
HECBENCH_SRC_DIR = '/home/aunesto17/Paralela/OMPar/evaluation/data/HeCBench/src'

def get_pragma_for_loop(node, source_code):
    """
    Looks at the previous siblings of a for_statement node to find a #pragma omp.
    Tree-sitter typically parses preprocessor directives as previous siblings to the loop.
    """
    prev = node.prev_sibling
    while prev is not None:
        # Skip over any comments sitting between the pragma and the loop
        if prev.type == 'comment':
            prev = prev.prev_sibling
            continue
        
        # Extract the text of the previous sibling
        prev_text = source_code[prev.start_byte:prev.end_byte].decode('utf8', errors='ignore').strip()
        
        # Check if it is an OpenMP pragma
        if prev_text.startswith('#pragma omp'):
            # Clean it up (e.g., remove newlines if the pragma spans multiple lines with '\')
            clean_pragma = " ".join(prev_text.replace('\\\n', ' ').split())
            return clean_pragma
            
        # If the sibling is neither a comment nor a pragma, stop looking.
        # This means the loop has no pragma attached to it.
        break
        
    return ""

def extract_loops_and_pragmas(node, source_code):
    """
    Recursively finds all for loops and their associated pragmas in the AST.
    """
    samples = []
    if node.type == 'for_statement':
        # Extract the raw text of the for loop
        loop_code = source_code[node.start_byte:node.end_byte].decode('utf8', errors='ignore')
        
        # Look for a preceding pragma
        pragma = get_pragma_for_loop(node, source_code)
        is_positive = bool(pragma)
        
        samples.append({
            "code": loop_code,
            "label": is_positive,
            "pragma": pragma
        })
        
    for child in node.children:
        samples.extend(extract_loops_and_pragmas(child, source_code))
        
    return samples

def main():
    if not os.path.exists(SO_FILE_PATH):
        print(f"Error: Could not find tree-sitter .so file at {SO_FILE_PATH}")
        return

    if not os.path.exists(HECBENCH_SRC_DIR):
        print(f"Error: Could not find HeCBench source directory at {HECBENCH_SRC_DIR}")
        return

    # Initialize Tree-sitter parser
    cpp_language = Language(SO_FILE_PATH, 'cpp')
    parser = Parser()
    parser.set_language(cpp_language)

    output_dataset = 'labeled_use_cases.jsonl'
    
    # 1. Collect all C and C++ source files directly from HeCBench src
    source_files = []
    for ext in ('*.c', '*.cpp', '*.cc', '*.cxx'):
        source_files.extend(glob.glob(os.path.join(HECBENCH_SRC_DIR, '**', ext), recursive=True))

    print(f"Found {len(source_files)} source files. Extracting loops and labels...")

    positive_samples = []
    negative_samples = []

    # 2. Extract all loops
    for file_path in source_files:
        try:
            with open(file_path, 'rb') as f:
                source_code = f.read()
            
            tree = parser.parse(source_code)
            samples = extract_loops_and_pragmas(tree.root_node, source_code)
            
            for sample in samples:
                if sample['label']:
                    positive_samples.append(sample)
                else:
                    negative_samples.append(sample)
        except Exception as e:
            continue # Silently skip files with complex parsing errors

    print(f"\nRaw Extraction Complete:")
    print(f"  - Positive loops (WITH pragmas): {len(positive_samples)}")
    print(f"  - Negative loops (WITHOUT pragmas): {len(negative_samples)}")

    # 3. Balance the dataset to mimic the paper's methodology
    print("\nBalancing dataset to match the paper's distribution (385 positive, 385 negative)...")
    
    # Shuffle to ensure random selection
    random.seed(42) # Set seed for reproducibility 
    random.shuffle(positive_samples)
    random.shuffle(negative_samples)
    
    # Truncate to the paper's specifications (or max available if slightly less)
    target_count = 385
    final_positives = positive_samples[:target_count]
    final_negatives = negative_samples[:len(final_positives)]
    
    final_dataset = final_positives + final_negatives
    random.shuffle(final_dataset) # Mix positive and negative loops together

    # 4. Save to JSONL
    with open(output_dataset, 'w') as outfile:
        for sample in final_dataset:
            outfile.write(json.dumps(sample) + '\n')

    print(f"Success! Saved {len(final_dataset)} labeled loops to {output_dataset}.")
    print("You can now pass this file directly into 'run_ompar.py' to test accuracy.")

if __name__ == '__main__':
    main()
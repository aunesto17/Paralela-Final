import json
import os
from tree_sitter import Language, Parser
from loop_extractor import parse_code_and_get_for_loops

# Use the existing compiled language file from the tokenizer directory
SO_FILE_PATH = '../../tokenizer/tokompiler/parsers/tokompiler-languages.so' 

def main():
    if not os.path.exists(SO_FILE_PATH):
        print(f"Error: Could not find tree-sitter .so file at {SO_FILE_PATH}")
        return

    cpp_language = Language(SO_FILE_PATH, 'cpp')
    parser = Parser()
    parser.set_language(cpp_language)

    input_dataset = 'dataset.jsonl'
    output_dataset = 'extracted_use_cases.jsonl'
    
    extracted_loops_count = 0

    print(f"Reading full files from {input_dataset}...")
    
    with open(input_dataset, 'r', errors='ignore') as infile, open(output_dataset, 'w') as outfile:
        for line in infile:
            sample = json.loads(line.strip())
            full_code = sample.get('code', '')
            
            try:
                loops = parse_code_and_get_for_loops(full_code, parser)
                
                for loop_code_bytes in loops:
                    # tree-sitter returns bytes. We MUST decode to string for JSON serialization
                    loop_code_str = loop_code_bytes.decode('utf8') if isinstance(loop_code_bytes, bytes) else loop_code_bytes
                    
                    out_sample = {
                        "code": loop_code_str,
                        "label": False, 
                        "pragma": ""    
                    }
                    outfile.write(json.dumps(out_sample) + '\n')
                    extracted_loops_count += 1
            except Exception as e:
                # Print the error instead of silently passing
                print(f"Error extracting loop: {e}")

    print(f"\nSuccess! Extracted {extracted_loops_count} individual loops.")
    print(f"Saved to {output_dataset}.")

if __name__ == '__main__':
    main()
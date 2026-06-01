import json
from transformers import GPT2Tokenizer

def main():
    # Load the OMPar tokenizer
    vocab_file = 'tokenizer/gpt/gpt_vocab/gpt2-vocab.json'
    merge_file = 'tokenizer/gpt/gpt_vocab/gpt2-merges.txt'
    tokenizer = GPT2Tokenizer(vocab_file=vocab_file, merges_file=merge_file)

    # Set a safe token limit for an 8GB GPU
    MAX_TOKENS = 800

    input_dataset = 'evaluation/data/labeled_use_cases.jsonl'
    output_dataset = 'evaluation/data/filtered_use_cases.jsonl'

    kept_samples = 0
    dropped_samples = 0

    print(f"Filtering loops longer than {MAX_TOKENS} tokens...")

    with open(input_dataset, 'r') as infile, open(output_dataset, 'w') as outfile:
        for line in infile:
            sample = json.loads(line.strip())
            
            # Count the exact number of tokens the model will see
            token_count = len(tokenizer(sample['code'])['input_ids'])
            
            if token_count <= MAX_TOKENS:
                outfile.write(line)
                kept_samples += 1
            else:
                dropped_samples += 1

    print("=" * 45)
    print("FILTERING COMPLETE")
    print("=" * 45)
    print(f"Loops Kept:    {kept_samples}")
    print(f"Loops Dropped: {dropped_samples}")
    print(f"Saved to:      {output_dataset}")

if __name__ == '__main__':
    main()
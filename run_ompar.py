import argparse
import json
import torch
import os
from compAI import OMPAR

if __name__=='__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.register('type', 'bool', lambda v: v.lower() in ['yes', 'true', 't', '1', 'y'])

    parser.add_argument('--vocab_file', default='tokenizer/gpt/gpt_vocab/gpt2-vocab.json')
    parser.add_argument('--merge_file', default='tokenizer/gpt/gpt_vocab/gpt2-merges.txt')
    parser.add_argument('--model_weights', help='Path to the OMPify model weights', required=True)
    parser.add_argument('--dataset', default='evaluation/data/filtered_use_cases.jsonl', help='Path to the evaluation dataset')
    
    parser.add_argument('--output_results', default='evaluation/ompar_results.json', help='File to save the evaluation metrics')

    main_args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ompar = OMPAR(model_path=main_args.model_weights, device=device, args=main_args)

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    with open(main_args.dataset, 'r') as f:
        for i, line in enumerate(f):
            sample = json.loads(line.strip())

            code = sample['code']
            label = sample['label']

            pred_pragma = ompar.auto_comp(code)
            predicted_positive = bool(pred_pragma)
            
            if label and predicted_positive:
                tp += 1
            elif not label and predicted_positive:
                fp += 1
            elif not label and not predicted_positive:
                tn += 1
            elif label and not predicted_positive:
                fn += 1

            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1} loops...")

    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    accuracy = (tp + tn) / total if total > 0 else 0

    print("\n" + "=" * 45)
    print("OMPAR ACCURACY WITH GROUND-TRUTH LABEL")
    print("=" * 45)
    print(f"Total Loops evaluated: {total}\n")
    print(f"TP: {tp}")
    print(f"FP: {fp}")
    print(f"TN: {tn}")
    print(f"FN: {fn}")
    print("-" * 45)
    print(f"Precision: {precision * 100:.0f}%")
    print(f"Recall:    {recall * 100:.0f}%")
    print(f"Accuracy:  {accuracy * 100:.0f}%")
    print("=" * 45)

    # ---------------------------------------------------------
    # NEW: Save the results to a JSON file for later visualization
    # ---------------------------------------------------------
    results_data = {
        "Total": total,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Precision": precision,
        "Recall": recall,
        "Accuracy": accuracy
    }
    
    # Ensure the target directory exists
    os.makedirs(os.path.dirname(main_args.output_results) or '.', exist_ok=True)
    
    with open(main_args.output_results, 'w') as out_f:
        json.dump(results_data, out_f, indent=4)
        
    print(f"\nResults successfully saved to {main_args.output_results}")
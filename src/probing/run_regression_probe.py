import torch as t
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from itertools import permutations, product
from src.probing.train_probe_regression import LinearProbe

def load_probe(probe_path, input_size, device='cuda'):
    """Load a trained probe."""
    model = LinearProbe(input_size).to(device)
    model.load_state_dict(t.load(probe_path))
    model.eval()
    return model

def run_probe(probe, residuals, device='cuda'):
    """Run probe on residuals and return predictions."""
    residuals = t.tensor(residuals, dtype=t.float32).to(device)
    with t.no_grad():
        predictions = probe(residuals).squeeze()
    return predictions.cpu().numpy()

def get_data_pairs(max_nums):
    """Recreate the number pairs used in training."""
    nums = list(range(max_nums))
    pairs = sorted(list(product(nums, repeat=2)), key=lambda x: (x[0], x[1]))
    nums1, nums2 = zip(*pairs)
    return list(nums1), list(nums2)


def main():
    device = 'cuda' if t.cuda.is_available() else 'cpu'
    model_name = 'llama-3.1-8b-instruct'
    max_nums = 500
    layer = 25

    # Load residuals
    print("Loading residuals...")
    residuals = t.load(f'residuals/{model_name}_similarity_string_None_{max_nums}_layer_{layer}.pt')
    input_size = residuals.shape[-1]  # Get hidden size from residuals

    # Load probes
    print("Loading probes...")
    layer 
    string_probe = load_probe(f'probes/{model_name}_regression_string/layer_{layer}_probe.pt', input_size, device)
    int_probe = load_probe(f'probes/{model_name}_regression_int/layer_{layer}_probe.pt', input_size, device)

    # Get original number pairs
    nums1, nums2 = get_data_pairs(max_nums)

    # Run probes
    print("Running string similarity probe...")
    string_predictions = run_probe(string_probe, residuals, device)
    print("Running integer similarity probe...")
    int_predictions = run_probe(int_probe, residuals, device)

    # Create results dictionary
    results = {
        'metadata': {
            'model': model_name,
            'layer': layer,
            'max_nums': max_nums
        },
        'predictions': []
    }

    # Combine predictions with number pairs
    print("Organizing results...")
    for idx in tqdm(range(len(nums1))):
        results['predictions'].append({
            'num1': nums1[idx],
            'num2': nums2[idx],
            'string_similarity': float(string_predictions[idx]),
            'int_similarity': float(int_predictions[idx])
        })

    # Save results
    output_dir = Path('probe_results')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f'{model_name}_layer{layer}_predictions2.json'
    print(f"Saving results to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()

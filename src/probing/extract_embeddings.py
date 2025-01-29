import torch as t
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from src.probing.create_data import StrIntData, SimilarityData
from src.utils import model_paths


def collect_residuals(model, prompts, tokenizer, token_to_save_emb_on, device='cuda', batch_size=32, layers=None, save_every=100):
    residuals_list = []
    save_count = 0
    temp_saves = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Processing batches"):
        batch_prompts = prompts[i:i + batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False).to(device)
        
        with t.inference_mode():
            outputs = model(**inputs, output_hidden_states=True)
        
        hidden_states = outputs.hidden_states
        num_layers = len(layers)
        hidden_size = hidden_states[0].shape[-1]

        batch_residuals = t.zeros((len(batch_prompts), num_layers, hidden_size)).to(device)
        
        # Each hidden_state has shape (batch_size, sequence_length, hidden_size)
        for i, layer_idx in enumerate(layers):
            # For each example in the batch, get the token we want
            batch_residuals[:, i] = hidden_states[layer_idx][range(len(batch_prompts)), token_to_save_emb_on]

        residuals_list.append(batch_residuals.cpu())
        t.cuda.empty_cache()  # Free memory after each batch

        # Save every save_every batches
        if len(residuals_list) >= save_every:
            print(f"\nSaving intermediate results (part {save_count})...")
            temp_file = f'residuals/temp_residuals_{save_count}.pt'
            temp_saves.append(temp_file)
            t.save(t.cat(residuals_list), temp_file)
            residuals_list = []  # Clear the list
            save_count += 1

    # Save any remaining residuals
    if residuals_list:
        print(f"\nSaving final part {save_count}...")
        temp_file = f'residuals/temp_residuals_{save_count}.pt'
        temp_saves.append(temp_file)
        t.save(t.cat(residuals_list), temp_file)

    # Concatenate all saved parts
    print("\nConcatenating all parts...")
    final_residuals = []
    for temp_file in temp_saves:
        final_residuals.append(t.load(temp_file))
        os.remove(temp_file)  # Clean up temporary file
    
    return t.cat(final_residuals)

def save_residuals(residuals, file_name):
    print(f"\nSaving final results to {file_name}...")
    t.save(residuals, file_name)

def main():
    model_name = 'llama-3.1-8b-instruct'
    model_path = model_paths[model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=t.float16).to('cuda')
    # for format in ['none']:
    #     data = StrIntData(format=format, max_nums=999, tokenizer=tokenizer)
    #     prompts = data.return_data(tokenizer)
    #     token_ids = data.return_token_to_save_emb_on(tokenizer)
    #     name = data.return_name()
    #     residuals = [collect_residuals(model, prompt, tokenizer, token_id) for (prompt, token_id) in zip(prompts, token_ids)]
    #     os.makedirs('residuals', exist_ok=True)
    #     save_residuals(residuals, f'residuals/{model_name}_{name}.pt')

    # put in any format 
    layers = list(range(33))
    data = SimilarityData(format='string', n_comparisons=10000, max_nums=1000, seed=42, tokenizer=tokenizer)
    prompts = data.return_data()

    token_ids = data.return_token_to_save_emb_on()

    str_groundtruths = data.return_groundtruths(format='string')
    int_groundtruths = data.return_groundtruths(format='int')
    # save the groundtruths in the same folder as residuals 
    os.makedirs('groundtruths', exist_ok=True)
    
    name = data.return_name()
    t.save(str_groundtruths, f'groundtruths/{model_name}_{name}_string_groundtruths.pt')
    t.save(int_groundtruths, f'groundtruths/{model_name}_{name}_int_groundtruths.pt')
    print(f'Collecting residuals for {model_name} with {data.return_name()}')
    
    # Process all prompts in one call
    if False:
        residuals = collect_residuals(model, prompts, tokenizer, token_ids[0], layers=layers, save_every=500)
        
        os.makedirs('residuals', exist_ok=True)
        layers_str = '_'.join(map(str, layers))
        save_residuals(residuals, f'residuals/{model_name}_{name}_layer_{layers_str}.pt')

if __name__ == '__main__':
    main()
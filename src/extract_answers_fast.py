import logging
from tqdm import tqdm 
from openai import OpenAI
import pandas as pd
import os 
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from anthropic import Anthropic
import time
import random
from collections import defaultdict
load_dotenv()

from prompts import PromptDataset, generate_prompts_chat, generate_prompts_triplets
from utils import model_paths, printv, estimate_cost
from model_configs import model_configs, return_client, model_name_mappings

prompt_layout_styles = {
    'default': 'default',
    'default-claude': 'default',
    'int': 'default',
    'str': 'default',
    'corporate': 'default',
    'base-8': 'default',
    'base-2': 'default',
    'base-4': 'default',
    'base-21': 'default',
    'scientific': 'default',
    'roman': 'default',
    'number_triplets_3dig': 'triplet',
    'number_triplets_3dig_flipped': 'triplet',
    'number_triplets_5dig': 'triplet',
    'number_triplets_5dig_flipped': 'triplet',
    'number_triplets_3dig_similar': 'triplet',
    'number_triplets_3dig_flipped_similar': 'triplet',
    'number_triplets_5dig_similar': 'triplet',
    'number_triplets_5dig_flipped_similar': 'triplet',
    'levenshtein_eval': 'levenshtein',
    'medical_triplets': 'triplet',
    'medical_triplets_similar': 'triplet',
    'medical_triplets_concentration': 'triplet',
    'medical_triplets_concentration_similar': 'triplet',
    'medical_triplets_concentration_flipped': 'triplet',
    'medical_triplets_concentration_similar_flipped': 'triplet',
    'medical_triplets_concentration_5dig': 'triplet',
    'medical_triplets_concentration_similar_5dig': 'triplet',
    'medical_triplets_concentration_flipped_5dig': 'triplet',
    'medical_triplets_concentration_similar_flipped_5dig': 'triplet',
    'luminosity_triplets': 'triplet',
    'luminosity_triplets_reversed': 'triplet',
    'force_triplets': 'triplet',
    'force_triplets_reversed': 'triplet',
    'resistance_triplets': 'triplet',
    'resistance_triplets_reversed': 'triplet',
    'luminosity_triplets_5dig': 'triplet',
    'luminosity_triplets_reversed_5dig': 'triplet',
    'force_triplets_5dig': 'triplet',
    'force_triplets_reversed_5dig': 'triplet',
    'resistance_triplets_5dig': 'triplet',
    'resistance_triplets_reversed_5dig': 'triplet',
}


def process_missing_combinations(model_name, setting) -> pd.DataFrame:
    """
    Process missing combinations by applying a given function to each pair of numbers.
    
    Args:
        func: Function that takes two integers and returns a float
        
    Returns:
        DataFrame with processed missing combinations
    """
    # Read the data
    c = pd.read_csv(f'data/{model_name}/{setting}/with_dups.csv')
    c_dedup = c.drop_duplicates(subset=['0', '1'])

    # Create all possible combinations
    data = []
    for i in range(1000):
        for j in range(1000):
            data.append([i, j])
    data = pd.DataFrame(data, columns=['0', '1'])

    # Find missing combinations
    missing = data.merge(c_dedup, on=['0', '1'], how='left', indicator=True)
    missing = missing[missing['_merge'] == 'left_only'][['0', '1']]
    
    # convert c_dedup to a nested dictionary with dict[0][1] = 2
    dups = defaultdict(dict)
    for i, row in c_dedup.iterrows():
        dups[row['0']][row['1']] = row['2']
    return missing, dups

def process_cutoff_generations(model_name, setting, num_chars: int = 24):
    """
    When generating with too short of a token limit and the answer isn't included. 
    For these cases we'll simply regenerate with a longer token limit. 
    """
    # load the data
    c = pd.read_csv(f'data/{model_name}/{setting}/cutoff.csv')
    
    c_cutoff = c[c['2'].str.len() <= num_chars]

    data = []
    for i in range(1000):
        for j in range(1000):
            data.append([i, j])
    data = pd.DataFrame(data, columns=['0', '1'])

    missing = data.merge(c_cutoff, on=['0', '1'], how='left', indicator=True)
    missing = missing[missing['_merge'] == 'left_only'][['0', '1']]

    # automatically reuse the dups that are shorter than the cutoff 
    dups = defaultdict(dict)
    for i, row in c_cutoff.iterrows():
        dups[row['0']][row['1']] = row['2']

    return missing, dups


def return_params(model_provider):
    temperature = model_configs[model_provider]['temperature']
    batch_size = model_configs[model_provider]['batch_size']
    num_threads = model_configs[model_provider]['num_threads']
    return temperature, batch_size, num_threads

def save_results(answers, save_path, interim=False):
    df = pd.DataFrame(answers)
    if interim:
        interim_path = save_path.replace('.csv', '_interim.csv')
        df.to_csv(interim_path, index=False)
    else:
        df.to_csv(save_path, index=False)
        # Clean up interim file if it exists
        interim_path = save_path.replace('.csv', '_interim.csv')
        if os.path.exists(interim_path):
            os.remove(interim_path)

def load_interim_results(save_path, rerun_errors=False):
    interim_path = save_path.replace('.csv', '_interim.csv')
    if rerun_errors:
        interim_path = save_path 
    if os.path.exists(interim_path):
        df = pd.read_csv(interim_path)
        df = df[~df['2'].astype(str).str.contains('failure')]
        return df.values.tolist(), len(df)
    return [], 0

def return_prompts(n_comparisons, prompt_layout, prompt_style):
    # change the path to the triplets dynamically
    if '5dig' in prompt_layout:
        path_to_triplets = 'data/prompts/close_5dig_triplets.npy'
    else:
        path_to_triplets = 'data/prompts/triplets_close_3_digits.npy'
    if prompt_style == 'triplet':
        prompts, pairwise = generate_prompts_triplets(path_to_triplets, prompt_layout=prompt_layout)
    elif prompt_style == 'default':
        prompts, pairwise = generate_prompts_chat(n_comparisons, prompt_layout=prompt_layout)
    return prompts, pairwise

def model_generation(batch_prompts, 
                     batch_pairwise, 
                     client, 
                     model_name, 
                     temperature, 
                     provider, 
                     c_dedup=None, 
                     max_tokens=64):
    if c_dedup is not None:
        # if batch_pairwise is in c_dedup, then just return the value
        resp = c_dedup.get(batch_pairwise[0][0], {}).get(batch_pairwise[0][1], None)
        if resp is not None:
            return list(zip(batch_pairwise, [resp]*len(batch_pairwise)))
    if provider == 'anthropic':
        completion = client.messages.create(
            model=model_name_mappings[model_name],
            messages=batch_prompts,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        responses = [completion.content[0].text]
    else:
        completion = client.chat.completions.create(
            model=model_name_mappings[model_name],
            messages=batch_prompts,
            temperature=temperature,
            max_tokens=max_tokens
        )
        responses = [choice.message.content for choice in completion.choices]
    return list(zip(batch_pairwise, responses))

if __name__ == '__main__':
    rerun_errors = False
    just_run_on_missing = False 
    just_run_on_cutoff = False

    debug = False 
    overwrite = True 
    if debug: 
        n_comparisons = 10
    else: 
        n_comparisons = 1000

    # CHANGE THESE 
    # prompt_layouts = ['int', 'str']
    # prompt_layouts = ['medical_triplets_concentration_flipped_5dig','medical_triplets_concentration_similar_flipped_5dig', 'medical_triplets_concentration_5dig', 'medical_triplets_concentration_similar_5dig']
    prompt_layouts = ['luminosity_triplets', 'luminosity_triplets_reversed', 'force_triplets', 'force_triplets_reversed', 'resistance_triplets', 'resistance_triplets_reversed', 'luminosity_triplets_5dig', 'luminosity_triplets_reversed_5dig', 'force_triplets_5dig', 'force_triplets_reversed_5dig', 'resistance_triplets_5dig', 'resistance_triplets_reversed_5dig']
    # prompt_layouts = ['base-21']
    max_tokens = 128
    # prompt_layouts = ['str','int']
    # Define list of models and their providers to run
    run_models = [
        {'model_name': 'DeepSeek-V3', 'provider': 'together'},
        {'model_name': 'llama-3.1-8b-instruct-turbo', 'provider': 'together'},
        {'model_name': 'llama-3.1-70b-instruct-turbo', 'provider': 'together'},
        {'model_name': "mixtral-8x22b-instruct", 'provider': 'together'},
        {'model_name': "claude-3.5-sonnet", 'provider': 'anthropic'},
    ]

    for model_config in run_models:
        temperature, batch_size, num_threads = return_params(model_config['provider'])
        model_name = model_config['model_name']
        provider = model_config['provider']
        print(f"\nProcessing model: {model_name} with provider: {provider}")
        
        client = return_client(provider)

        for prompt_layout in prompt_layouts:

            if just_run_on_missing:
                missing, c_dedup = process_missing_combinations(model_name, prompt_layout)
            elif just_run_on_cutoff:
                missing, c_dedup = process_cutoff_generations(model_name, prompt_layout)
            else:
                missing, c_dedup = None, None
            
            print(f"Processing prompt layout: {prompt_layout}")
            prompt_style = prompt_layout_styles[prompt_layout]
                
            prompts, pairwise = return_prompts(n_comparisons, prompt_layout, prompt_style)
            
            max_compars = len(prompts)
            
            save_dir = os.path.join('data', model_name, prompt_layout)
            save_path = os.path.join(save_dir, f'{max_compars}_answers.csv')
            if os.path.exists(save_path):
                print(f"File already exists, skipping: {save_path}")
                # continue
            os.makedirs(save_dir, exist_ok=True)
            
            dataset = PromptDataset(prompts)
            # Group pro mpts into batches
            prompt_batches = [dataset[i:i+batch_size] for i in range(0, max_compars, batch_size)]
            # Group pairwise data into corresponding batches
            pairwise_batches = [pairwise[i:i+batch_size] for i in range(0, max_compars, batch_size)]
            # Check for interim results
            if not overwrite:
                all_answers, processed_count = load_interim_results(save_path, rerun_errors)
            else:
                all_answers = []
                processed_count = 0
            
            cost = estimate_cost(prompts[processed_count:], model_name, model_name_mappings)
            print(f"Estimated cost for remaining items: ${cost:.4f}")   

            if processed_count > 0:
                print(f"Resuming from interim save with {processed_count} existing results")
                # Adjust the batches to skip already processed items
                skip_batches = processed_count // batch_size
                prompt_batches = prompt_batches[skip_batches:]
                pairwise_batches = pairwise_batches[skip_batches:]
            else:
                processed_count = 0
                all_answers = []

            def process_batch(batch_idx, batch_prompts, batch_pairwise):
                max_retries = 5
                base_delay = 1  # Base delay in seconds
                
                for attempt in range(max_retries):
                    try:
                        # running th emodel generation 
                        if just_run_on_missing or just_run_on_cutoff:
                            batch_results = model_generation(batch_prompts, batch_pairwise, client, model_name, temperature, provider, c_dedup, max_tokens)
                        else:
                            batch_results = model_generation(batch_prompts, batch_pairwise, client, model_name, temperature, provider, c_dedup, max_tokens)
                        return batch_results
                    
                    except Exception as e:
                        error_str = str(e).lower()
                        is_rate_limit = any(phrase in error_str for phrase in 
                                          ['too many requests', 'rate limit', '429', 'capacity'])
                        
                        if is_rate_limit and attempt < max_retries - 1:
                            delay = (base_delay * (2 ** attempt)) + (random.random() * 0.1)
                            print(f"Rate limit hit on batch {batch_idx}, attempt {attempt + 1}/{max_retries}. "
                                         f"Retrying in {delay:.2f} seconds...")
                            time.sleep(delay)
                            continue
                        else:
                            print(f"Error on batch {batch_idx}, attempt {attempt + 1}: {str(e)}")
                            raise e
                            return [(pw, f'failure - {e}') for pw in batch_pairwise]
                
                return [(pw, f'failure - max retries exceeded') for pw in batch_pairwise]

            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(process_batch, idx, batch_prompts, batch_pw) 
                          for idx, (batch_prompts, batch_pw) in enumerate(zip(prompt_batches, pairwise_batches))]

                with tqdm(total=len(prompt_batches), desc=f"Processing {model_name}") as pbar:
                    for future_idx, future in enumerate(futures):
                        try:
                            batch_results = future.result()
                            for pw, result in batch_results:
                                if prompt_style == 'triplet':
                                    d1, d2, d3 = pw[0], pw[1], pw[2]
                                    all_answers.append([d1, d2, d3, result])
                                else:
                                    d1, d2 = pw[0], pw[1]
                                    all_answers.append([d1, d2, result])
                                processed_count += 1
                                if processed_count % 100000 == 0:
                                    save_results(all_answers, save_path, interim=True)
                                    print(f"\nSaved interim results at {processed_count} steps")
                            pbar.update(1)
                        except Exception as e:
                            print(f"Error processing future {future_idx+1}: {str(e)}")        
            
            # Save final results
            save_results(all_answers, save_path)

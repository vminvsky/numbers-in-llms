import string
import pandas as pd 
from Levenshtein import distance

import matplotlib.pyplot as plt 
import seaborn as sns
from tqdm import tqdm


def estimate_cost(prompts, model_name, model_name_mappings):
    """
    Estimate the cost of running prompts through a model using tiktoken.
    
    Args:
        prompts (list): List of prompt strings
        model_name (str): Name of the model to use
        
    Returns:
        float: Estimated cost in USD
    """
    import tiktoken

    # Map model names to encoding and cost per token
    model_costs = {
        'meta-llama/Meta-Llama-3.1-8B-Instruct': ('cl100k_base', 0.00000012),
        'meta-llama/Meta-Llama-3.1-70B-Instruct': ('cl100k_base', 0.000015),
        'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo': ('cl100k_base', 0.000002),
        'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo': ('cl100k_base', 0.0000015),
        'meta-llama/Llama-3.3-70B-Instruct-Turbo': ('cl100k_base', 0.0000015),
        'google/gemma-2-27b-it': ('cl100k_base', 0.000006),
        'deepseek-ai/DeepSeek-V3': ('cl100k_base', 0.00000125),
        'claude-3-5-sonnet-20241022': ('cl100k_base', 0.00000125),
        'mistralai/Mixtral-8x22B-Instruct-v0.1': ('cl100k_base', 0.00000125),
    }

    full_model_name = model_name_mappings[model_name]

    encoding_name, cost_per_token = model_costs[full_model_name]
    
    encoding = tiktoken.get_encoding(encoding_name)

    total_tokens = 0
    for prompt in prompts:
        tokens = len(encoding.encode(prompt['content']))
        total_tokens += tokens

    # Add 20% for response tokens
    total_tokens = int(total_tokens * 1.2)
    estimated_cost = total_tokens * cost_per_token

    print(f"Estimated tokens: {total_tokens}")
    print(f"Estimated cost: ${estimated_cost:.4f}")
    
    return estimated_cost

model_paths = {
        'gpt2': '/scratch/gpfs/vv7118/models/hub/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e',
        'llama-3.1-8b': '/scratch/gpfs/vv7118/models/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/48d6d0fc4e02fb1269b36940650a1b7233035cbb',
        'llama-3.1-8b-instruct': '/scratch/gpfs/vv7118/models/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/5206a32e0bd3067aef1ce90f5528ade7d866253f',
        'llama-3-8b': '/scratch/gpfs/vv7118/models/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/62bd457b6fe961a42a631306577e622c83876cb6',
        'llama-3-8b-sae': '/scratch/gpfs/vv7118/models/hub/models--EleutherAI--sae-llama-3-8b-32x/snapshots/32926540825db694b6228df703f4528df4793d67',
}

def return_pairwise_combinations(n):
    return [(i, j) for i in range(n) for j in range(n)]

def convert_to_df(list_of_tuples):
    df = pd.DataFrame(list_of_tuples, columns=['n1','n2','num1', 'num2', 'similarity'])
    df = df[['n1','n2','similarity']]
    df[['n1','n2']] = df[['n1','n2']].astype(int)
    df = df.pivot(index='n1', columns='n2', values='similarity')
    return df

def return_pairwise_combinations_full(n, setting='default'):
    list_of_pairs = [(i, j) for i in range(n) for j in range(n)]
    if setting == 'str':
        return list_of_pairs, list_of_pairs
    elif setting == 'default':
        return list_of_pairs, list_of_pairs
    elif setting == 'int':
        return list_of_pairs, list_of_pairs
    elif setting == 'roman':
        return [(convert_num_to_roman_numerals(i), convert_num_to_roman_numerals(j)) for i, j in list_of_pairs], list_of_pairs
    elif setting == 'scientific':
        return [(convert_to_scientific(i), convert_to_scientific(j)) for i, j in list_of_pairs], list_of_pairs
    elif 'base' in setting:
        base = int(setting.split('-')[-1])
        return [(int2base(i, base), int2base(j, base)) for i, j in list_of_pairs], list_of_pairs
    elif 'corporate' in setting:
        return list_of_pairs, list_of_pairs
    else:
        raise NotImplementedError(f'setting -- `{setting}` not implemented')

def convert_num_to_roman_numerals(num):
    # note this only works for numbers between 1 and 3999
    if num == 0:
        return 'nulla'
    m = ["", "M", "MM", "MMM"]
    c = ["", "C", "CC", "CCC", "CD", "D",
         "DC", "DCC", "DCCC", "CM "]
    x = ["", "X", "XX", "XXX", "XL", "L",
         "LX", "LXX", "LXXX", "XC"]
    i = ["", "I", "II", "III", "IV", "V",
         "VI", "VII", "VIII", "IX"]
    
    thousands = m[num // 1000]
    hundereds = c[(num % 1000) // 100]
    tens = x[(num % 100) // 10]
    ones = i[num % 10]

    ans = (thousands + hundereds + tens + ones)
    return ans


def convert_to_scientific(num):
    return f"{num:.6e}"

def create_levenshtein_matrix(setting):
    pairs, base_pairs = return_pairwise_combinations_full(1000, setting)
    
    max_elt_len = 0
    for p1, p2 in pairs:
        string_form_p1 = str(p1)
        if len(string_form_p1) > max_elt_len:
            max_elt_len = len(string_form_p1)

    base_sims = []
    for (n1, n2), (d1, d2) in zip(base_pairs, pairs):
        if isinstance(d1, int):
            d1 = str(d1)
            d2 = str(d2)
        dist = 1 - distance(d1, d2) / max_elt_len
        base_sims.append((n1,n2, d1, d2, dist))
    return convert_to_df(base_sims)


def printv(*args, **kwargs):
    if 'verbose' in kwargs and kwargs['verbose']:
        print(*args)
    else:
        pass


digs = string.digits + string.ascii_letters

def int2base(x, base):
    if x < 0:
        sign = -1
    elif x == 0:
        return digs[0]
    else:
        sign = 1

    x *= sign
    digits = []

    while x:
        digits.append(digs[x % base])
        x = x // base

    if sign < 0:
        digits.append('-')

    digits.reverse()

    return ''.join(digits)
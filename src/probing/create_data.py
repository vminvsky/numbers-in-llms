import os
import pickle
from collections import defaultdict
import numpy as np
from Levenshtein import distance
from transformers import AutoTokenizer
from dataclasses import dataclass, field
import random 
from functools import lru_cache
from itertools import product
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from itertools import islice
from typing import List, Tuple, Dict
from joblib import Parallel, delayed  # <-- Import from joblib



@dataclass
class AbstractData:

    template_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

How similar are the two numbers on a scale of 0 (completely dissimilar) to 1 (completely similar)? Respond only with the rating.
Number: XXX
Number: YYY<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Rating:"""
    
    @staticmethod
    def _prepare_text_static(text: str, assistant_start: str = None) -> tuple:
        """Return a *tuple* of messages to leverage caching without string eval."""
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': text}
        ]
        if assistant_start is not None:
            messages.append({'role': 'assistant', 'content': assistant_start})
        return tuple(messages)

    def _prepare_input_to_model(self, text: str, assistant_start: str = None) -> str:
        """Process the text and apply tokenizer template."""
        # Retrieve a *tuple*, convert to list
        messages = list(self._prepare_text_static(text, assistant_start))
        
        if assistant_start is not None:
            tokens = self.tokenizer.apply_chat_template(
                messages, tokenize=False, continue_final_message=True, add_special_tokens=False
            )
        else:
            tokens = self.tokenizer.apply_chat_template(messages, tokenize=False, add_special_tokens=False)
        
        # Possibly skip or reduce repeated replacements
        tokens = tokens.replace('Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024', '')
        return tokens


@dataclass 
class StrIntData(AbstractData):
    format: str
    max_nums: int 

    @staticmethod
    def _get_token_position(text: str, tokens: str, num: int) -> int:
        """Get the position of a number token in the tokenized text."""
        tokens_list = tokens.split()
        for i, token in enumerate(tokens_list):
            if str(num) == token:
                return i
        return -1  # Return -1 if not found

    def return_data(self):
        numbs = list(range(1, self.max_nums + 1))
        prompts = []
        print("Generating prompts for StrIntData...")
        for num in tqdm(numbs, desc="Processing numbers"):
            text = self._return_text(num)
            text = self._prepare_input_to_model(text)
            prompts.append(text)
        return prompts
    
    def _return_text(self, num):
        return f'str({num})' if self.format == 'string' else f'int({num})' if self.format == 'int' else f'{num}'

    def return_name(self):
        if self.format == 'string':
            return f'str_{self.max_nums}'
        elif self.format == 'int':
            return f'int_{self.max_nums}'
        else:
            return f'{self.max_nums}'
    
    def return_token_to_save_emb_on(self):
        """Returns the token position of the actual number in the string."""
        token_positions = []
        print("Finding token positions...")
        for num in tqdm(range(1, self.max_nums + 1), desc="Processing numbers"):
            text = self._return_text(num)
            tokens = self._prepare_input_to_model(text)
            pos = self._get_token_position(text, tokens, num)
            if pos != -1:
                token_positions.append(pos)
        return tuple(token_positions)  # Convert to tuple for hashability

def log_linear_distance(num1, num2, eps_reg=0.0001):
    return np.abs(np.log(num1 + eps_reg) - np.log(num2 + eps_reg))

def log_linear_similarity(num1, num2, eps_reg=0.0001):
    return np.exp(-log_linear_distance(num1, num2, eps_reg))

def create_dist_matrices(num1, num2, max_nums, use_log_linear_sim=True):
    data_l = defaultdict(dict)
    data_e = defaultdict(dict)

    # Create a set of pairs to avoid duplicate calculations
    pairs = set()
    for i, j in zip(num1, num2):
        # Add pair in sorted order to avoid duplicates
        pairs.add(tuple(sorted([i, j])))

    # Calculate distances once per unique pair
    for i, j in pairs:
        l_dist = 1-distance(str(i), str(j)) / max(len(str(i)), len(str(j)))
        euc_dist = 1 - np.sqrt((i - j) ** 2) / max_nums
        log_linear_dist = log_linear_similarity(i, j)
        # Store distance both ways since symmetric
        data_l[i][j] = data_l[j][i] = l_dist
        if use_log_linear_sim:
            data_e[i][j] = data_e[j][i] = log_linear_dist
        else:
            data_e[i][j] = data_e[j][i] = euc_dist

    return data_l, data_e

@dataclass
class SimilarityData(AbstractData):
    format: str
    tokenizer: AutoTokenizer
    n_comparisons: int = 20
    max_nums: int = 1000
    seed: int = 42
    n_processes: int = 30  # New parameter for controlling parallelization
    nums1: list[int] = None
    nums2: list[int] = None
    prompts: list[str] = field(default_factory=list)
    _cache_dir: str = field(default='cache/similarity_data', init=False)
    use_log_linear_sim: bool = True
    overwrite: bool = True 

    def __post_init__(self):
        os.makedirs(self._cache_dir, exist_ok=True)
        if not self.overwrite:
            self._try_load_cache()
            return
        if self.n_processes is None:
            self.n_processes = cpu_count()
        

        random.seed(self.seed)
        np.random.seed(self.seed)
        if self.n_comparisons is None:
            self.nums1, self.nums2 = self.split_permutations(list(range(self.max_nums)))
            print(self.nums1[0], self.nums2[0])
        else:
            self.nums1 = np.random.choice(range(self.max_nums), self.n_comparisons, replace=True)
            self.nums2 = np.random.choice(range(self.max_nums), self.n_comparisons, replace=True)
        print('getting dist matrices')
        self.data_l, self.data_e = create_dist_matrices(
            self.nums1, self.nums2, self.max_nums, use_log_linear_sim=self.use_log_linear_sim
        )
        print('got dist matrices')
    def _get_cache_path(self):
        return os.path.join(
            self._cache_dir,
            f'data_{self.format}_{self.n_comparisons}_{self.max_nums}_{self.seed}.pkl'
        )

    def _try_load_cache(self):
        cache_path = self._get_cache_path()
        if os.path.exists(cache_path):
            print(f"Loading cached data from {cache_path}")
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                self.nums1 = cached_data['nums1']
                self.nums2 = cached_data['nums2']
                self.data_l = cached_data['data_l']
                self.data_e = cached_data['data_e']
                self.prompts = cached_data['prompts']
            return True
        return False

    def _save_to_cache(self):
        """Save current data to cache."""
        cache_path = self._get_cache_path()
        print(f"Saving data to cache: {cache_path}")
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'nums1': self.nums1,
                'nums2': self.nums2,
                'data_l': self.data_l,
                'data_e': self.data_e,
                'prompts': self.prompts,
                'use_log_linear_sim': self.use_log_linear_sim
            }, f)

    def return_data(self):
        if self.prompts:
            return self.prompts
        else:
            print('No cached prompts, generating new ones')

        self.prompts = [self.template_prompt.replace('XXX', str(i)).replace('YYY', str(j)) for i, j in tqdm(zip(self.nums1, self.nums2), total=len(self.nums1))]
        self._save_to_cache()
        return self.prompts
    
    @staticmethod
    def split_permutations(lst):
        # Create sorted pairs in order (0,0), (0,1), (0,2), ..., (1,0), (1,1), ...
        pairs = sorted(list(product(lst, repeat=2)), key=lambda x: (x[0], x[1]))
        nums1, nums2 = zip(*pairs)
        return list(nums1), list(nums2)
        
    def return_groundtruths(self, format=None):
        format = format or self.format
        data_source = self.data_l if format == 'string' else self.data_e
        return [data_source[i][j] for i, j in zip(self.nums1, self.nums2)]

    def generate_prompt(self, i, j):
        prompt = f"""How similar are the two numbers on a scale of 0 (completely dissimilar) to 1 (completely similar)? Respond only with the rating."""
        prompt += f'''\nNumber: {i}\nNumber: {j}'''
        assistant_start = f'''Rating:'''
        return prompt, assistant_start
    
    def return_name(self):
        return f'similarity_{self.format}_{self.n_comparisons}_{self.max_nums}'
    
    def return_token_to_save_emb_on(self):
        # we will take the final token of the prompt
        # assume that it encodes all the informaiton 
        return [-1]

# test 
if __name__ == '__main__':
    model_name = 'meta-llama/Llama-3.1-8B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # data = StrIntData(format='string', max_nums=20, tokenizer=tokenizer)
    # print(data.return_data())
    # print([tokenizer.tokenize(k) for k in data.return_data()])
    # print(data.return_token_to_save_emb_on(tokenizer))
    # data_l, data_e = create_dist_matrices(1000)
    # save the dictionaries

    data = SimilarityData(format='string', n_comparisons=5000, max_nums=1000, seed=42, tokenizer=tokenizer)
    print(data.return_data())
    print(data.return_groundtruths())
    print(data.return_token_to_save_emb_on())
    print(data.nums1)
    print(data.nums2)
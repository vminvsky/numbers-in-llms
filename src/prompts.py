import numpy as np 
from torch.utils.data import DataLoader, Dataset

from utils import int2base, return_pairwise_combinations, convert_num_to_roman_numerals, convert_to_scientific



labels = list(range(1000))


class PromptDataset(Dataset):
    def __init__(self, prompts):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]

def return_pairwise_combinations(n, upper_triangular=False):
    if upper_triangular:
        return [(i, j) for i in range(n) for j in range(i+1, n)]
    return [(i, j) for i in range(n) for j in range(n)]


labels = list(range(1000))

def medical_triplets(current_pair):
    prompt = f'''You require approximately {current_pair[0]}mg of a drug. Two test tubes are available: one containing {current_pair[1]}mg and the other {current_pair[2]}mg. Your task is to determine which test tube provides the closest amount to your required dosage. Which one will you choose? Respond only with the mg value of the test tube you choose.'''
    return prompt 

def medical_triplets_similar(current_pair):
    prompt = f'''You require approximately {current_pair[0]}mg of a drug. Two test tubes are available: one containing {current_pair[1]}mg and the other {current_pair[2]}mg. Your task is to determine which test tube provides the most similar amount to your required dosage. Which one will you choose? Respond only with the mg value of the test tube you choose.'''
    return prompt 


def medical_triplets_concentration(current_pair):
    prompt = f'''You require a compound with a concentration of approximately {current_pair[0]} ppm. Two test tubes are available: one containing {current_pair[1]} ppm and the other {current_pair[2]} ppm. Your task is to determine which test tube provides the closest concentration to your required dosage. Which one will you choose? Respond only with the ppm value of the test tube you choose.'''
    return prompt 

def medical_triplets_concentration_similar(current_pair):
    prompt = f'''You require a compound with a concentration of approximately {current_pair[0]} ppm. Two test tubes are available: one containing {current_pair[1]} ppm and the other {current_pair[2]} ppm. Your task is to determine which test tube provides the most similar concentration to your required dosage. Which one will you choose? Respond only with the ppm value of the test tube you choose.'''
    return prompt 


def medical_triplets_concentration_flipped(current_pair):
    prompt = f'''You require a compound with a concentration of approximately {current_pair[0]} ppm. Two test tubes are available: one containing {current_pair[2]} ppm and the other {current_pair[1]} ppm. Your task is to determine which test tube provides the closest concentration to your required dosage. Which one will you choose? Respond only with the ppm value of the test tube you choose.'''
    return prompt 

def medical_triplets_concentration_similar_flipped(current_pair):
    prompt = f'''You require a compound with a concentration of approximately {current_pair[0]} ppm. Two test tubes are available: one containing {current_pair[2]} ppm and the other {current_pair[1]} ppm. Your task is to determine which test tube provides the most similar concentration to your required dosage. Which one will you choose? Respond only with the ppm value of the test tube you choose.'''
    return prompt 

def medical_triplets_concentration_5dig(current_pair):
    prompt = f'''You require a compound with a concentration of approximately {current_pair[0]} ppm. Two test tubes are available: one containing {current_pair[1]} ppm and the other {current_pair[2]} ppm. Your task is to determine which test tube provides the closest concentration to your required dosage. Which one will you choose? Respond only with the ppm value of the test tube you choose.'''
    return prompt 

def medical_triplets_concentration_similar_5dig(current_pair):
    prompt = f'''You require a compound with a concentration of approximately {current_pair[0]} ppm. Two test tubes are available: one containing {current_pair[1]} ppm and the other {current_pair[2]} ppm. Your task is to determine which test tube provides the most similar concentration to your required dosage. Which one will you choose? Respond only with the ppm value of the test tube you choose.'''
    return prompt 


def medical_triplets_concentration_flipped_5dig(current_pair):
    prompt = f'''You require a compound with a concentration of approximately {current_pair[0]} ppm. Two test tubes are available: one containing {current_pair[2]} ppm and the other {current_pair[1]} ppm. Your task is to determine which test tube provides the closest concentration to your required dosage. Which one will you choose? Respond only with the ppm value of the test tube you choose.'''
    return prompt 

def medical_triplets_concentration_similar_flipped_5dig(current_pair):
    prompt = f'''You require a compound with a concentration of approximately {current_pair[0]} ppm. Two test tubes are available: one containing {current_pair[2]} ppm and the other {current_pair[1]} ppm. Your task is to determine which test tube provides the most similar concentration to your required dosage. Which one will you choose? Respond only with the ppm value of the test tube you choose.'''
    return prompt 

def luminosity_triplets(current_pair):
    prompt = f'''You require a workspace illumination of approximately {current_pair[0]} lux. Two lamps are available: one producing {current_pair[1]} lux and the other {current_pair[2]} lux. Your task is to determine which lamp provides the most similar intensity to your requirement. Which one will you choose? Respond only with the lux value of the lamp you choose.'''
    return prompt

def luminosity_triplets_reversed(current_pair):
    prompt = f'''You require a workspace illumination of approximately {current_pair[0]} lux. Two lamps are available: one producing {current_pair[2]} lux and the other {current_pair[1]} lux. Your task is to determine which lamp provides the most similar intensity to your requirement. Which one will you choose? Respond only with the lux value of the lamp you choose.'''
    return prompt

def force_triplets(current_pair):
    prompt = f'''You require a force of approximately {current_pair[0]} N. Two objects are available: one with a mass of {current_pair[1]} kg and the other {current_pair[2]} kg. Your task is to determine which object provides the most similar force to your requirement. Which one will you choose? Respond only with the mass value of the object you choose.'''
    return prompt

def force_triplets_reversed(current_pair):
    prompt = f'''You require a force of approximately {current_pair[0]} N. Two objects are available: one with a mass of {current_pair[2]} kg and the other {current_pair[1]} kg. Your task is to determine which object provides the most similar force to your requirement. Which one will you choose? Respond only with the mass value of the object you choose.'''
    return prompt

def resistance_triplets(current_pair):
    prompt = f'''You require a resistance of approximately {current_pair[0]} ohms. Two resistors are available: one with a resistance of {current_pair[1]} ohms and the other {current_pair[2]} ohms. Your task is to determine which resistor provides the most similar resistance to your requirement. Which one will you choose? Respond only with the resistance value of the resistor you choose.'''
    return prompt

def resistance_triplets_reversed(current_pair):
    prompt = f'''You require a resistance of approximately {current_pair[0]} ohms. Two resistors are available: one with a resistance of {current_pair[2]} ohms and the other {current_pair[1]} ohms. Your task is to determine which resistor provides the most similar resistance to your requirement. Which one will you choose? Respond only with the resistance value of the resistor you choose.'''
    return prompt


def luminosity_triplets_5dig(current_pair):
    prompt = f'''You require a workspace illumination of approximately {current_pair[0]} lux. Two lamps are available: one producing {current_pair[1]} lux and the other {current_pair[2]} lux. Your task is to determine which lamp provides the most similar intensity to your requirement. Which one will you choose? Respond only with the lux value of the lamp you choose.'''
    return prompt

def luminosity_triplets_reversed_5dig(current_pair):
    prompt = f'''You require a workspace illumination of approximately {current_pair[0]} lux. Two lamps are available: one producing {current_pair[2]} lux and the other {current_pair[1]} lux. Your task is to determine which lamp provides the most similar intensity to your requirement. Which one will you choose? Respond only with the lux value of the lamp you choose.'''
    return prompt

def force_triplets_5dig(current_pair):
    prompt = f'''You require a force of approximately {current_pair[0]} N. Two objects are available: one with a mass of {current_pair[1]} kg and the other {current_pair[2]} kg. Your task is to determine which object provides the most similar force to your requirement. Which one will you choose? Respond only with the mass value of the object you choose.'''
    return prompt

def force_triplets_reversed_5dig(current_pair):
    prompt = f'''You require a force of approximately {current_pair[0]} N. Two objects are available: one with a mass of {current_pair[2]} kg and the other {current_pair[1]} kg. Your task is to determine which object provides the most similar force to your requirement. Which one will you choose? Respond only with the mass value of the object you choose.'''
    return prompt

def resistance_triplets_5dig(current_pair):
    prompt = f'''You require a resistance of approximately {current_pair[0]} ohms. Two resistors are available: one with a resistance of {current_pair[1]} ohms and the other {current_pair[2]} ohms. Your task is to determine which resistor provides the most similar resistance to your requirement. Which one will you choose? Respond only with the resistance value of the resistor you choose.'''
    return prompt

def resistance_triplets_reversed_5dig(current_pair):
    prompt = f'''You require a resistance of approximately {current_pair[0]} ohms. Two resistors are available: one with a resistance of {current_pair[2]} ohms and the other {current_pair[1]} ohms. Your task is to determine which resistor provides the most similar resistance to your requirement. Which one will you choose? Respond only with the resistance value of the resistor you choose.'''
    return prompt

def number_triplets_3dig(current_pair):
    prompt = f'''Is the number {current_pair[0]} closer to {current_pair[1]} or {current_pair[2]}? Respond only with the option you think is closer and no other text.'''#generate_pair(current_pair, show_rating=False)
    return prompt

def number_triplets_3dig_flipped(current_pair):
    prompt = f'''Is the number {current_pair[0]} closer to {current_pair[2]} or {current_pair[1]}? Respond only with the option you think is closer and no other text.'''#generate_pair(current_pair, show_rating=False)
    return prompt

def number_triplets_5dig(current_pair):
    prompt = f'''Is the number {current_pair[0]} closer to {current_pair[1]} or {current_pair[2]}? Respond only with the option you think is closer and no other text.'''#generate_pair(current_pair, show_rating=False)
    return prompt

def number_triplets_5dig_flipped(current_pair):
    prompt = f'''Is the number {current_pair[0]} closer to {current_pair[2]} or {current_pair[1]}? Respond only with the option you think is closer and no other text.'''#generate_pair(current_pair, show_rating=False)
    return prompt

def number_triplets_3dig_similar(current_pair):
    prompt = f'''Is the number {current_pair[0]} more similar to {current_pair[1]} or {current_pair[2]}? Respond only with the option you think is closer and no other text.'''#generate_pair(current_pair, show_rating=False)
    return prompt

def number_triplets_3dig_flipped_similar(current_pair):
    prompt = f'''Is the number {current_pair[0]} more similar to {current_pair[2]} or {current_pair[1]}? Respond only with the option you think is closer and no other text.'''#generate_pair(current_pair, show_rating=False)
    return prompt

def number_triplets_5dig_similar(current_pair):
    prompt = f'''Is the number {current_pair[0]} more similar to {current_pair[1]} or {current_pair[2]}? Respond only with the option you think is closer and no other text.'''#generate_pair(current_pair, show_rating=False)
    return prompt

def number_triplets_5dig_flipped_similar(current_pair):
    prompt = f'''Is the number {current_pair[0]} more similar to {current_pair[2]} or {current_pair[1]}? Respond only with the option you think is closer and no other text.'''#generate_pair(current_pair, show_rating=False)
    return prompt


def generate_prompt(current_pair):
    prompt = f"""How similar are the two numbers on a scale of 0 (completely dissimilar) to 1 (completely similar)? Respond only with the rating."""
    prompt += f'''\nNumber: {labels[current_pair[0]]}\nNumber: {labels[current_pair[1]]}\nRating:'''
    return prompt

def generate_prompt_claude(current_pair):
    prompt = f"""How similar are the two numbers on a scale of 0 (completely dissimilar) to 100 (completely similar)? Respond only with the rating."""
    prompt += f'''\nNumber: {labels[current_pair[0]]}\nNumber: {labels[current_pair[1]]}\nRating:'''
    return prompt


def generate_prompt_int(current_pair):
    prompt = f"""How similar are the two numbers on a scale of 0 (completely dissimilar) to 1 (completely similar)? Respond only with the rating."""
    prompt += f'''\nNumber: int({labels[current_pair[0]]})\nNumber: int({labels[current_pair[1]]})\nRating:'''
    return prompt

def generate_prompt_scientific(current_pair):
    prompt = f"""How similar are the two numbers on a scale of 0 (completely dissimilar) to 1 (completely similar)? Respond only with the rating."""
    prompt += f'''\nNumber: {convert_to_scientific(labels[current_pair[0]])}\nNumber: {convert_to_scientific(labels[current_pair[1]])}\nRating:'''
    return prompt

def generate_prompt_roman(current_pair):
    prompt = f"""How similar are the two Roman Numeral numbers on a scale of 0 (completely dissimilar) to 1 (completely similar)? Respond only with the rating."""
    prompt += f'''\nNumber: {convert_num_to_roman_numerals(labels[current_pair[0]])}\nNumber: {convert_num_to_roman_numerals(labels[current_pair[1]])}\nRating:'''
    return prompt

def generate_prompt_str(current_pair):
    prompt = f"""How similar are the two numbers on a scale of 0 (completely dissimilar) to 1 (completely similar)? Respond only with the rating."""
    prompt += f'''\nNumber: str({labels[current_pair[0]]})\nNumber: str({labels[current_pair[1]]})\nRating:'''
    return prompt

def generate_prompt_base(current_pair, base: int):
    num1 = labels[current_pair[0]]
    num2 = labels[current_pair[1]]
    base_num1 = int2base(num1, base)
    base_num2 = int2base(num2, base)
    prompt = f"""How similar are the two numbers in base {base} on a scale of 0 (completely dissimilar) to 1 (completely similar)? Respond only with the rating."""
    prompt += f'''\nBase {base} number: {base_num1}\nBase {base} number: {base_num2}\nRating:'''
    return prompt

def generate_prompt_corporate(current_pair):
    prompt = f"How close are the earnings between the following two company reports on a scale of 0 (maximally distant) to 1 (identical)? Respond only with the rating." 
    prompt += f"\nCompany A: ${labels[current_pair[0]]}.00\nCompany B: ${labels[current_pair[1]]}.00\nRating:"
    return prompt 

def simple_corporate(index):
    return f"Company earnings: ${index}"

def simple_corporate_int(index):
    return f"Company earnings: int(${index})"

def simple_corporate_str(index):
    return f"Company earnings: str(${index})"

def simple_recommender(index):
    return f"Based on your usage, you've spent {index} hours on fitness activities this month. Would you like tips to optimize your workout?"


def generate_prompts(n):
    return [(generate_prompt(pair), pair) for pair in return_pairwise_combinations(n)]

def generate_prompts_embed(n, prompt_layout: str = 'corporate', use_zips: bool = True):
    zipcode_subset = np.load('data/zipcode_subset.npy', allow_pickle=True)
    range_ = list(range(n)) if not use_zips else zipcode_subset
    if prompt_layout == 'corporate':
        return [simple_corporate(int(i)) for i in range_], range_
    elif prompt_layout == 'recommender':    
        return [simple_recommender(int(i)) for i in range_], range_
    elif prompt_layout == 'corporate-int':
        return [simple_corporate_int(int(i)) for i in range_], range_
    elif prompt_layout == 'corporate-str':
        return [simple_corporate_str(int(i)) for i in range_], range_
    else:
        raise ValueError('Prompt layout not recognized - please use "corporate"')

def generate_prompts_triplets(triplets_path, prompt_layout: str = 'number_triplets_3dig'):
    triplets = np.load(triplets_path)
    
    if prompt_layout == 'number_triplets_3dig':
        return [{"role": "user", "content": f"{number_triplets_3dig(triplet)}"} for triplet in triplets], triplets
    elif prompt_layout == 'number_triplets_3dig_flipped':
        return [{"role": "user", "content": f"{number_triplets_3dig_flipped(triplet)}"} for triplet in triplets], triplets
    elif prompt_layout == 'number_triplets_5dig':
        return [{"role": "user", "content": f"{number_triplets_5dig(triplet)}"} for triplet in triplets], triplets
    elif prompt_layout == 'number_triplets_5dig_flipped':
        return [{"role": "user", "content": f"{number_triplets_5dig_flipped(triplet)}"} for triplet in triplets], triplets
    elif prompt_layout == 'number_triplets_3dig_similar':
        return [{"role": "user", "content": f"{number_triplets_3dig_similar(triplet)}"} for triplet in triplets], triplets
    elif prompt_layout == 'number_triplets_3dig_flipped_similar':
        return [{"role": "user", "content": f"{number_triplets_3dig_flipped_similar(triplet)}"} for triplet in triplets], triplets
    elif prompt_layout == 'number_triplets_5dig_similar':
        return [{"role": "user", "content": f"{number_triplets_5dig_similar(triplet)}"} for triplet in triplets], triplets
    elif prompt_layout == 'number_triplets_5dig_flipped_similar':
        return [{"role": "user", "content": f"{number_triplets_5dig_flipped_similar(triplet)}"} for triplet in triplets], triplets
    elif prompt_layout == 'medical_triplets':
        return [{"role": "user", "content": f"{medical_triplets(triplet)}"} for triplet in triplets], triplets
    elif prompt_layout == 'medical_triplets_similar':
        return [{"role": "user", "content": f"{medical_triplets_similar(triplet)}"} for triplet in triplets], triplets
    elif prompt_layout == 'medical_triplets_concentration':
        return [{"role": "user", "content": f"{medical_triplets_concentration(triplet)}"} for triplet in triplets], triplets
    elif prompt_layout == 'medical_triplets_concentration_similar':
        return [{"role": "user", "content": f"{medical_triplets_concentration_similar(triplet)}"} for triplet in triplets], triplets
    elif prompt_layout == 'medical_triplets_concentration_flipped':
        return [{"role": "user", "content": f"{medical_triplets_concentration_flipped(triplet)}"} for triplet in triplets], triplets
    elif prompt_layout == 'medical_triplets_concentration_similar_flipped':
        return [{"role": "user", "content": f"{medical_triplets_concentration_similar_flipped(triplet)}"} for triplet in triplets], triplets
    elif prompt_layout == 'medical_triplets_concentration_5dig':
        return [{"role": "user", "content": f"{medical_triplets_concentration_5dig(triplet)}"} for triplet in triplets], triplets
    elif prompt_layout == 'medical_triplets_concentration_similar_5dig':
        return [{"role": "user", "content": f"{medical_triplets_concentration_similar_5dig(triplet)}"} for triplet in triplets], triplets
    elif prompt_layout == 'medical_triplets_concentration_flipped_5dig':
        return [{"role": "user", "content": f"{medical_triplets_concentration_flipped_5dig(triplet)}"} for triplet in triplets], triplets
    elif prompt_layout == 'medical_triplets_concentration_similar_flipped_5dig':
        return [{"role": "user", "content": f"{medical_triplets_concentration_similar_flipped_5dig(triplet)}"} for triplet in triplets], triplets
    elif prompt_layout == 'luminosity_triplets':
        return [{"role": "user", "content": f"{luminosity_triplets(triplet)}"} for triplet in triplets], triplets
    elif prompt_layout == 'luminosity_triplets_reversed':
        return [{"role": "user", "content": f"{luminosity_triplets_reversed(triplet)}"} for triplet in triplets], triplets
    elif prompt_layout == 'force_triplets':
        return [{"role": "user", "content": f"{force_triplets(triplet)}"} for triplet in triplets], triplets
    elif prompt_layout == 'force_triplets_reversed':
        return [{"role": "user", "content": f"{force_triplets_reversed(triplet)}"} for triplet in triplets], triplets
    elif prompt_layout == 'resistance_triplets':
        return [{"role": "user", "content": f"{resistance_triplets(triplet)}"} for triplet in triplets], triplets
    elif prompt_layout == 'resistance_triplets_reversed':
        return [{"role": "user", "content": f"{resistance_triplets_reversed(triplet)}"} for triplet in triplets], triplets
    elif prompt_layout == 'luminosity_triplets_5dig':
        return [{"role": "user", "content": f"{luminosity_triplets_5dig(triplet)}"} for triplet in triplets], triplets
    elif prompt_layout == 'luminosity_triplets_reversed_5dig':
        return [{"role": "user", "content": f"{luminosity_triplets_reversed_5dig(triplet)}"} for triplet in triplets], triplets
    elif prompt_layout == 'force_triplets_5dig':
        return [{"role": "user", "content": f"{force_triplets_5dig(triplet)}"} for triplet in triplets], triplets
    elif prompt_layout == 'force_triplets_reversed_5dig':
        return [{"role": "user", "content": f"{force_triplets_reversed_5dig(triplet)}"} for triplet in triplets], triplets
    elif prompt_layout == 'resistance_triplets_5dig':
        return [{"role": "user", "content": f"{resistance_triplets_5dig(triplet)}"} for triplet in triplets], triplets
    elif prompt_layout == 'resistance_triplets_reversed_5dig':
        return [{"role": "user", "content": f"{resistance_triplets_reversed_5dig(triplet)}"} for triplet in triplets], triplets
    else:
        raise ValueError('Prompt layout not recognized - please use one of "number_triplets_3dig", "number_triplets_3dig_flipped", "number_triplets_5dig", "number_triplets_5dig_flipped", "number_triplets_3dig_similar", "number_triplets_3dig_flipped_similar", "number_triplets_5dig_similar", "number_triplets_5dig_flipped_similar", "medical_triplets"')

def generate_prompts_chat(n, prompt_layout: str = 'default'):
    if prompt_layout == 'default':
        return [{"role": "user", "content": f"{generate_prompt(pair)}"} for pair in return_pairwise_combinations(n)], return_pairwise_combinations(n)
    elif prompt_layout == 'int':
        return [{"role": "user", "content": f"{generate_prompt_int(pair)}"} for pair in return_pairwise_combinations(n)], return_pairwise_combinations(n)
    elif prompt_layout == 'scientific':
        return [{"role": "user", "content": f"{generate_prompt_scientific(pair)}"} for pair in return_pairwise_combinations(n)], return_pairwise_combinations(n)
    elif prompt_layout == 'str':
        return [{"role": "user", "content": f"{generate_prompt_str(pair)}"} for pair in return_pairwise_combinations(n)], return_pairwise_combinations(n)
    elif prompt_layout == 'corporate':
        return [{"role": "user", "content": f"{generate_prompt_corporate(pair)}"} for pair in return_pairwise_combinations(n)], return_pairwise_combinations(n)
    elif 'base' in prompt_layout:
        assert '-' in prompt_layout, 'Please specify the base in the format "base-<int>"'
        base = int(prompt_layout.split('-')[1])
        return [{"role": "user", "content": f"{generate_prompt_base(pair, base=base)}"} for pair in return_pairwise_combinations(n)], return_pairwise_combinations(n)
    elif 'roman' in prompt_layout:
        return [{"role": "user", "content": f"{generate_prompt_roman(pair)}"} for pair in return_pairwise_combinations(n)], return_pairwise_combinations(n) 
    elif 'default-claude' in prompt_layout:
        return [{"role": "user", "content": f"{generate_prompt_claude(pair)}"} for pair in return_pairwise_combinations(n)], return_pairwise_combinations(n) 
    else:
        raise ValueError('Prompt layout not recognized - please use one of "default", "int", "str", or "corporate"')

if __name__=='__main__':
    lis, pairwise = generate_prompts_chat(1000, 'roman')
    triplets, triplet_pairs = generate_prompts_triplets(np.load('data/prompts/triplets_close_3_digits.npy'), 'number_triplets_3dig_similar')
    print(lis[0:10])
    print(pairwise[0:10])
    print(triplets[0:10])
    print(triplet_pairs[0:10])
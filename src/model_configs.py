import os 
from openai import OpenAI
from anthropic import Anthropic

model_configs = {
    "anthropic": {
        "temperature": 0.0,
        "batch_size": 1,
        "num_threads": 25,
    },
    "together": {
        "temperature": 0.0,
        "batch_size": 1,
        "num_threads": 50,
    },
    "openai": {
        "temperature": 0.0,
        "batch_size": 1,
        "num_threads": 70,
    }
}

def return_client(provider):
    if provider == 'openai':
        client = OpenAI(
            base_url="http://localhost:2436/v1",
            api_key="token-abc123",
        )
    elif provider == 'together':
        api_key = os.environ.get("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError("TOGETHER_API_KEY not found in environment variables")
        client = OpenAI(
            base_url="https://api.together.xyz/v1",
            api_key=api_key,
        )
    elif provider == 'anthropic':
        client = Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
        )
    else:
        raise ValueError('Provider not recognized - please use "openai", "together", or "anthropic"')
    return client

model_name_mappings = {
    'llama-3.1-8b-instruct': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'llama-3.1-70b-instruct': 'meta-llama/Meta-Llama-3.1-70B-Instruct',
    'llama-3.1-8b-instruct-turbo': "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    'llama-3.1-70b-instruct-turbo': "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    'llama-3.3-70b-instruct-turbo': "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "Gemma 2 27B": "google/gemma-2-27b-it",
    "DeepSeek-V3": "deepseek-ai/DeepSeek-V3",
    "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
    'mixtral-8x22b-instruct': 'mistralai/Mixtral-8x22B-Instruct-v0.1'
}
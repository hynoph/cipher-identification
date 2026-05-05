import os

OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DEEPSEEK_API_KEY  = os.getenv("DEEPSEEK_API_KEY")
TOGETHER_API_KEY  = os.getenv("TOGETHER_API_KEY")

MODELS = [
    {"name": "claude-sonnet",  "provider": "anthropic", "model_id": "claude-sonnet-4-6"},
    {"name": "deepseek-r1",    "provider": "deepseek",  "model_id": "deepseek-reasoner"},
    {"name": "llama-3.3-70b",  "provider": "together",  "model_id": "meta-llama/Llama-3.3-70B-Instruct-Turbo"},
]

TEMPERATURE = 0
MAX_TOKENS  = 500
RETRY_LIMIT = 3
DELAY_SECS  = 0.5
SAVE_EVERY  = 50

from fundar_llms.utils import modelname
from typing import Optional

auto_tokenizer_ = None
class TokenizerModel(str):
    def __new__(cls, name, needs_auth):
        return super().__new__(cls, name)
    
    def __init__(self, name, needs_auth):
        self.needs_auth = needs_auth

    def auto_tokenizer_from_pretrained(self):
        global auto_tokenizer_
        if not auto_tokenizer_:
            from transformers import AutoTokenizer
            auto_tokenizer_ = AutoTokenizer
        return auto_tokenizer_.from_pretrained(self)

DEFAULT_TOKENIZER_MAP = {
    'phi3.5': \
        TokenizerModel('microsoft/Phi-3.5-mini-instruct', needs_auth=False),
    'phi3': \
        TokenizerModel('microsoft/Phi-3-mini-128k-instruct', needs_auth=False),
    'llama3.1': \
        TokenizerModel('meta-llama/Llama-3.1-8B-Instruct', needs_auth=True),
    'llama3.2': \
        TokenizerModel('meta-llama/Llama-3.2-3B-Instruct', needs_auth=True),
}

def get_tokenizer(model_name: str, 
                  tokenizer_map: Optional[dict[str, TokenizerModel]] = None, 
                  default: Optional[TokenizerModel] = None) -> TokenizerModel:
    if not tokenizer_map:
        global DEFAULT_TOKENIZER_MAP
        tokenizer_map = DEFAULT_TOKENIZER_MAP
    
    if not default:
        default = tokenizer_map['phi3.5']
    
    return tokenizer_map.get(model_name, default)


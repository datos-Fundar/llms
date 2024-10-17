from ollama import Client as OllamaClient
from fundar_llms.utils import get_available_vram

class Client(OllamaClient):
    def list_models(self, max_vram = None): # default max vram: (3.2 * 1.074E9)
        if max_vram is None:
            max_vram, _ = get_available_vram()
        models = self.list()
        models = models['models']
        models = [
            model['name']
            for model in models
            if model['size'] <= max_vram
        ]

        return models
from .LLMEnums import LLMEnums
from .providers.OpenAIProvider import OpenAIProvider
from .providers.LocalProvider import LocalProvider

class LLMProviderFactory:
    def __init__(self, config):
        self.config = config

    def create(self, provider: str):
        if provider == LLMEnums.OPENAI.value:
            return OpenAIProvider(
                api_key=self.config.OPENAI_API_KEY,
                api_url=self.config.OPENAI_API_URL,
                default_input_max_characters=self.config.INPUT_DAFAULT_MAX_CHARACTERS,
                default_generation_max_output_tokens=self.config.GENERATION_DAFAULT_MAX_TOKENS,
                default_generation_temperature=self.config.GENERATION_DAFAULT_TEMPERATURE
            )
        
        if provider == LLMEnums.LOCAL.value:
            # Create Local Provider
            local_provider = LocalProvider(
                default_input_max_characters=self.config.INPUT_DAFAULT_MAX_CHARACTERS
            )
            # Pre-load the local embedding model
            local_provider.set_embedding_model(
                model_id=self.config.EMBEDDING_MODEL_ID, 
                embedding_size=self.config.EMBEDDING_MODEL_SIZE
            )
            return local_provider

        return None
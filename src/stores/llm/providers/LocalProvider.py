from ..LLMInterface import LLMInterface
from sentence_transformers import SentenceTransformer

class LocalProvider(LLMInterface):
    def __init__(self, default_input_max_characters: int = 1024):
        self.model = None
        self.embedding_size = 384
        self.default_input_max_characters = default_input_max_characters

    def set_generation_model(self, model_id: str):
        # Local generation (LLMs) requires large resources; 
        # for now, we focus on local embedding.
        pass

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.model = SentenceTransformer(model_id)
        self.embedding_size = embedding_size

    def generate_text(self, prompt: str, chat_history: list=[], max_output_tokens: int=None,
                      temperature: float = None):
        return "Local text generation not implemented. Using Local Embedding only."

    def embed_text(self, text: str, document_type: str = None):
        if self.model is None:
            # Default model if not explicitly set
            self.set_embedding_model("all-MiniLM-L6-v2", 384)
            
        # Truncate text if it exceeds max characters
        processed_text = text[:self.default_input_max_characters]
        embedding = self.model.encode(processed_text)
        return embedding.tolist()

    def construct_prompt(self, prompt: str, role: str):
        return {"role": role, "content": prompt}
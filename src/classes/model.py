import os
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

class MedicalModel:
    def __init__(self, model_name="mlx-community/Meta-Llama-3-8B-Instruct-4bit", adapter_path="./persisted"):
        self.model_name = model_name
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        self.sampler = make_sampler(temp=0.1)
        self.stop_tokens = ["<|eot_id|>", "<|end_of_text|>", "Question:", "\n---"]

    def load_model(self):
        """Loads the model and tokenizer, applying adapters if present."""
        print(f"Loading model: {self.model_name}" + (f" with adapters from: {self.adapter_path}" if self.adapter_path else "..."))
        self.model, self.tokenizer = load(self.model_name, adapter_path=self.adapter_path)
        return self.model, self.tokenizer

    def clean_response(self, response):
        """Truncate response at common stop tokens."""
        for token in self.stop_tokens:
            if token in response:
                response = response.split(token)[0]
        return response.strip()

    def ask(self, user_question, context=None, max_tokens=512):
        """Generate and clean a response for a given question, optionally using provided context (RAG)."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before calling ask().")

        if context:
            # RAG Prompt
            prompt = (
                f"Relevant medical information:\n{context}\n\n"
                f"Based on the information above, please answer the following question. "
                f"Include the source URL in your answer.\n\n"
                f"Question: {user_question}\n"
                f"Answer:"
            )
        else:
            # Standard Prompt
            prompt = f"Question: {user_question}\nAnswer:"
        
        raw_response = generate(
            self.model, 
            self.tokenizer, 
            prompt=prompt, 
            max_tokens=max_tokens,
            sampler=self.sampler,
            verbose=False
        )
        
        return self.clean_response(raw_response)

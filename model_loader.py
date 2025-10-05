import torch
from transformers import pipeline

class ModelLoader:
    """
    Loads and manages the Hugging Face text generation model.
    Optimized for CPU inference.
    """
    
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """
        Initialize and load the text generation model for CPU.
        
        Args:
            model_name (str): Hugging Face model identifier
        """
        self.model_name = model_name
        self.device = "cpu"
        self.pipe = None
        self.load_model()
    
    def load_model(self):
        """Load the model using Hugging Face pipeline on CPU"""
        try:
            print(f"Loading model: {self.model_name}...")
            print("Running on CPU (this may take a moment)...")
            
            self.pipe = pipeline(
                "text-generation",
                model=self.model_name,
                device=-1  # -1 indicates CPU
            )
            
            print(f"✓ Model loaded successfully on CPU!\n")
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise
    
    def generate_response(self, messages, max_new_tokens=80):
        try:
            prompt = self.pipe.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            outputs = self.pipe(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.3,  # Lower temperature for more focused responses
                top_k=30,
                top_p=0.85,
                repetition_penalty=1.2,  # Prevent repetition
                pad_token_id=self.pipe.tokenizer.eos_token_id
            )
            
            return outputs[0]["generated_text"]
            
        except Exception as e:
            print(f"✗ Generation error: {e}")
            return "Sorry, I encountered an error generating a response."
    
    def get_device_info(self):
        """
        Return current device information.
        
        Returns:
            dict: Device information
        """
        return {"device": "CPU"}

import json
import logging
from typing import List, Dict, Any, Optional
import requests

from app.config import settings

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        """
        Initializes the LLM Service.
        It detects OpenRouter key vs native Google Gemini key formats automatically,
        and sets up main and fallback models.
        """
        self.api_key = api_key or settings.GEMINI_API_KEY
        
        # Load main and fallback models from config
        self.main_model = model_name or settings.LLM_MAIN_MODEL
        self.fallback_model_1 = settings.LLM_FALLBACK_MODEL_1
        self.fallback_model_2 = settings.LLM_FALLBACK_MODEL_2
        
        # Keep back-compat property model_name
        self.model_name = self.main_model
        
        # Simple heuristics to detect OpenRouter keys vs Google Gemini keys
        if self.api_key.startswith("sk-or-"):
            self.provider = "openrouter"
            self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        else:
            self.provider = "google"
            # For native Google, URL template changes per model name; we build it dynamically per call
            self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"

    def generate_response(
        self, 
        prompt: str, 
        system_instruction: Optional[str] = None, 
        temperature: float = 0.7
    ) -> str:
        """
        Generates a textual response for the given prompt, trying the main model first,
        and falling back to the remaining models if the call fails.
        """
        if not self.api_key:
            raise ValueError("API key is not configured. Please set GEMINI_API_KEY in your .env file.")
            
        models_to_try = [
            self.main_model,
            self.fallback_model_1,
            self.fallback_model_2
        ]
        # Filter out empty entries
        models_to_try = [m for m in models_to_try if m]
        
        last_exception = None
        for model in models_to_try:
            try:
                logger.info(f"Attempting LLM generation with model: {model}")
                if self.provider == "openrouter":
                    return self._call_openrouter_with_model(model, prompt, system_instruction, temperature)
                else:
                    return self._call_google_with_model(model, prompt, system_instruction, temperature)
            except Exception as e:
                logger.warning(f"Model {model} failed with error: {e}. Trying fallback...")
                last_exception = e
                
        raise RuntimeError(f"LLM generation failed for all configured models. Last error: {last_exception}")

    def generate_rag_response(
        self, 
        query: str, 
        contexts: List[str], 
        system_instruction: Optional[str] = None
    ) -> str:
        """
        Generates a RAG-augmented response using context snippets and a user query.
        """
        formatted_context = "\n\n".join([f"[Context {i+1}]: {ctx}" for i, ctx in enumerate(contexts)])
        
        prompt = f"""Use the following context snippets to answer the user query. If the context does not contain enough information to answer, state that you do not know based on the context.

---
CONTEXT:
{formatted_context}
---

USER QUERY:
{query}

ANSWER:"""

        default_system = "You are a helpful knowledge assistant. Always answer accurately and truthfully based on the provided context."
        sys_inst = system_instruction or default_system
        
        return self.generate_response(prompt, system_instruction=sys_inst, temperature=0.3)

    def _call_openrouter_with_model(self, model: str, prompt: str, system_instruction: Optional[str], temperature: float) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/HEMANTH05BB/RAG-AI-Knowledge-Operating-System",
            "X-Title": "AI Knowledge Operating System"
        }
        
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            res_data = response.json()
            return res_data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"OpenRouter API call failed for model {model}: {e}")
            raise RuntimeError(f"OpenRouter API call failed: {e}")

    def _call_google_with_model(self, model: str, prompt: str, system_instruction: Optional[str], temperature: float) -> str:
        url = f"{self.base_url}/{model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        
        contents = [{"parts": [{"text": prompt}]}]
        
        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature
            }
        }
        
        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }
            
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            res_data = response.json()
            return res_data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            logger.error(f"Google Gemini API call failed for model {model}: {e}")
            raise RuntimeError(f"Google Gemini API call failed: {e}")

    def generate(self, prompt: str) -> str:
        """
        Alias for generate_response to support exact API requirements.
        """
        return self.generate_response(prompt)

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
        It detects OpenRouter key vs native Google Gemini key formats automatically.
        """
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model_name = model_name or settings.LLM_MODEL_NAME
        
        # Simple heuristics to detect OpenRouter keys vs Google Gemini keys
        if self.api_key.startswith("sk-or-"):
            self.provider = "openrouter"
            self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        else:
            self.provider = "google"
            self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"

    def generate_response(
        self, 
        prompt: str, 
        system_instruction: Optional[str] = None, 
        temperature: float = 0.7
    ) -> str:
        """
        Generates a textual response for the given prompt from the configured LLM provider.
        """
        if not self.api_key:
            raise ValueError("API key is not configured. Please set GEMINI_API_KEY in your .env file.")
            
        if self.provider == "openrouter":
            return self._call_openrouter(prompt, system_instruction, temperature)
        else:
            return self._call_google(prompt, system_instruction, temperature)

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

    def _call_openrouter(self, prompt: str, system_instruction: Optional[str], temperature: float) -> str:
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
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            res_data = response.json()
            
            # OpenRouter / chat completions response structure:
            # choices[0].message.content
            return res_data["choices"][0]["message"]["content"]
            
        except Exception as e:
            logger.error(f"OpenRouter API call failed: {e}")
            raise RuntimeError(f"LLM generation failed: {e}")

    def _call_google(self, prompt: str, system_instruction: Optional[str], temperature: float) -> str:
        # Build Google Gemini API payload
        # Native URL requires key parameter
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
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
            
            # Google Gemini response structure:
            # candidates[0].content.parts[0].text
            return res_data["candidates"][0]["content"]["parts"][0]["text"]
            
        except Exception as e:
            logger.error(f"Google Gemini API call failed: {e}")
            raise RuntimeError(f"LLM generation failed: {e}")

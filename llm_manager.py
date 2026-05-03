import os
from dotenv import load_dotenv
from openai import OpenAI
from google import genai

load_dotenv()

class LLMManager:
    def __init__(self):
        self.provider = os.getenv("ACTIVE_LLM", "openai").lower()
        print(f"[*] Inicializando LLM Manager com provedor: {self.provider.upper()}")

        if self.provider == "openai":
            self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = os.getenv("OPENAI_MODEL", "gpt-4o")
        
        elif self.provider == "deepseek":
            self.deepseek_client = OpenAI(
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com"
            )
            self.model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
        
        elif self.provider == "gemini":
            self.gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            self.model = os.getenv("GEMINI_MODEL", "gemini-3-flash")
        
        else:
            raise ValueError(f"Provedor LLM {self.provider} não suportado.")
    
    def get_trading_decision(self, prompt: str, system_prompt: str = "") -> str:
        """Roteia a chamada para a LLM configurada e retorna a resposta."""
        try:
            if self.provider == "openai":
                return self._call_openai(prompt, system_prompt)
            elif self.provider == "deepseek":
                return self._call_deepseek(prompt, system_prompt)
            elif self.provider == "gemini":
                return self._call_gemini(prompt, system_prompt)
        except Exception as e:
            print(f"[!] Erro ao chamar a API {self.provider.upper()}: {e}")
            return "ERRO"

    def _call_openai(self, prompt: str, system_prompt: str) -> str:
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=10
        )
        return response.choices[0].message.content.strip().upper()

    def _call_deepseek(self, prompt: str, system_prompt: str) -> str:
        response = self.deepseek_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=10
        )
        return response.choices[0].message.content.strip().upper()
    
    def _call_gemini(self, prompt: str, system_prompt: str) -> str:
        full_prompt = f"{system_prompt}\n\nUser: {prompt}"
        response = self.gemini_client.models.generate_content(
            model=self.model,
            contents=full_prompt
        )
        return response.text.strip().upper()

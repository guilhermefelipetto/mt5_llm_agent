import json

from openai import OpenAI

from config import settings


_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "trading_action",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "OPEN_LONG",
                        "OPEN_SHORT",
                        "HOLD",
                        "CLOSE",
                        "TIGHTEN_STOP",
                    ],
                    "description": (
                        "OPEN_LONG/OPEN_SHORT abre posição (ou inverte se já há "
                        "oposta aberta). HOLD não faz nada (deixa correr o trade "
                        "atual ou permanece flat). CLOSE encerra a posição "
                        "atual antes de SL/TP. TIGHTEN_STOP move o SL para "
                        "travar lucro - só permitido apertando, nunca afrouxando."
                    ),
                },
                "confidence": {
                    "type": "number",
                    "description": "Confiança de 0.0 a 1.0.",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Justificativa técnica curta (1-2 frases).",
                },
                "new_sl": {
                    "type": ["number", "null"],
                    "description": (
                        "Novo SL em preço absoluto, obrigatório para "
                        "TIGHTEN_STOP. Null para todas as outras ações."
                    ),
                },
            },
            "required": ["action", "confidence", "reasoning", "new_sl"],
            "additionalProperties": False,
        },
    },
}


class LLMManager:
    def __init__(self):
        self._client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        self.provider = "openai"
        print(f"[*] LLM Manager: OpenAI / {self.model}")

    def get_decision(self, user_prompt: str, system_prompt: str = "") -> dict:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                response_format=_RESPONSE_FORMAT,
                max_completion_tokens=2000,
            )
            choice = response.choices[0]
            if choice.finish_reason == "length":
                print("[!] OpenAI: resposta truncada - aumente max_completion_tokens.")
            return json.loads(choice.message.content)
        except Exception as e:
            print(f"[!] Erro LLM: {e}")
            return self._neutral()

    @staticmethod
    def _neutral() -> dict:
        return {
            "action": "HOLD",
            "confidence": 0.0,
            "reasoning": "Falha na inferência.",
            "new_sl": None,
        }

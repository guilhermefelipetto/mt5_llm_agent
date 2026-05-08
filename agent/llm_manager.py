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
                        "OPEN_LONG/OPEN_SHORT abre nova posição (escolha "
                        "intended_horizon). HOLD não faz nada - slot vazio é "
                        "melhor que slot com trade ruim. CLOSE encerra uma "
                        "posição existente antes de SL/TP. TIGHTEN_STOP aperta "
                        "o SL para travar lucro. Para CLOSE/TIGHTEN_STOP com "
                        "múltiplas posições abertas, position_id é obrigatório."
                    ),
                },
                "intended_horizon": {
                    "type": ["string", "null"],
                    "enum": ["scalp", "intraday", "swing", None],
                    "description": (
                        "Horizonte da operação. OBRIGATÓRIO para "
                        "OPEN_LONG/OPEN_SHORT; null para HOLD/CLOSE/"
                        "TIGHTEN_STOP. scalp = M5/M1 dominante (~minutos a "
                        "poucas horas), intraday = H1/M30 dominante (~horas), "
                        "swing = D1/4H dominante (~dias). Cada horizonte tem "
                        "SL/TP, time exit e cadência de revisita próprios."
                    ),
                },
                "position_id": {
                    "type": ["integer", "null"],
                    "description": (
                        "Ticket da posição-alvo de CLOSE/TIGHTEN_STOP. Null "
                        "para OPEN/HOLD. Se houver apenas uma posição aberta, "
                        "pode ser null e o sistema infere; com múltiplas, é "
                        "obrigatório."
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
            "required": [
                "action", "intended_horizon", "position_id",
                "confidence", "reasoning", "new_sl",
            ],
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
            "intended_horizon": None,
            "position_id": None,
            "confidence": 0.0,
            "reasoning": "Falha na inferência.",
            "new_sl": None,
        }

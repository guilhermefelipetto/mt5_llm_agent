import os
import json

import openai
import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from llm_manager import LLMManager
app = FastAPI(title="MT5 LLM Agent API")
llm = LLMManager()

class MarketData(BaseModel):
    symbol: str
    price: float

def log_experiment_decision(data: MarketData, provider: str, raw_decision: str, clean_decision: str):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "symbol": data.symbol,
        "price": data.price,
        "provider": provider,
        "raw_response": raw_decision,
        "final_decision": clean_decision
    }

    with open("trading_experiment_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")


@app.post("/analyze")
async def analyze_market(data: MarketData):
    print(f"\n[+] Novo Tick Recebido -> {data.symbol}: {data.price}")

    system_prompt = """
    Você é um bot de trading quantitativo focado em Forex. Responda APENAS com uma destas três palavras: COMPRA, VENDA ou NEUTRO.
    """

    user_prompt = f"""
    O ativo {data.symbol} está custando {data.price} neste exato momento. baseado na sua intuição estatística (ou random walk), qual sua decisão?
    """

    raw_decision = llm.get_trading_decision(user_prompt, system_prompt)
    print(f"[+] Resposta Bruta ({llm.provider}): {raw_decision}")

    clean_decision = "NEUTRO"
    if "COMPRA" in raw_decision:
        clean_decision = "COMPRA"
    elif "VENDA" in raw_decision:
        clean_decision = "VENDA"
    
    log_experiment_decision(data, llm.provider, raw_decision, clean_decision)

    print(f"[*] Decisão Final enviada ao MT5: {clean_decision}")

    return {
        "provider": llm.provider,
        "decision": clean_decision
    }

@app.get("/")
def read_root():
    return {"Status": "Online", "Ativo": llm.provider}

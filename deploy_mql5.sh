#!/bin/bash
# Copia o EA para o diretório Experts do MT5 via Bottles.

EXPERTS_DIR="$HOME/.var/app/com.usebottles.bottles/data/bottles/bottles/MetaTrader5/drive_c/Program Files/MetaTrader 5/MQL5/Experts"

if [ ! -d "$EXPERTS_DIR" ]; then
    echo "[ERRO] Diretório do MT5 não encontrado: $EXPERTS_DIR"
    exit 1
fi

cp mql5/LLM_Trader.mq5 "$EXPERTS_DIR/LLM_Trader.mq5"
echo "[OK] LLM_Trader.mq5 copiado para Experts."
echo
echo "Próximos passos:"
echo "  1. Abra o MetaEditor, compile e arraste o EA para o gráfico."
echo "  2. Em Tools > Options > Expert Advisors > Allow WebRequest, libere"
echo "     a URL do servidor:"
echo "       - Windows nativo:  http://127.0.0.1:8000"
echo "       - Linux/Wine:      http://<IP-LAN>:8000  (ex: http://192.168.0.10:8000)"
echo "         Loopback falha sob Wine - use o IP de LAN do host."
echo "  3. Ajuste InpServerURL nos parâmetros do EA para a mesma URL liberada."

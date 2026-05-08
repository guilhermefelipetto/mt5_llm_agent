//+------------------------------------------------------------------+
//|                                                   LLM_Trader.mq5 |
//|                              Copyright 2026, Guilherme Felipetto |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, Guilherme Felipetto"
#property version   "1.70"

#include <Trade/Trade.mqh>

input long   InpMagicNumber  = 20260505;                          // Magic number do EA
input int    InpSlippage     = 10;                                // Desvio max em points
input int    InpPollInterval = 30;                                // Intervalo entre polls (s)
input string InpServerURL    = "http://127.0.0.1:8000/signal";    // Endpoint do servidor

CTrade trade;

//+------------------------------------------------------------------+
//| Filling mode suportado pelo broker para o simbolo                |
//+------------------------------------------------------------------+
ENUM_ORDER_TYPE_FILLING DetectFillingMode(const string symbol)
{
   long modes = SymbolInfoInteger(symbol, SYMBOL_FILLING_MODE);
   if((modes & SYMBOL_FILLING_FOK) != 0) return ORDER_FILLING_FOK;
   if((modes & SYMBOL_FILLING_IOC) != 0) return ORDER_FILLING_IOC;
   return ORDER_FILLING_FOK;
}

int OnInit()
{
   trade.SetExpertMagicNumber(InpMagicNumber);
   trade.SetDeviationInPoints(InpSlippage);
   trade.SetTypeFilling(DetectFillingMode(_Symbol));

   if(!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED))
      Print("[!] AVISO: AutoTrading desabilitado no terminal.");
   if(!MQLInfoInteger(MQL_TRADE_ALLOWED))
      Print("[!] AVISO: Trading nao permitido para este EA.");

   PrintFormat("EA LLM_Trader v1.70 (multi-position) | Symbol: %s | Magic: %d | Poll: %ds",
               _Symbol, InpMagicNumber, InpPollInterval);
   PrintFormat("Libere %s em Tools > Options > Expert Advisors > Allow WebRequest.",
               InpServerURL);
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| OnTick - poll do sinal e dispatch                                |
//+------------------------------------------------------------------+
void OnTick()
{
   static datetime last_check = 0;
   if(TimeCurrent() - last_check < InpPollInterval) return;
   last_check = TimeCurrent();

   string body;
   if(!FetchSignal(body)) return;

   string action = ParseField(body, "action");
   string valid  = ParseField(body, "valid");
   if(action == "" || action == "HOLD" || valid == "false") return;

   string sig_symbol = ParseField(body, "symbol");
   if(sig_symbol != "" && StringFind(_Symbol, sig_symbol) < 0)
   {
      PrintFormat("[!] Grafico (%s) nao bate com sinal (%s). Ignorando.",
                  _Symbol, sig_symbol);
      return;
   }

   string reasoning = ParseField(body, "reasoning");

   if(action == "OPEN_LONG" || action == "OPEN_SHORT")
   {
      HandleOpen(action, body, reasoning);
   }
   else if(action == "CLOSE")
   {
      HandleClose(body, reasoning);
   }
   else if(action == "TIGHTEN_STOP")
   {
      HandleTightenStop(body, reasoning);
   }
   else
   {
      PrintFormat("[!] Acao desconhecida: %s", action);
   }
}

//+------------------------------------------------------------------+
//| OPEN_LONG / OPEN_SHORT - abre nova posicao.                      |
//| O servidor (Python) ja validou que nao colide com outras nossas. |
//| O EA NAO faz mais auto-close de oposta (multi-position v1.7).    |
//+------------------------------------------------------------------+
void HandleOpen(const string action, const string body, const string reasoning)
{
   double sl  = StringToDouble(ParseField(body, "sl_price"));
   double tp  = StringToDouble(ParseField(body, "tp_price"));
   double lot = StringToDouble(ParseField(body, "lot"));
   string horizon = ParseField(body, "intended_horizon");
   if(lot <= 0) lot = 0.01;

   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   string comment = StringFormat("LLM v1.7 %s", horizon);

   if(action == "OPEN_LONG")
   {
      if(trade.Buy(lot, _Symbol, ask, sl, tp, comment))
         PrintFormat("[+] OPEN_LONG [%s] @ %.5f | SL: %.5f TP: %.5f | %s",
                     horizon, ask, sl, tp, reasoning);
      else
         PrintFormat("[!] Erro OPEN_LONG: %d %s",
                     trade.ResultRetcode(), trade.ResultRetcodeDescription());
   }
   else
   {
      if(trade.Sell(lot, _Symbol, bid, sl, tp, comment))
         PrintFormat("[+] OPEN_SHORT [%s] @ %.5f | SL: %.5f TP: %.5f | %s",
                     horizon, bid, sl, tp, reasoning);
      else
         PrintFormat("[!] Erro OPEN_SHORT: %d %s",
                     trade.ResultRetcode(), trade.ResultRetcodeDescription());
   }
}

//+------------------------------------------------------------------+
//| CLOSE - fecha posicao especifica via position_id (ticket).       |
//+------------------------------------------------------------------+
void HandleClose(const string body, const string reasoning)
{
   long ticket = StringToInteger(ParseField(body, "position_id"));
   if(ticket <= 0)
   {
      Print("[!] CLOSE sem position_id valido.");
      return;
   }

   if(!PositionSelectByTicket((ulong)ticket))
   {
      PrintFormat("[!] CLOSE: ticket %d nao encontrado.", ticket);
      return;
   }
   if(PositionGetInteger(POSITION_MAGIC) != InpMagicNumber)
   {
      PrintFormat("[!] CLOSE: ticket %d nao e nosso (magic diferente).", ticket);
      return;
   }

   if(trade.PositionClose((ulong)ticket))
      PrintFormat("[+] CLOSE ticket %d | %s", ticket, reasoning);
   else
      PrintFormat("[!] Erro CLOSE ticket %d: %d %s",
                  ticket, trade.ResultRetcode(), trade.ResultRetcodeDescription());
}

//+------------------------------------------------------------------+
//| TIGHTEN_STOP - modifica SL de posicao especifica via ticket.     |
//| Servidor ja validou que aperta (nao afrouxa).                    |
//+------------------------------------------------------------------+
void HandleTightenStop(const string body, const string reasoning)
{
   long ticket = StringToInteger(ParseField(body, "position_id"));
   double new_sl = StringToDouble(ParseField(body, "new_sl"));
   if(ticket <= 0 || new_sl <= 0)
   {
      Print("[!] TIGHTEN_STOP sem position_id ou new_sl validos.");
      return;
   }

   if(!PositionSelectByTicket((ulong)ticket))
   {
      PrintFormat("[!] TIGHTEN_STOP: ticket %d nao encontrado.", ticket);
      return;
   }
   if(PositionGetInteger(POSITION_MAGIC) != InpMagicNumber)
   {
      PrintFormat("[!] TIGHTEN_STOP: ticket %d nao e nosso.", ticket);
      return;
   }

   double current_tp = PositionGetDouble(POSITION_TP);
   if(trade.PositionModify((ulong)ticket, new_sl, current_tp))
      PrintFormat("[+] TIGHTEN_STOP ticket %d -> SL=%.5f | %s",
                  ticket, new_sl, reasoning);
   else
      PrintFormat("[!] Erro TIGHTEN_STOP ticket %d: %d %s",
                  ticket, trade.ResultRetcode(), trade.ResultRetcodeDescription());
}

//+------------------------------------------------------------------+
//| WebRequest                                                        |
//+------------------------------------------------------------------+
bool FetchSignal(string &body_out)
{
   string headers = "Content-Type: application/json\r\n";
   string resp_headers;
   char post_data[];
   char response[];

   ResetLastError();
   int res = WebRequest("GET", InpServerURL, headers, 5000, post_data, response, resp_headers);

   if(res == -1)
   {
      PrintFormat("[!] WebRequest erro %d - libere %s nas configuracoes.",
                  GetLastError(), InpServerURL);
      return false;
   }
   if(res != 200)
   {
      PrintFormat("[!] API retornou HTTP %d", res);
      return false;
   }

   body_out = CharArrayToString(response, 0, ArraySize(response));
   return true;
}

//+------------------------------------------------------------------+
//| Parser JSON simples                                              |
//+------------------------------------------------------------------+
string ParseField(const string json, const string field)
{
   string search = "\"" + field + "\":";
   int start = StringFind(json, search);
   if(start < 0) return "";
   start += StringLen(search);

   while(start < StringLen(json) && StringGetCharacter(json, start) == ' ') start++;

   bool is_str = StringGetCharacter(json, start) == '"';
   if(is_str) start++;

   int end = start;
   if(is_str)
      while(end < StringLen(json) && StringGetCharacter(json, end) != '"') end++;
   else
      while(end < StringLen(json) &&
            StringGetCharacter(json, end) != ',' &&
            StringGetCharacter(json, end) != '}') end++;

   return StringSubstr(json, start, end - start);
}

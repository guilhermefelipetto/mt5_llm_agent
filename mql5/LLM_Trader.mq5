//+------------------------------------------------------------------+
//|                                                   LLM_Trader.mq5 |
//|                              Copyright 2026, Guilherme Felipetto |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, Guilherme Felipetto"
#property version   "1.40"

#include <Trade/Trade.mqh>

input long   InpMagicNumber  = 20260505;                          // Magic number do EA
input int    InpSlippage     = 10;                                // Desvio máx em points
input int    InpPollInterval = 30;                                // Intervalo entre polls (s)
input string InpServerURL    = "http://127.0.0.1:8000/signal";    // Endpoint do servidor

CTrade trade;

//+------------------------------------------------------------------+
//| Detecta o filling mode suportado pelo broker para o símbolo      |
//+------------------------------------------------------------------+
ENUM_ORDER_TYPE_FILLING DetectFillingMode(const string symbol)
{
   long modes = SymbolInfoInteger(symbol, SYMBOL_FILLING_MODE);
   if((modes & SYMBOL_FILLING_FOK) != 0) return ORDER_FILLING_FOK;
   if((modes & SYMBOL_FILLING_IOC) != 0) return ORDER_FILLING_IOC;
   return ORDER_FILLING_FOK;
}

//+------------------------------------------------------------------+
//| OnInit                                                           |
//+------------------------------------------------------------------+
int OnInit()
{
   trade.SetExpertMagicNumber(InpMagicNumber);
   trade.SetDeviationInPoints(InpSlippage);
   trade.SetTypeFilling(DetectFillingMode(_Symbol));

   if(!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED))
      Print("[!] AVISO: AutoTrading desabilitado no terminal.");
   if(!MQLInfoInteger(MQL_TRADE_ALLOWED))
      Print("[!] AVISO: Trading não permitido para este EA.");

   PrintFormat("EA LLM_Trader v1.40 (stateful) | Symbol: %s | Magic: %d | Poll: %ds",
               _Symbol, InpMagicNumber, InpPollInterval);
   PrintFormat("Libere %s em Tools > Options > Expert Advisors > Allow WebRequest.",
               InpServerURL);
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| OnTick - poll do sinal e dispatch de ação                        |
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
      PrintFormat("[!] Gráfico (%s) não bate com sinal (%s). Ignorando.",
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
      HandleClose(reasoning);
   }
   else if(action == "TIGHTEN_STOP")
   {
      HandleTightenStop(body, reasoning);
   }
   else
   {
      PrintFormat("[!] Ação desconhecida: %s", action);
   }
}

//+------------------------------------------------------------------+
//| OPEN_LONG / OPEN_SHORT - fecha oposta se houver e abre nova      |
//+------------------------------------------------------------------+
void HandleOpen(const string action, const string body, const string reasoning)
{
   bool want_long = (action == "OPEN_LONG");

   if(HasOurPosition(_Symbol, InpMagicNumber))
   {
      long pos_type = PositionGetInteger(POSITION_TYPE);
      bool same_side = (want_long && pos_type == POSITION_TYPE_BUY)
                    || (!want_long && pos_type == POSITION_TYPE_SELL);
      if(same_side) return; // já estamos na direção certa

      if(!trade.PositionClose(_Symbol))
      {
         PrintFormat("[!] Falha ao fechar oposta: %d %s",
                     trade.ResultRetcode(), trade.ResultRetcodeDescription());
         return;
      }
      Print("[~] Posição oposta fechada para reversão.");
   }

   double sl  = StringToDouble(ParseField(body, "sl_price"));
   double tp  = StringToDouble(ParseField(body, "tp_price"));
   double lot = StringToDouble(ParseField(body, "lot"));
   if(lot <= 0) lot = 0.01;

   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);

   if(want_long)
   {
      if(trade.Buy(lot, _Symbol, ask, sl, tp, "LLM v1.4"))
         PrintFormat("[+] OPEN_LONG @ %.5f | SL: %.5f  TP: %.5f | %s",
                     ask, sl, tp, reasoning);
      else
         PrintFormat("[!] Erro OPEN_LONG: %d %s",
                     trade.ResultRetcode(), trade.ResultRetcodeDescription());
   }
   else
   {
      if(trade.Sell(lot, _Symbol, bid, sl, tp, "LLM v1.4"))
         PrintFormat("[+] OPEN_SHORT @ %.5f | SL: %.5f  TP: %.5f | %s",
                     bid, sl, tp, reasoning);
      else
         PrintFormat("[!] Erro OPEN_SHORT: %d %s",
                     trade.ResultRetcode(), trade.ResultRetcodeDescription());
   }
}

//+------------------------------------------------------------------+
//| CLOSE - fecha posição atual antes de SL/TP                       |
//+------------------------------------------------------------------+
void HandleClose(const string reasoning)
{
   if(!HasOurPosition(_Symbol, InpMagicNumber)) return;

   if(trade.PositionClose(_Symbol))
      PrintFormat("[+] CLOSE executado | %s", reasoning);
   else
      PrintFormat("[!] Erro CLOSE: %d %s",
                  trade.ResultRetcode(), trade.ResultRetcodeDescription());
}

//+------------------------------------------------------------------+
//| TIGHTEN_STOP - modifica SL para travar lucro                     |
//| Servidor já validou que aperta (não afrouxa).                    |
//+------------------------------------------------------------------+
void HandleTightenStop(const string body, const string reasoning)
{
   if(!HasOurPosition(_Symbol, InpMagicNumber)) return;

   double new_sl = StringToDouble(ParseField(body, "new_sl"));
   if(new_sl <= 0)
   {
      Print("[!] TIGHTEN_STOP recebido sem new_sl válido.");
      return;
   }

   double current_tp = PositionGetDouble(POSITION_TP);
   ulong  ticket     = (ulong)PositionGetInteger(POSITION_TICKET);

   if(trade.PositionModify(ticket, new_sl, current_tp))
      PrintFormat("[+] TIGHTEN_STOP → SL=%.5f | %s", new_sl, reasoning);
   else
      PrintFormat("[!] Erro TIGHTEN_STOP: %d %s",
                  trade.ResultRetcode(), trade.ResultRetcodeDescription());
}

//+------------------------------------------------------------------+
//| WebRequest do sinal corrente                                     |
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
      PrintFormat("[!] WebRequest erro %d - libere %s nas configurações.",
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
//| Posição NOSSA (mesmo magic) no símbolo? Side effect: seleciona.  |
//+------------------------------------------------------------------+
bool HasOurPosition(const string symbol, const long magic)
{
   for(int i = 0; i < PositionsTotal(); i++)
   {
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetString(POSITION_SYMBOL) == symbol &&
         PositionGetInteger(POSITION_MAGIC) == magic)
         return true;
   }
   return false;
}

//+------------------------------------------------------------------+
//| Parser JSON simples (sem dependência externa)                    |
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

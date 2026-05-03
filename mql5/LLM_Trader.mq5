//+------------------------------------------------------------------+
//|                                                   LLM_Trader.mq5 |
//|                              Copyright 2026, Guilherme Felipetto |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, Guilherme Felipetto"
#property link      "https://www.mql5.com"
#property version   "1.00"

#include <Trade/Trade.mqh>
CTrade trade;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   Print("EA LLM_TRADER iniciando. Pronto para consultar a IA.");
   return(INIT_SUCCEEDED);
  }
  
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   static datetime last_check_time = 0;
   datetime current_time = TimeCurrent();
   
   if (current_time - last_check_time < 60){
      return;
   }
   last_check_time = current_time;
   
   string symbol = _Symbol;
   double price = SymbolInfoDouble(symbol, SYMBOL_ASK);
   
   string url = "http://127.0.0.1:8000/analyze";
   string headers;
   char post_data[], response[];
   string response_str;
   
   string json_body = StringFormat("{\"symbol\":\"%s\", \"price\":%f}", symbol, price);
   StringToCharArray(json_body, post_data);
   
   ResetLastError();
   int timeout = 5000;
   
   int res = WebRequest("POST", url, "Content-Type: application/json", timeout, post_data, response, headers);
   
   if(res == -1){
      Print("Erro no WebRequest: ", GetLastError());
      // Erro 4014 significa que a URL não tem permissão nas configurações do MT5!
   }
   else if(res == 200){
      response_str = CharArrayToString(response);
      Print("Resposta da API recebida: ", response_str);
      
      string decision = "";
      if (StringFind(response_str, "COMPRA") > 0){
         decision = "COMPRA";
      }
      else if(StringFind(response_str, "VENDA") > 0){
         decision = "VENDA";
      }
      else{
         decision = "NEUTRO";
      }
      
      Print("Decisão da LLM: ", decision);
      double lot = 0.01;
      
      if (decision == "COMPRA"){
         if(!PositionSelect(symbol)){
            bool result = trade.Buy(lot, symbol, price, 0, 0, "LLM Buy");
            if(result)
               Print("Ordem de COMPRA executada com sucesso.");
            else
               Print("Erro ao executar COMPRA: ", GetLastError());
         }
      }
      else if(decision == "VENDA"){
         if(!PositionSelect(symbol)){
            bool result = trade.Sell(lot, symbol, price, 0, 0, "LLM Sell");
            if(result)
               Print("Ordem de VENDA executada com sucesso.");
            else
               Print("Erro ao executar VENDA: ", GetLastError());
         }
      }
   }
   else{
      Print("A API retornou um código de erro HTTP: ", res);
   }
  }
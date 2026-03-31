//+------------------------------------------------------------------+
//|                                            NovaGold_Baseline.mq5 |
//|                                  Copyright 2026, NovaGold Reborn |
//+------------------------------------------------------------------+
#property copyright "NovaGold Reborn"
#property link      ""
#property version   "1.00"

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\SymbolInfo.mqh>

//--- Inputs
input double   InpLotSize              = 0.01;      // Lot Size
input int      InpKeltnerEMAPeriod     = 20;        // Keltner EMA Period
input int      InpKeltnerATRPeriod     = 14;        // Keltner ATR Period
input double   InpKeltnerMultiplier    = 2.0;       // Keltner Multiplier
input double   InpSLMultiplier         = 2.0;       // Stop Loss (ATR mult)
input double   InpBreakevenMultiplier  = 1.0;       // Breakeven Start (ATR mult)
input double   InpTrailingMultiplier   = 1.5;       // Trailing Distance (ATR mult)

//--- Global Variables
CTrade         trade;
CSymbolInfo    sym;
CPositionInfo  pos;

int            handle_ema;
int            handle_atr;
datetime       last_bar_time;

double         ema_buffer[];
double         atr_buffer[];


//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   sym.Name(_Symbol);
   sym.RefreshRates();
   
   trade.SetExpertMagicNumber(12635);
   
   handle_ema = iMA(_Symbol, PERIOD_M5, InpKeltnerEMAPeriod, 0, MODE_EMA, PRICE_CLOSE);
   handle_atr = iATR(_Symbol, PERIOD_M5, InpKeltnerATRPeriod);
   
   if(handle_ema == INVALID_HANDLE || handle_atr == INVALID_HANDLE)
     {
      Print("Erreur d'initialisation des indicateurs");
      return(INIT_FAILED);
     }
     
   ArraySetAsSeries(ema_buffer, true);
   ArraySetAsSeries(atr_buffer, true);

   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   sym.RefreshRates();
   ManageTrailingStops();

   // On check les signaux uniquement à la création d'une nouvelle bougie M5
   datetime current_bar_time = iTime(_Symbol, PERIOD_M5, 0);
   if(current_bar_time == last_bar_time) return;

   // Récupération des données Keltner (index 1 = la bougie qui vient juste de se clôturer)
   if(CopyBuffer(handle_ema, 0, 1, 3, ema_buffer) <= 0) return;
   if(CopyBuffer(handle_atr, 0, 1, 3, atr_buffer) <= 0) return;
   
   double close1 = iClose(_Symbol, PERIOD_M5, 1);
   double close2 = iClose(_Symbol, PERIOD_M5, 2);
   
   double ema1 = ema_buffer[0];
   double atr1 = atr_buffer[0];
   double upper1 = ema1 + InpKeltnerMultiplier * atr1;
   double lower1 = ema1 - InpKeltnerMultiplier * atr1;
   
   double ema2 = ema_buffer[1];
   double atr2 = atr_buffer[1];
   double upper2 = ema2 + InpKeltnerMultiplier * atr2;
   double lower2 = ema2 - InpKeltnerMultiplier * atr2;

   // Verifier si position existante
   bool is_in_position = PositionsTotal() > 0;

   // LOGIQUE LONG : Close précédent sous Upper, Close actuel au-dessus de Upper
   if(close2 <= upper2 && close1 > upper1 && !is_in_position)
     {
      double sl = sym.Ask() - (InpSLMultiplier * atr1);
      trade.Buy(InpLotSize, _Symbol, sym.Ask(), sl, 0, "Nova Keltner Breakout LONG");
      last_bar_time = current_bar_time;
     }
   // LOGIQUE SHORT : Close précédent au-dessus de Lower, Close actuel en-dessous de Lower
   else if(close2 >= lower2 && close1 < lower1 && !is_in_position)
     {
      double sl = sym.Bid() + (InpSLMultiplier * atr1);
      trade.Sell(InpLotSize, _Symbol, sym.Bid(), sl, 0, "Nova Keltner Breakout SHORT");
      last_bar_time = current_bar_time;
     }
  }

//+------------------------------------------------------------------+
//| Manage Trailing Stops on every tick                              |
//+------------------------------------------------------------------+
void ManageTrailingStops()
  {
   if(!pos.Select(_Symbol)) return;
   
   double open_price = pos.PriceOpen();
   double current_sl = pos.StopLoss();
   
   // On a besoin de l'ATR actuel pour évaluer la distance
   double atr_array[];
   ArraySetAsSeries(atr_array, true);
   if(CopyBuffer(handle_atr, 0, 0, 1, atr_array) <= 0) return;
   double current_atr = atr_array[0];
   
   // Distances
   double be_distance = InpBreakevenMultiplier * current_atr;
   double trail_distance = InpTrailingMultiplier * current_atr;

   if(pos.PositionType() == POSITION_TYPE_BUY)
     {
      double pnl = sym.Bid() - open_price;
      
      // Activation du BreakEven si on dépasse le seuil
      if(pnl >= be_distance)
        {
         double new_sl = sym.Bid() - trail_distance;
         
         // On s'assure que le nouveau SL verrouille au moins le BreakEven
         if(new_sl < open_price) new_sl = open_price; 
         
         // On déplace le SL que s'il monte (et s'il est au-dessus du minimum autorise)
         if(new_sl > current_sl && (sym.Bid() - new_sl) >= sym.StopsLevel()*sym.Point())
           {
            trade.PositionModify(pos.Ticket(), new_sl, pos.TakeProfit());
           }
        }
     }
   else if(pos.PositionType() == POSITION_TYPE_SELL)
     {
      double pnl = open_price - sym.Ask();
      
      // Activation du BreakEven si on dépasse le seuil
      if(pnl >= be_distance)
        {
         double new_sl = sym.Ask() + trail_distance;
         
         // On s'assure que le nouveau SL verrouille au moins le BreakEven
         if(new_sl > open_price || new_sl == 0) new_sl = open_price;
         
         // On déplace le SL que s'il descend et s'il ne touche aucun zero precedant (et > stoplevel)
         if((new_sl < current_sl || current_sl == 0) && (new_sl - sym.Ask()) >= sym.StopsLevel()*sym.Point())
           {
            trade.PositionModify(pos.Ticket(), new_sl, pos.TakeProfit());
           }
        }
     }
  }

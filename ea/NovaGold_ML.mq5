//+------------------------------------------------------------------+
//|                                                NovaGold_ML.mq5   |
//|                                  Copyright 2026, NovaGold Reborn |
//|  Keltner Breakout + filtre ML (ONNX LightGBM)                    |
//+------------------------------------------------------------------+
#property copyright "NovaGold Reborn"
#property version   "1.00"
#property tester_file "model_M5.onnx"
#property tester_file "model_M1.onnx"

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\SymbolInfo.mqh>

//--- Inputs
input double InpLotSize             = 0.01;            // Lot Size
input int    InpKeltnerEMAPeriod    = 20;              // Keltner EMA Period
input int    InpKeltnerATRPeriod    = 14;              // Keltner ATR Period
input double InpKeltnerMultiplier   = 2.0;             // Keltner Multiplier
input double InpSLMultiplier        = 2.0;             // Stop Loss (ATR mult)
input double InpBreakevenMultiplier = 1.0;             // Breakeven (ATR mult)
input double InpTrailingMultiplier  = 1.5;             // Trailing SL (ATR mult)
input float  InpMLThreshold         = 0.62f;           // ML seuil probabilite [0-1]
input string InpModelFile           = "model_M5.onnx"; // Fichier ONNX (dans MQL5/Files/)

// Dynamic trailing parameters based on regime
input double InpTrailingTrending    = 2.0;             // Trailing when ADX > 25 (trending)
input double InpTrailingRange       = 0.75;            // Trailing when ADX < 20 (range)
input double InpADXPeriod           = 14;              // ADX period for trend detection
input double InpADXHighThreshold    = 25.0;            // ADX > X = trending market
input double InpADXLowThreshold     = 20.0;            // ADX < X = range market

//--- Globals
CTrade        trade;
CSymbolInfo   sym;
CPositionInfo pos;

int    handle_ema;
int    handle_atr;
int    handle_rsi;
int    handle_atr_h1;
int    handle_ema_h1;
int    handle_adx;
long   onnx_model  = INVALID_HANDLE;
datetime last_bar_time = 0;

//+------------------------------------------------------------------+
//| Init                                                             |
//+------------------------------------------------------------------+
int OnInit()
  {
   sym.Name(_Symbol);
   trade.SetExpertMagicNumber(12636);

   // Indicateurs M5
   handle_ema   = iMA (_Symbol, PERIOD_M5, InpKeltnerEMAPeriod, 0, MODE_EMA, PRICE_CLOSE);
   handle_atr   = iATR(_Symbol, PERIOD_M5, InpKeltnerATRPeriod);
   handle_rsi   = iRSI(_Symbol, PERIOD_M5, 14, PRICE_CLOSE);
   handle_adx   = iADX(_Symbol, PERIOD_M5, InpADXPeriod);
   // Indicateurs H1 (pour ema_slope_h1 et atr_ratio_mtf)
   handle_atr_h1 = iATR(_Symbol, PERIOD_H1, InpKeltnerATRPeriod);
   handle_ema_h1 = iMA (_Symbol, PERIOD_H1, InpKeltnerEMAPeriod, 0, MODE_EMA, PRICE_CLOSE);

   if(handle_ema   == INVALID_HANDLE || handle_atr    == INVALID_HANDLE ||
      handle_rsi   == INVALID_HANDLE || handle_adx     == INVALID_HANDLE ||
      handle_atr_h1 == INVALID_HANDLE || handle_ema_h1 == INVALID_HANDLE)
     {
      Print("[Nova ML] Erreur init indicateurs");
      return INIT_FAILED;
     }

   // Charger le modele ONNX
   onnx_model = OnnxCreate(InpModelFile, ONNX_DEFAULT);
   if(onnx_model == INVALID_HANDLE)
     {
      Print("[Nova ML] Impossible de charger le modele: ", InpModelFile);
      Print("[Nova ML] Verifiez que le fichier est dans MQL5/Files/");
      return INIT_FAILED;
     }

   // Shape input: [1, 8]  (1 sample, 8 features)
   ulong in_shape[] = {1, 8};
   if(!OnnxSetInputShape(onnx_model, 0, in_shape))
     {
      Print("[Nova ML] Erreur OnnxSetInputShape");
      return INIT_FAILED;
     }

   // Shape output 0: labels [1]
   ulong out_label_shape[] = {1};
   if(!OnnxSetOutputShape(onnx_model, 0, out_label_shape))
     {
      Print("[Nova ML] Erreur OnnxSetOutputShape labels");
      return INIT_FAILED;
     }

   // Shape output 1: probas [1, 2]
   ulong out_proba_shape[] = {1, 2};
   if(!OnnxSetOutputShape(onnx_model, 1, out_proba_shape))
     {
      Print("[Nova ML] Erreur OnnxSetOutputShape probas");
      return INIT_FAILED;
     }

   Print("[Nova ML] Modele ONNX charge: ", InpModelFile);
   return INIT_SUCCEEDED;
  }

//+------------------------------------------------------------------+
//| Deinit                                                           |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   if(onnx_model != INVALID_HANDLE)
      OnnxRelease(onnx_model);
  }

//+------------------------------------------------------------------+
//| Tick                                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   sym.RefreshRates();
   ManageTrailingStops();

   // Une fois par bougie M5 fermee
   datetime current_bar = iTime(_Symbol, PERIOD_M5, 0);
   if(current_bar == last_bar_time) return;

   // --- Buffers indicateurs ---
   double ema_buf[];   ArraySetAsSeries(ema_buf,   true);
   double atr_buf[];   ArraySetAsSeries(atr_buf,   true);
   double rsi_buf[];   ArraySetAsSeries(rsi_buf,   true);
   double adx_buf[];   ArraySetAsSeries(adx_buf,   true);
   double atr_h1_buf[];ArraySetAsSeries(atr_h1_buf,true);
   double ema_h1_buf[];ArraySetAsSeries(ema_h1_buf,true);

   if(CopyBuffer(handle_ema,    0, 1, 3,  ema_buf)    <= 0) return;
   if(CopyBuffer(handle_atr,    0, 1, 2,  atr_buf)    <= 0) return;
   if(CopyBuffer(handle_rsi,    0, 1, 1,  rsi_buf)    <= 0) return;
   if(CopyBuffer(handle_adx,    0, 0, 2,  adx_buf)    <= 0) return;
   if(CopyBuffer(handle_atr_h1, 0, 0, 2,  atr_h1_buf) <= 0) return;
   if(CopyBuffer(handle_ema_h1, 0, 0, 6,  ema_h1_buf) <= 0) return;

   double adx = adx_buf[0];  // ADX actuel pour détection régime

   double close1 = iClose(_Symbol, PERIOD_M5, 1);
   double close2 = iClose(_Symbol, PERIOD_M5, 2);

   double ema1   = ema_buf[0];
   double atr1   = atr_buf[0];
   double upper1 = ema1 + InpKeltnerMultiplier * atr1;
   double lower1 = ema1 - InpKeltnerMultiplier * atr1;

   double ema2   = ema_buf[1];
   double atr2   = atr_buf[1];
   double upper2 = ema2 + InpKeltnerMultiplier * atr2;
   double lower2 = ema2 - InpKeltnerMultiplier * atr2;

   bool is_long  = (close2 <= upper2 && close1 > upper1);
   bool is_short = (close2 >= lower2 && close1 < lower1);

   if(PositionsTotal() > 0 || (!is_long && !is_short))
      return;

   // --- Construction des 8 features ---
   // Ordre identique a FEATURE_COLUMNS dans ml/features.py:
   // [tick_volume_ratio, atr_ratio_mtf, keltner_distance, rsi_14,
   //  momentum_5, session_id, ema_slope_h1, vol_ratio_20]
   float features[8];

   // 1. tick_volume_ratio = vol[1] / mean(vol[2..21])
   long vol1 = iVolume(_Symbol, PERIOD_M5, 1);
   double vol_sum = 0;
   for(int k = 2; k <= 21; k++) vol_sum += (double)iVolume(_Symbol, PERIOD_M5, k);
   double vol_mean = (vol_sum > 0) ? vol_sum / 20.0 : 1.0;
   features[0] = (float)(vol1 / vol_mean);

   // 2. atr_ratio_mtf = atr_M5[1] / atr_H1[1]
   double atr_h1 = atr_h1_buf[1]; // derniere H1 cloturee
   features[1] = (float)(atr_h1 > 0 ? atr1 / atr_h1 : 0.0);

   // 3. keltner_distance = (close - band) / atr
   if(is_long)
      features[2] = (float)((close1 - upper1) / atr1);
   else
      features[2] = (float)((lower1 - close1) / atr1);

   // 4. rsi_14
   features[3] = (float)rsi_buf[0];

   // 5. momentum_5 = (close[1] - close[6]) / atr[1]
   double close6 = iClose(_Symbol, PERIOD_M5, 6);
   features[4] = (float)(atr1 > 0 ? (close1 - close6) / atr1 : 0.0);

   // 6. session_id (heures UTC)
   //    0=asian (00:00-07:45), 1=london (08:00-10:30),
   //    2=overlap (13:00-16:30), 3=off_session
   MqlDateTime gmt;
   TimeGMT(gmt);
   int hhmm = gmt.hour * 100 + gmt.min;
   int session_id;
   if     (hhmm >= 0    && hhmm <= 745)  session_id = 0;
   else if(hhmm >= 800  && hhmm <= 1030) session_id = 1;
   else if(hhmm >= 1300 && hhmm <= 1630) session_id = 2;
   else                                   session_id = 3;
   features[5] = (float)session_id;

   // 7. ema_slope_h1 = slope EMA H1 (5 barres) / atr_H1
   //    ema_h1_buf[0]=recent, ema_h1_buf[4]=ancien
   double slope = (ema_h1_buf[0] - ema_h1_buf[4]) / 5.0;
   features[6] = (float)(atr_h1 > 0 ? slope / atr_h1 : 0.0);

   // 8. vol_ratio_20 = std(close[1..20]) / mean(close[1..20])
   double closes20[20];
   double sum20 = 0;
   for(int k = 1; k <= 20; k++) { closes20[k-1] = iClose(_Symbol, PERIOD_M5, k); sum20 += closes20[k-1]; }
   double mean20 = sum20 / 20.0;
   double var20  = 0;
   for(int k = 0; k < 20; k++) var20 += (closes20[k] - mean20) * (closes20[k] - mean20);
   double std20 = MathSqrt(var20 / 20.0);
   features[7] = (float)(mean20 > 0 ? std20 / mean20 : 0.0);

   // --- Inference ONNX ---
   long  out_label[1];
   float out_proba[2]; // [proba_class0, proba_class1]

   if(!OnnxRun(onnx_model, ONNX_DEFAULT, features, out_label, out_proba))
     {
      Print("[Nova ML] Erreur OnnxRun: ", GetLastError());
      return;
     }

   float proba = out_proba[1]; // probabilite TP (class 1)

   // --- Dynamic trailing based on ADX regime ---
   // ADX > 25 = trending (trailing large), ADX < 20 = range (trailing serré)
   double dynamic_trailing_mult;
   double dynamic_breakeven_mult;

   if(adx >= InpADXHighThreshold)
     {
      dynamic_trailing_mult = InpTrailingTrending;    // 2.0x ATR
      dynamic_breakeven_mult = 1.5;                    // BE activé plus tard
      PrintFormat("[Nova ML] Régime: TENDANCE (ADX=%.1f)", adx);
     }
   else if(adx <= InpADXLowThreshold)
     {
      dynamic_trailing_mult = InpTrailingRange;       // 0.75x ATR
      dynamic_breakeven_mult = 0.5;                    // BE activé tôt
      PrintFormat("[Nova ML] Régime: RANGE (ADX=%.1f)", adx);
     }
   else
     {
      // Zone intermédiaire - interpolation linéaire
      double ratio = (adx - InpADXLowThreshold) / (InpADXHighThreshold - InpADXLowThreshold);
      dynamic_trailing_mult = InpTrailingRange + ratio * (InpTrailingTrending - InpTrailingRange);
      dynamic_breakeven_mult = 0.5 + ratio * (1.5 - 0.5);
      PrintFormat("[Nova ML] Régime: TRANSITION (ADX=%.1f)", adx);
     }

   PrintFormat("[Nova ML] Signal %s | proba=%.3f | seuil=%.2f | Trailing mult=%.2f",
               is_long ? "LONG" : "SHORT", proba, InpMLThreshold, dynamic_trailing_mult);

   if(proba < InpMLThreshold)
     {
      PrintFormat("[Nova ML] Signal filtre (proba=%.3f < %.2f)", proba, InpMLThreshold);
      last_bar_time = current_bar;
      return;
     }

   // --- Execution avec paramètres dynamiques ---
   if(is_long)
     {
      double sl = sym.Ask() - InpSLMultiplier * atr1;
      if(trade.Buy(InpLotSize, _Symbol, sym.Ask(), sl, 0, "Nova ML LONG"))
         PrintFormat("[Nova ML] BUY @ %.2f | SL=%.2f | Trailing=%.2fxATR | proba=%.3f",
                     sym.Ask(), sl, dynamic_trailing_mult, proba);
     }
   else
     {
      double sl = sym.Bid() + InpSLMultiplier * atr1;
      if(trade.Sell(InpLotSize, _Symbol, sym.Bid(), sl, 0, "Nova ML SHORT"))
         PrintFormat("[Nova ML] SELL @ %.2f | SL=%.2f | Trailing=%.2fxATR | proba=%.3f",
                     sym.Bid(), sl, dynamic_trailing_mult, proba);
     }

   last_bar_time = current_bar;
  }

//+------------------------------------------------------------------+
//| Trailing Stop (identique NovaGold_Baseline)                      |
//+------------------------------------------------------------------+
void ManageTrailingStops()
  {
   if(!pos.Select(_Symbol)) return;

   double open_price = pos.PriceOpen();
   double current_sl = pos.StopLoss();

   // Refresh ADX for dynamic regime detection
   double adx_arr[];
   ArraySetAsSeries(adx_arr, true);
   if(CopyBuffer(handle_adx, 0, 0, 1, adx_arr) <= 0) return;
   double cur_adx = adx_arr[0];

   double atr_arr[];
   ArraySetAsSeries(atr_arr, true);
   if(CopyBuffer(handle_atr, 0, 0, 1, atr_arr) <= 0) return;
   double cur_atr = atr_arr[0];

   // Dynamic multipliers based on ADX regime
   double be_dist;
   double trail_dist;

   if(cur_adx >= InpADXHighThreshold)
     {
      // Trending market - wider trailing, later breakeven
      trail_dist = InpTrailingTrending * cur_atr;
      be_dist = 1.5 * cur_atr;
     }
   else if(cur_adx <= InpADXLowThreshold)
     {
      // Range market - tighter trailing, earlier breakeven
      trail_dist = InpTrailingRange * cur_atr;
      be_dist = 0.5 * cur_atr;
     }
   else
     {
      // Transition zone - linear interpolation
      double ratio = (cur_adx - InpADXLowThreshold) / (InpADXHighThreshold - InpADXLowThreshold);
      trail_dist = (InpTrailingRange + ratio * (InpTrailingTrending - InpTrailingRange)) * cur_atr;
      be_dist = (0.5 + ratio * (1.5 - 0.5)) * cur_atr;
     }

   if(pos.PositionType() == POSITION_TYPE_BUY)
     {
      double pnl = sym.Bid() - open_price;
      if(pnl >= be_dist)
        {
         double new_sl = sym.Bid() - trail_dist;
         if(new_sl < open_price) new_sl = open_price;
         if(new_sl > current_sl && (sym.Bid() - new_sl) >= sym.StopsLevel() * sym.Point())
            trade.PositionModify(pos.Ticket(), new_sl, pos.TakeProfit());
        }
     }
   else if(pos.PositionType() == POSITION_TYPE_SELL)
     {
      double pnl = open_price - sym.Ask();
      if(pnl >= be_dist)
        {
         double new_sl = sym.Ask() + trail_dist;
         if(new_sl > open_price || new_sl == 0) new_sl = open_price;
         if((new_sl < current_sl || current_sl == 0) && (new_sl - sym.Ask()) >= sym.StopsLevel() * sym.Point())
            trade.PositionModify(pos.Ticket(), new_sl, pos.TakeProfit());
        }
     }
  }

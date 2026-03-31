# NovaGold Reborn — Regime-Aware Breakout Filter pour XAU/USD

> Pipeline de trading ML pour l'or, du backtest au live MT5.
> Architecture en 3 couches : Régime → Signal → Filtre ML.

---

## Vision du projet

Construire un système de day trading semi-automatique sur XAU/USD qui :

- Détecte les breakouts via Keltner Channel
- Filtre les faux signaux avec un modèle LightGBM
- S'adapte automatiquement aux sessions (SGE, London, NY)
- Exécute sur MT5 avec un money management basé sur l'ATR
- Produit des résultats de backtest **réalistes** (commissions réelles, lots = $, résolution tick par tick)

---

## Architecture finale

```
┌─────────────────────────────────────────────────────────────────┐
│                    COUCHE 1 — RÉGIME (zéro ML)                  │
│                                                                 │
│  Filtre Temporel            Filtre Énergie                      │
│  ├── SGE Open 01:30-03:00   ATR(14) M5 / ATR(14) H1            │
│  ├── London    08:00-10:30  → Si ratio < seuil : marché mort   │
│  └── Overlap  12:00-16:30   → On ignore tous les signaux       │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                COUCHE 2 — SIGNAL (règles dures)                 │
│                                                                 │
│  Keltner Channel Breakout                                       │
│  ├── Prix casse bande haute → Signal LONG candidat              │
│  ├── Prix casse bande basse → Signal SHORT candidat             │
│  └── Paramètres : EMA(20), ATR(14), multiplicateur 2.0          │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│              COUCHE 3 — FILTRE ML (LightGBM)                    │
│                                                                 │
│  Input Features (~8-10) :                                       │
│  ├── tick_volume_ratio   (ticks bougie / moyenne 20 bougies)    │
│  ├── atr_ratio_mtf       (ATR M5 / ATR H1)                     │
│  ├── keltner_distance    (prix - bande au moment du breakout)   │
│  ├── rsi_14              (RSI normalisé)                        │
│  ├── momentum_5          (rendement 5 dernières bougies)        │
│  ├── session_id          (0=SGE, 1=London, 2=Overlap, 3=Autre) │
│  ├── ema_slope_h1        (pente EMA H1 normalisée)              │
│  └── bb_width_m15        (largeur Bollinger normalisée M15)     │
│                                                                 │
│  Output : proba_go ∈ [0, 1]                                     │
│  Règle : trade SI proba_go > seuil (défaut 0.62)                │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                  SORTIE — Trailing ATR                           │
│                                                                 │
│  SL Initial     : 2.0 × ATR(14) M5                             │
│  Breakeven      : activé à 1.0 × ATR(14) M5 de profit          │
│  Trailing Stop  : suit à 1.5 × ATR(14) M5 du prix              │
│  TP Fixe        : aucun (le trailing gère la sortie)            │
│  Max positions  : 1                                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Roadmap d'implémentation

Le projet se construit **étape par étape**. Chaque phase est autonome et testable.

### Phase 0 — Infrastructure & données (cette phase)

```
[x] Structure du projet
[x] README & architecture
[ ] Script d'export MT5 → Parquet (ticks + bougies M1/M5)
[ ] CLI/TUI avec commandes simples
[ ] Chargement données chunk par chunk (RAM-safe)
```

### Phase 1 — Stratégie Keltner Breakout simple (sans ML)

```
[ ] Calcul Keltner Channel sur bougies M5
[ ] Détection des breakouts (cassure bande haute/basse)
[ ] Simulation SL/TP/Trailing avec résolution tick par tick
[ ] Backtester réaliste (commissions, spread, lots en $)
[ ] Rapport de performance (PF, winrate, drawdown, equity curve)
[ ] Optimisation paramètres (grid search / Optuna)
```

### Phase 2 — Filtres de régime (heuristiques, sans ML)

```
[ ] Filtre session temporelle (SGE / London / Overlap)
[ ] Filtre énergie ATR multi-timeframe
[ ] Comparaison performance avec/sans filtres
[ ] Ajustement des fenêtres de session
```

### Phase 3 — Feature engineering & labeling

```
[ ] Construction des features multi-timeframe
[ ] Labeling Triple Barrier (résolu tick par tick)
[ ] Walk-forward split (8 sem train / 2 sem test)
[ ] Analyse de corrélation des features
```

### Phase 4 — Modèle LightGBM

```
[ ] Entraînement classificateur binaire (GO / NO-GO)
[ ] Walk-forward backtest complet
[ ] Optimisation hyperparamètres (Optuna)
[ ] Comparaison Phase 1 vs Phase 1+2 vs Phase 1+2+4
[ ] Pipeline de réentraînement automatique
```

### Phase 5 — Connexion MT5 live

```
[ ] Bridge Python ↔ MT5 (lib MetaTrader5)
[ ] Exécution des signaux en paper trading
[ ] Money management ATR dynamique
[ ] Monitoring & alertes (Telegram/Discord)
```

### Phase 6 — Enrichissements optionnels

```
[ ] Score KNN Lorentzien comme feature supplémentaire
[ ] Intégration TradingAgents comme biais directionnel
[ ] Multi-paires (EURUSD, BTCUSD)
[ ] Dashboard web temps réel
```

---

## Structure du projet

```
novagold-reborn/
│
├── README.md
├── pyproject.toml
├── .env.example                 # MT5_LOGIN, MT5_PASSWORD, MT5_SERVER
│
├── cli/
│   ├── __init__.py
│   ├── app.py                   # Point d'entrée CLI (Typer + Rich)
│   ├── commands/
│   │   ├── data.py              # Commandes d'export/import données
│   │   ├── backtest.py          # Commandes de backtest
│   │   ├── optimize.py          # Commandes d'optimisation
│   │   ├── train.py             # Commandes ML (Phase 4)
│   │   └── live.py              # Commandes live trading (Phase 5)
│   └── display.py               # Formatage Rich (tableaux, progress bars)
│
├── core/
│   ├── __init__.py
│   ├── config.py                # Configuration centralisée
│   ├── types.py                 # Dataclasses (Trade, Signal, Position)
│   └── constants.py             # Sessions, symboles, valeurs par défaut
│
├── data/
│   ├── __init__.py
│   ├── mt5_export.py            # Export MT5 → Parquet (ticks + bougies)
│   ├── loader.py                # Chargement chunk par chunk (RAM-safe)
│   └── parquet/                 # Données exportées (gitignore)
│       ├── ticks/
│       │   ├── XAUUSD_ticks_2025-01.parquet
│       │   └── ...
│       └── candles/
│           ├── XAUUSD_M5_2025.parquet
│           └── ...
│
├── strategy/
│   ├── __init__.py
│   ├── keltner.py               # Calcul Keltner Channel
│   ├── breakout.py              # Détection des signaux de breakout
│   ├── regime.py                # Filtres de régime (session + ATR)
│   └── exits.py                 # Logique SL/Breakeven/Trailing (ATR)
│
├── backtest/
│   ├── __init__.py
│   ├── engine.py                # Moteur de backtest (chunk par chunk)
│   ├── resolver.py              # Résolution tick par tick du SL/TP
│   ├── metrics.py               # Calcul PF, Sharpe, drawdown, etc.
│   ├── report.py                # Génération rapport (Rich + export)
│   └── costs.py                 # Modèle de coûts (spread, commission, swap)
│
├── ml/                          # Phase 4
│   ├── __init__.py
│   ├── features.py              # Feature engineering multi-TF
│   ├── labeling.py              # Triple Barrier labeling
│   ├── trainer.py               # Walk-forward LightGBM
│   ├── optimizer.py             # Optuna hyperparameter tuning
│   └── models/                  # Modèles sauvegardés (gitignore)
│
├── live/                        # Phase 5
│   ├── __init__.py
│   ├── mt5_bridge.py            # Connexion MT5 + exécution ordres
│   ├── signal_loop.py           # Boucle de trading temps réel
│   └── monitor.py               # Monitoring + alertes
│
├── tests/
│   ├── test_keltner.py
│   ├── test_backtest.py
│   ├── test_resolver.py
│   └── test_costs.py
│
└── scripts/
    ├── export_mt5_data.py       # Script standalone d'export
    └── quick_backtest.py        # Backtest rapide one-liner
```

---

## CLI — Commandes disponibles

Le CLI utilise **Typer** (commandes simples) + **Rich** (affichage propre).

### Données

```bash
# Exporter les ticks depuis MT5 vers parquet
nova data export --symbol XAUUSD --type ticks --start 2025-01-01 --end 2025-12-31

# Exporter les bougies M1 et M5
nova data export --symbol XAUUSD --type candles --timeframe M5 --start 2025-01-01

# Vérifier les données disponibles
nova data info

# Exemple de sortie :
# ┌──────────┬───────────┬────────────┬────────────┬──────────┐
# │ Symbole  │ Type      │ Début      │ Fin        │ Taille   │
# ├──────────┼───────────┼────────────┼────────────┼──────────┤
# │ XAUUSD   │ ticks     │ 2025-01-02 │ 2025-12-31 │ 2.3 GB   │
# │ XAUUSD   │ M5        │ 2025-01-02 │ 2025-12-31 │ 45 MB    │
# │ XAUUSD   │ M1        │ 2025-01-02 │ 2025-06-30 │ 180 MB   │
# └──────────┴───────────┴────────────┴────────────┴──────────┘
```

### Backtest

```bash
# Backtest simple avec paramètres par défaut
nova backtest run --symbol XAUUSD --start 2025-06-01 --end 2025-12-31

# Backtest avec paramètres custom
nova backtest run \
  --symbol XAUUSD \
  --start 2025-01-01 \
  --end 2025-12-31 \
  --lot 0.01 \
  --commission 3.50 \
  --spread 0.20 \
  --keltner-period 20 \
  --keltner-mult 2.0 \
  --sl-atr-mult 2.0 \
  --trailing-atr-mult 1.5 \
  --breakeven-atr-mult 1.0

# Backtest avec filtres de régime (Phase 2)
nova backtest run --symbol XAUUSD --regime-filter --sessions SGE,OVERLAP

# Backtest comparatif (avec et sans filtres)
nova backtest compare --symbol XAUUSD --start 2025-01-01 --end 2025-12-31

# Exemple de sortie du backtest :
# ╭──────────────────── Backtest XAUUSD ─────────────────────╮
# │                                                           │
# │  Période     : 2025-06-01 → 2025-12-31 (130 jours)       │
# │  Capital     : $10,000.00                                 │
# │  Lot size    : 0.01 (1$/point)                            │
# │  Commission  : $3.50/lot round-turn                       │
# │  Spread moyen: $0.20                                      │
# │                                                           │
# │  ─── Résultats ───                                        │
# │  Trades total     : 187                                   │
# │  Trades gagnants  : 89 (47.6%)                            │
# │  Trades perdants  : 98 (52.4%)                            │
# │  Gain moyen       : +$8.34                                │
# │  Perte moyenne    : -$4.12                                │
# │  Profit Factor    : 1.64                                  │
# │  Espérance/trade  : +$1.82                                │
# │  Profit net       : +$340.34                              │
# │  Max Drawdown     : -$127.50 (1.27%)                      │
# │  Sharpe Ratio     : 1.42                                  │
# │  Commissions payées: -$65.45                              │
# │                                                           │
# │  ─── Par session ───                                      │
# │  SGE Open    : 31 trades | WR 51.6% | PF 1.89            │
# │  London      : 52 trades | WR 48.1% | PF 1.58            │
# │  Overlap     : 78 trades | WR 46.2% | PF 1.61            │
# │  Hors session: 26 trades | WR 42.3% | PF 0.94            │
# │                                                           │
# ╰───────────────────────────────────────────────────────────╯
```

### Optimisation

```bash
# Grid search sur les paramètres Keltner + sortie
nova optimize grid \
  --symbol XAUUSD \
  --start 2025-01-01 \
  --end 2025-09-30 \
  --param keltner_period:15,20,25,30 \
  --param keltner_mult:1.5,2.0,2.5,3.0 \
  --param sl_atr_mult:1.5,2.0,2.5 \
  --param trailing_atr_mult:1.0,1.5,2.0

# Optimisation Optuna (plus intelligent, moins de combinaisons)
nova optimize optuna \
  --symbol XAUUSD \
  --start 2025-01-01 \
  --end 2025-09-30 \
  --trials 200

# Walk-forward optimization (paramètres évolutifs)
nova optimize walkforward \
  --symbol XAUUSD \
  --start 2025-01-01 \
  --end 2025-12-31 \
  --train-weeks 12 \
  --test-weeks 4
```

### ML (Phase 4)

```bash
# Construire les features + labels
nova train prepare --symbol XAUUSD --start 2025-01-01 --end 2025-12-31

# Entraîner le modèle walk-forward
nova train run --symbol XAUUSD --train-weeks 8 --test-weeks 2

# Backtest avec le modèle ML
nova backtest run --symbol XAUUSD --use-ml --model latest --threshold 0.62
```

### Live (Phase 5)

```bash
# Paper trading (simulation temps réel sur MT5 demo)
nova live paper --symbol XAUUSD

# Live trading
nova live start --symbol XAUUSD --lot 0.01 --max-risk 1.0
```

---

## Spécifications du backtester

### Principe : reproduire MT5 exactement

Le backtester doit produire des résultats **comparables** à un backtest MT5 Strategy Tester
sur la même période avec les mêmes données. Pour cela :

### Conversion lots → dollars

```
XAU/USD :
  1 lot standard = 100 oz → 1 point ($0.01) = $1.00
  0.1 lot        = 10 oz  → 1 point = $0.10
  0.01 lot       = 1 oz   → 1 point = $0.01

  Donc pour 0.01 lot :
    Mouvement de $5.00 (500 points) = $5.00 de P/L
    Mouvement de $1.00 (100 points) = $1.00 de P/L
```

### Modèle de coûts

```python
@dataclass
class CostModel:
    spread: float = 0.20        # Spread moyen en $ (= 20 points)
    commission_per_lot: float = 3.50  # Commission round-turn par lot standard
    swap_long: float = -0.50    # Swap journalier par lot (si position overnight)
    swap_short: float = 0.30    # Swap journalier par lot
    slippage: float = 0.05      # Slippage moyen en $ (= 5 points)

    # Le spread est appliqué à l'entrée (ask - bid)
    # La commission est proportionnelle à la taille du lot
    # Le slippage est ajouté aléatoirement (uniform 0 à slippage)
```

### Résolution tick par tick

Pour chaque trade ouvert, le backtester :

1. Cherche l'entrée exacte sur la bougie de breakout
2. Charge les ticks à partir de ce moment
3. Simule tick par tick : SL touché ? Breakeven activé ? Trailing mis à jour ?
4. Enregistre le prix de sortie exact et le P/L

Si les ticks ne sont pas disponibles pour une période, fallback sur bougies M1
avec simulation OHLC (open → high/low selon direction → close).

### Traitement chunk par chunk

```python
# Les données sont chargées mois par mois
# Jamais plus d'1 mois de ticks en RAM
for chunk in loader.iter_months("XAUUSD", start, end):
    signals = strategy.detect_breakouts(chunk.candles_m5)
    for signal in signals:
        ticks = chunk.get_ticks(signal.time, signal.time + timedelta(hours=4))
        trade = resolver.resolve(signal, ticks, cost_model)
        results.append(trade)
    del chunk  # Libère la RAM
```

---

## Configuration

Fichier `config.yaml` à la racine (ou arguments CLI) :

```yaml
# === Symbole ===
symbol: XAUUSD
point_value: 0.01         # 1 point = $0.01
lot_size: 0.01            # Taille de position par défaut
contract_size: 100        # 1 lot = 100 oz

# === Keltner Channel ===
keltner:
  ema_period: 20
  atr_period: 14
  multiplier: 2.0

# === Sorties ===
exits:
  sl_atr_mult: 2.0        # SL = 2.0 × ATR(14) M5
  breakeven_atr_mult: 1.0  # BE activé à 1.0 × ATR de profit
  trailing_atr_mult: 1.5   # Trailing suit à 1.5 × ATR
  trailing_step_points: 10 # Pas minimum de mise à jour

# === Régime (Phase 2) ===
regime:
  use_session_filter: true
  sessions:
    sge_open:
      start: "01:30"       # UTC
      end: "03:00"
    london:
      start: "08:00"
      end: "10:30"
    overlap:
      start: "12:00"
      end: "16:30"
  atr_ratio_threshold: 0.15  # ATR M5/H1 minimum

# === Coûts ===
costs:
  spread: 0.20             # $ (modifiable selon ton broker)
  commission_per_lot: 3.50  # $ round-turn (Raw account)
  slippage: 0.05            # $
  swap_long: -0.50          # $/lot/jour
  swap_short: 0.30

# === Backtest ===
backtest:
  initial_balance: 10000
  chunk_size: "1M"          # Charger 1 mois à la fois
  tick_resolution: true     # Résoudre SL/TP tick par tick
  max_positions: 1

# === ML (Phase 4) ===
ml:
  proba_threshold: 0.62
  train_weeks: 8
  test_weeks: 2
  step_weeks: 2
  model_params:
    n_estimators: 400
    max_depth: 5
    num_leaves: 20
    learning_rate: 0.02
    min_child_samples: 300
    subsample: 0.7
    colsample_bytree: 0.6

# === Live (Phase 5) ===
live:
  mt5_login: ${MT5_LOGIN}
  mt5_password: ${MT5_PASSWORD}
  mt5_server: ${MT5_SERVER}
  max_risk_percent: 1.0
```

---

## Dépendances

```
# Core
python = "^3.11"
pandas = "^2.2"
numpy = "^1.26"
pyarrow = "^15.0"          # Lecture/écriture Parquet

# CLI
typer = "^0.12"
rich = "^13.7"

# Backtest
numba = "^0.60"            # Accélération résolution tick par tick

# Optimisation
optuna = "^3.6"

# ML (Phase 4)
lightgbm = "^4.3"
scikit-learn = "^1.4"

# Live (Phase 5)
MetaTrader5 = "^5.0"       # Windows only (ou via bridge)

# Optionnel
plotly = "^5.20"            # Graphiques interactifs
```

---

## Export MT5 → Parquet

Script d'export à lancer depuis un PC Windows avec MT5 installé :

```bash
# Exporter les ticks (gros fichiers, faire par mois)
nova data export --symbol XAUUSD --type ticks \
  --start 2025-01-01 --end 2025-01-31 \
  --output data/parquet/ticks/

# Exporter les bougies M1 (plus léger)
nova data export --symbol XAUUSD --type candles --timeframe M1 \
  --start 2025-01-01 --end 2025-12-31

# Exporter les bougies M5
nova data export --symbol XAUUSD --type candles --timeframe M5 \
  --start 2025-01-01 --end 2025-12-31
```

Le format Parquet de sortie pour les ticks :

```
Colonnes : time (datetime64[ms]), bid (float64), ask (float64),
           last (float64), volume (int32), flags (int8)
```

Le format Parquet pour les bougies :

```
Colonnes : time (datetime64[s]), open, high, low, close (float64),
           tick_volume (int32), spread (int16), real_volume (int64)
```

---

## Priorité d'implémentation

```
Semaine 1-2 : Phase 0
  → Structure projet + CLI squelette + export MT5 + loader chunk

Semaine 3-4 : Phase 1
  → Keltner + breakout + backtester simple (bougies M5)
  → Résolution tick par tick du trailing
  → Rapport Rich dans le terminal

Semaine 5 : Phase 2
  → Filtres session + ATR ratio
  → Comparaison avec/sans filtres

Semaine 6-7 : Phase 3 + 4
  → Features + labeling + LightGBM walk-forward
  → Backtest ML complet

Semaine 8+ : Phase 5
  → Bridge MT5 + paper trading + monitoring
```

---

## Notes importantes

- **0.01 lot = 1$ par point sur l'or** — le backtester affiche tout en dollars réels
- Les commissions sont configurables pour matcher ton broker exact (Raw/ECN)
- Le spread peut être fixe OU lu depuis les ticks (bid-ask)
- Le backtest chunk par chunk ne charge jamais plus d'1 mois de ticks en RAM
- Walk-forward strict : le modèle ne voit JAMAIS les données futures
- Tout est en UTC — les sessions sont définies en UTC dans le config
- Les résultats doivent être comparables à MT5 Strategy Tester (±5% de marge)

---

## Licence

Usage personnel uniquement. Ce projet est un outil de recherche.
Le trading comporte des risques de perte en capital.

"""
Interface Utilisateur Textuelle (TUI) pour NovaGold Reborn.
Permet d'exécuter et de configurer facilement les commandes du projet
sans avoir à retenir les lignes de commande complexes.
"""

import sys
import os
import subprocess
import questionary
from rich.console import Console

console = Console()

def run_cmd(*args):
    """Exécute la commande de façon asynchrone pour l'affichage temps réel."""
    cmd_line = [sys.executable] + list(args)
    console.print(f"\n[bold black on white] Exécution: {' '.join(cmd_line)} [/bold black on white]\n")
    try:
        subprocess.run(cmd_line, check=True)
    except subprocess.CalledProcessError as e:
        console.print(f"\n[bold red]❌ La commande a échoué (Code {e.returncode})[/bold red]")
    except KeyboardInterrupt:
        console.print("\n[bold yellow]⚠️ Interrompu par l'utilisateur.[/bold yellow]")
    
    # Pause avant retour au menu
    input("\n➡ Appuyez sur Entrée pour retourner au menu...")


def ask_param(label: str, default) -> str:
    """Demande un paramètre numérique avec valeur par défaut affichée."""
    return questionary.text(f"  {label} :", default=str(default)).ask()


def menu_backtest():
    action = questionary.select(
        "🔎 Backtesting - Que voulez-vous lancer ?",
        choices=[
            "1. Comparatif Complet (Keltner vs Filtres — parallèle ⚡)",
            "2. Backtest avec Configuration Complète",
            "🔙 Retour"
        ]
    ).ask()

    if not action or action == "🔙 Retour":
        return

    # ── Période & Timeframe ──────────────────────────────────────────
    console.print("\n[bold cyan]── Période & Timeframe ──[/bold cyan]")
    tf_choice = questionary.select(
        "  Timeframe des signaux :",
        choices=["M1 (1 minute)", "M5 (5 minutes — défaut)", "M15 (15 minutes)", "M30 (30 minutes)"],
        default="M5 (5 minutes — défaut)"
    ).ask()
    timeframe = tf_choice.split(" ")[0]
    start = questionary.text("  Date de début (YYYY-MM-DD) :", default="2025-01-01").ask()
    end   = questionary.text("  Date de fin   (YYYY-MM-DD) :", default="2025-12-31").ask()

    if "Comparatif" in action:
        run_cmd("scripts/run_backtest_comparison.py",
                "--start", start, "--end", end, "--timeframe", timeframe)
        return

    # ── Keltner ─────────────────────────────────────────────────────
    console.print("\n[bold cyan]── Keltner Channel ──[/bold cyan]")
    keltner_period = ask_param("Période EMA", 20)
    keltner_mult   = ask_param("Multiplicateur ATR", 2.0)

    # ── Gestion du risque ────────────────────────────────────────────
    console.print("\n[bold cyan]── Gestion du Risque ──[/bold cyan]")
    sl_atr_mult        = ask_param("Stop Loss     (× ATR)", 1.0)
    breakeven_atr_mult = ask_param("Breakeven     (× ATR)", 0.5)
    trailing_atr_mult  = ask_param("Trailing Stop (× ATR)", 0.75)

    # ── Capital & Lot ────────────────────────────────────────────────
    console.print("\n[bold cyan]── Capital & Lot ──[/bold cyan]")
    balance  = ask_param("Capital initial ($)", 100)
    lot_size = ask_param("Taille de lot", 0.01)

    # ── Filtres de session ───────────────────────────────────────────
    console.print("\n[bold cyan]── Filtres de Session ──[/bold cyan]")
    session_filter = questionary.confirm(
        "  Activer le filtre de session ?", default=True
    ).ask()

    sessions_selected = ["asian", "london", "overlap"]
    if session_filter:
        sessions_selected = questionary.checkbox(
            "  Sessions autorisées :",
            choices=["asian", "london", "overlap"],
            default=["asian", "london", "overlap"],
        ).ask() or ["asian", "london", "overlap"]

    # ── Filtre ATR ratio ─────────────────────────────────────────────
    console.print("\n[bold cyan]── Filtre ATR Ratio (M5/H1) ──[/bold cyan]")
    atr_filter = questionary.confirm(
        "  Activer le filtre ATR ratio ? (nécessite données H1)", default=False
    ).ask()
    atr_threshold = 0.15
    if atr_filter:
        atr_threshold = float(ask_param("Seuil ATR ratio min", 0.15))

    # ── Filtre ML ────────────────────────────────────────────────────
    console.print("\n[bold cyan]── Filtre Machine Learning ──[/bold cyan]")
    ml_filter = False
    ml_threshold = 0.55
    model_path  = f"ml/models/model_{timeframe}.joblib"
    legacy_path = "ml/models/model.joblib"
    if os.path.exists(model_path) or (timeframe == "M5" and os.path.exists(legacy_path)):
        ml_filter = questionary.confirm(
            f"  Activer le filtre ML ? (modèle {timeframe} détecté)", default=False
        ).ask()
        if ml_filter:
            threshold_choice = questionary.select(
                "  Seuil de probabilité ML :",
                choices=["0.50  (permissif)", "0.52", "0.55  (défaut)", "0.58", "0.60  (strict)"],
                default="0.55  (défaut)",
            ).ask()
            ml_threshold = float(threshold_choice.split()[0])
    else:
        console.print(f"  [yellow]Aucun modèle trouvé pour {timeframe} — filtre ML désactivé[/yellow]")

    # ── Options d'affichage ──────────────────────────────────────────
    console.print("\n[bold cyan]── Options ──[/bold cyan]")
    show_trades = questionary.confirm(
        "  Afficher chaque trade ? (désactiver si beaucoup de trades)", default=True
    ).ask()
    resolution_choice = questionary.select(
        "  Résolution de simulation :",
        choices=["auto (ticks → M1 → M5)", "ticks", "m1", "m5"],
        default="auto (ticks → M1 → M5)",
    ).ask()
    resolution = resolution_choice.split(" ")[0]

    # ── Construction commande ────────────────────────────────────────
    cmd = [
        "-m", "cli.app", "backtest", "run",
        "--start", start, "--end", end,
        "--timeframe", timeframe,
        "--keltner-period", keltner_period,
        "--keltner-mult",   keltner_mult,
        "--sl-atr-mult",    sl_atr_mult,
        "--breakeven-atr-mult", breakeven_atr_mult,
        "--trailing-atr-mult",  trailing_atr_mult,
        "--balance", balance,
        "--lot",     lot_size,
        "--resolution", resolution,
    ]
    if session_filter:
        cmd += ["--regime-filter", "--sessions", ",".join(sessions_selected)]
    if atr_filter:
        cmd += ["--atr-ratio-filter", "--atr-ratio-threshold", str(atr_threshold)]
    if ml_filter:
        cmd += ["--use-ml-filter", "--ml-threshold", str(ml_threshold)]
    if not show_trades:
        cmd.append("--no-show-trades")

    run_cmd(*cmd)


def menu_data():
    action = questionary.select(
        "📊 Données - Que voulez-vous faire ?",
        choices=[
            "1. Télécharger Bougies M1/M5/M15/H1",
            "2. Télécharger les Ticks bruts (Très lourd)",
            "3. Reconstruire Bougies depuis Ticks (sans MT5)",
            "4. Vérifier les données locales",
            "🔙 Retour"
        ]
    ).ask()

    if not action or action == "🔙 Retour":
        return

    if "Vérifier" in action:
        run_cmd("-m", "cli.app", "data", "verify")
        return

    if "Reconstruire" in action:
        tf_choice = questionary.select(
            "Timeframe à reconstruire depuis les ticks :",
            choices=["M1", "M5", "M15", "M30", "H1"],
            default="M1"
        ).ask()
        console.print(f"[cyan]Reconstruction {tf_choice} depuis tous les fichiers tick disponibles...[/cyan]")
        run_cmd("-m", "cli.app", "data", "rebuild", "--timeframe", tf_choice)
        return

    start = questionary.text("Date de début (YYYY-MM-DD) :", default="2025-01-01").ask()
    end = questionary.text("Date de fin (YYYY-MM-DD) :", default="2025-12-31").ask()

    if "Bougies" in action:
        run_cmd("-m", "cli.app", "data", "export", "--type", "all", "--start", start, "--end", end, "--no-ticks")
    elif "Ticks" in action:
        run_cmd("-m", "cli.app", "data", "export", "--type", "ticks", "--start", start, "--end", end)


def menu_ml():
    action = questionary.select(
        "🧠 Machine Learning - Que faire ?",
        choices=[
            "1. Extraire les Features (Préparation)",
            "2. Entraîner le Modèle Walk-Forward",
            "🔙 Retour"
        ]
    ).ask()

    if not action or action == "🔙 Retour":
        return

    tf_choice = questionary.select(
        "Timeframe des signaux :",
        choices=["M1 (1 minute)", "M5 (5 minutes — défaut)", "M15 (15 minutes)", "M30 (30 minutes)"],
        default="M5 (5 minutes — défaut)"
    ).ask()
    timeframe = tf_choice.split(" ")[0]

    start = questionary.text("Date de début (YYYY-MM-DD) :", default="2025-01-01").ask()
    end = questionary.text("Date de fin (YYYY-MM-DD) :", default="2025-12-31").ask()

    if "Extraire" in action:
        run_cmd("-m", "cli.app", "train", "prepare",
                "--timeframe", timeframe, "--start", start, "--end", end)
    else:
        console.print(f"[cyan]Modèle sauvegardé dans ml/models/model_{timeframe}.joblib[/cyan]")
        run_cmd("-m", "cli.app", "train", "run",
                "--timeframe", timeframe, "--start", start, "--end", end)


def menu_optimize():
    cpu_count = os.cpu_count() or 4

    console.print(f"\n[cyan]💻 CPU détectés : {cpu_count} cœurs[/cyan]")
    console.print(f"[cyan]   Les backtests tourneront en parallèle ({cpu_count} workers max)[/cyan]\n")

    action = questionary.select(
        "🔧 Optimisation - Que voulez-vous faire ?",
        choices=[
            "1. Grid Search Rapide  (SL × Keltner — 12 combos)",
            "2. Grid Search Complet (SL × Keltner × Trailing — 48 combos)",
            "3. Grid Search Personnalisé",
            "🔙 Retour"
        ]
    ).ask()

    if not action or action == "🔙 Retour":
        return

    start = questionary.text("Date de début (YYYY-MM-DD) :", default="2025-01-01").ask()
    end = questionary.text("Date de fin (YYYY-MM-DD) :", default="2025-12-31").ask()

    metric = questionary.select(
        "Métrique à optimiser :",
        choices=[
            "profit_factor",
            "sharpe_ratio",
            "net_profit",
            "expectancy",
        ],
        default="profit_factor"
    ).ask()

    workers_input = questionary.text(
        f"Nombre de workers parallèles (Entrée = {cpu_count} cœurs détectés) :",
        default=str(cpu_count)
    ).ask()
    workers = int(workers_input) if workers_input else cpu_count

    cmd = ["-m", "cli.app", "optimize", "grid",
           "--start", start, "--end", end,
           "--metric", metric,
           "--workers", str(workers)]

    if "Rapide" in action:
        # SL: 0.5, 0.75, 1.0, 1.5  ×  Keltner: 1.5, 2.0, 2.5
        cmd += ["--param", "sl_atr_mult:0.5,0.75,1.0,1.5",
                "--param", "keltner_multiplier:1.5,2.0,2.5"]

    elif "Complet" in action:
        # SL × Keltner × Trailing × Breakeven
        cmd += ["--param", "sl_atr_mult:0.5,0.75,1.0,1.5",
                "--param", "keltner_multiplier:1.5,2.0,2.5",
                "--param", "trailing_atr_mult:0.5,0.75,1.0",
                "--param", "breakeven_atr_mult:0.25,0.5"]

    else:
        # Personnalisé — demander paramètre par paramètre
        console.print("\n[yellow]Format des valeurs : séparées par virgule, ex: 0.5,1.0,1.5[/yellow]")
        params_available = {
            "sl_atr_mult": "SL (ex: 0.5,1.0,1.5,2.0)",
            "keltner_multiplier": "Keltner mult (ex: 1.5,2.0,2.5)",
            "keltner_ema_period": "EMA période (ex: 14,20,30)",
            "trailing_atr_mult": "Trailing (ex: 0.5,0.75,1.0)",
            "breakeven_atr_mult": "Breakeven (ex: 0.25,0.5,0.75)",
        }
        for param_name, label in params_available.items():
            val = questionary.text(f"  {label} (laisser vide pour ignorer) :").ask()
            if val and val.strip():
                cmd += ["--param", f"{param_name}:{val.strip()}"]

    run_cmd(*cmd)


def menu_live():
    action = questionary.select(
        "🟢 Live Trading & Paper - Actions",
        choices=[
            "1. Vérifier le Statut du compte",
            "2. Lancer le Paper Trading (Mode Demo Sûr)",
            "3. Lancer le Live Trading (RÉEL ⚠️)",
            "🔙 Retour"
        ]
    ).ask()

    if not action or action == "🔙 Retour":
        return

    if "Statut" in action:
        run_cmd("-m", "cli.app", "live", "status")
    elif "Paper Trading" in action:
        console.print("[yellow]Le Paper Trading va utiliser vos infos MT5 dans le fichier .env et ne passera que des ordres simulés.[/yellow]")
        if questionary.confirm("Voulez-vous lancer le Paper Trading sur XAUUSD ?").ask():
            run_cmd("-m", "cli.app", "live", "paper")
    elif "Live Trading" in action:
        console.print("[red]⚠️ ATTENTION : LE MODE LIVE ENGAGE VOTRE CAPITAL SUR LE SERVEUR MT5.[/red]")
        validation = questionary.text("Écrivez 'JE COMPRENDS' pour continuer :").ask()
        if validation == "JE COMPRENDS":
            run_cmd("-m", "cli.app", "live", "start")
        else:
            console.print("Mode Live annulé.")


def main():
    while True:
        # Clear screen for better formatting
        os.system('cls' if os.name == 'nt' else 'clear')
        
        console.print("\n[bold gold1]===============================[/bold gold1]")
        console.print("[bold gold1]   ✨ NOVAGOLD REBORN TUI ✨[/bold gold1]")
        console.print("[bold gold1]===============================[/bold gold1]\n")
        
        cpu_count = os.cpu_count() or 4
        console.print(f"[dim]💻 {cpu_count} cœurs CPU disponibles[/dim]\n")

        action = questionary.select(
            "Sélectionnez une catégorie :",
            choices=[
                "📊 Exporter / Vérifier les Données",
                "🔎 Lancer un Backtest",
                "🔧 Optimisation Paramètres (Grid Search parallèle)",
                "🧠 Intelligence Artificielle (ML)",
                "🟢 Trading Live & Démo",
                "❌ Quitter"
            ]
        ).ask()

        if not action or action == "❌ Quitter":
            console.print("[green]Fermeture du TUI. Bon trading ![green]")
            break

        if "Données" in action:
            menu_data()
        elif "Backtest" in action:
            menu_backtest()
        elif "Optimisation" in action:
            menu_optimize()
        elif "Intelligence Artificielle" in action:
            menu_ml()
        elif "Trading Live" in action:
            menu_live()

if __name__ == "__main__":
    main()

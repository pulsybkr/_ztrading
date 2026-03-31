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


def menu_backtest():
    action = questionary.select(
        "🔎 Backtesting - Que voulez-vous lancer ?",
        choices=[
            "1. Comparatif Complet (Keltner vs Filtres)",
            "2. Backtest Baseline Simple (Keltner Pur)",
            "🔙 Retour"
        ]
    ).ask()

    if not action or action == "🔙 Retour":
        return

    start = questionary.text("Date de début (YYYY-MM-DD) :", default="2025-01-01").ask()
    end = questionary.text("Date de fin (YYYY-MM-DD) :", default="2025-12-31").ask()

    if "Comparatif" in action:
        run_cmd("scripts/run_backtest_comparison.py", "--start", start, "--end", end)
    else:
        tf_choice = questionary.select(
            "Timeframe des signaux :",
            choices=["M1 (1 minute)", "M5 (5 minutes — défaut)", "M15 (15 minutes)", "M30 (30 minutes)"],
            default="M5 (5 minutes — défaut)"
        ).ask()
        timeframe = tf_choice.split(" ")[0]  # "M5" depuis "M5 (5 minutes — défaut)"

        session_filter = questionary.confirm(
            "Activer le filtre de session ? (sge_open, london, overlap — recommandé)",
            default=True
        ).ask()

        ml_filter = False
        if timeframe == "M5":
            ml_filter = questionary.confirm(
                "Activer le filtre ML ? (nécessite un modèle entraîné dans ml/models/)",
                default=False
            ).ask()
        else:
            console.print(f"[yellow]Filtre ML désactivé (modèle entraîné sur M5 uniquement)[/yellow]")

        cmd = ["-m", "cli.app", "backtest", "run", "--start", start, "--end", end,
               "--timeframe", timeframe]
        if session_filter:
            cmd.append("--regime-filter")
        if ml_filter:
            cmd.append("--use-ml-filter")
        run_cmd(*cmd)


def menu_data():
    action = questionary.select(
        "📊 Données - Que voulez-vous télécharger ?",
        choices=[
            "1. Télécharger Bougies M1/M5/M15/H1",
            "2. Télécharger les Ticks bruts (Très lourd)",
            "3. Vérifier les données locales",
            "🔙 Retour"
        ]
    ).ask()

    if not action or action == "🔙 Retour":
        return

    if "Vérifier" in action:
        run_cmd("-m", "cli.app", "data", "verify")
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

    start = questionary.text("Date de début (YYYY-MM-DD) :", default="2025-01-01").ask()
    end = questionary.text("Date de fin (YYYY-MM-DD) :", default="2025-12-31").ask()

    if "Extraire" in action:
        run_cmd("-m", "cli.app", "train", "prepare", "--start", start, "--end", end)
    else:
        run_cmd("-m", "cli.app", "train", "run", "--start", start, "--end", end)


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
        
        action = questionary.select(
            "Sélectionnez une catégorie :",
            choices=[
                "📊 Exporter / Vérifier les Données",
                "🔎 Lancer un Backtest",
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
        elif "Intelligence Artificielle" in action:
            menu_ml()
        elif "Trading Live" in action:
            menu_live()

if __name__ == "__main__":
    main()

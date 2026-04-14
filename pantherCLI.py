import sys
from pathlib import Path
import argparse
import torch
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

try:
    from model import load_model, predict_top_k
except ImportError:
    print("Errore: assicurati che 'model.py' sia nella stessa cartella.")
    sys.exit(1)

console = Console()


def parse_fasta(text: str) -> str:
    """Pulisce l'input rimuovendo header FASTA e spazi. Concatena tutte le sequenze."""
    lines = text.strip().splitlines()
    if not lines:
        return ""
    seq_lines = [l.strip() for l in lines if not l.startswith(">")]
    return "".join(seq_lines).replace(" ", "").upper()


def parse_fasta_first(text: str) -> tuple[str, str]:
    """
    Estrae solo la prima sequenza da un file FASTA multi-sequenza.

    Returns:
        (header, sequenza) della prima entry.
    """
    header = ""
    seq_lines = []
    in_first = False
    for line in text.strip().splitlines():
        line = line.strip()
        if line.startswith(">"):
            if in_first:
                break
            header = line[1:]
            in_first = True
        elif in_first:
            seq_lines.append(line)
    return header, "".join(seq_lines).replace(" ", "").upper()


def read_multiline_input() -> str:
    """Legge input multi-riga finche non arriva una riga vuota o EOF (Ctrl+D)."""
    console.print("[bold]Incolla la sequenza proteica (anche multi-riga).[/bold]")
    console.print("[dim]Premi Invio su una riga vuota per confermare (oppure Ctrl+D):[/dim]\n")
    lines = []
    try:
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
    except EOFError:
        pass
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="PANTHER Protein Classifier CLI")
    parser.add_argument(
        "--seq",
        type=Path,
        metavar="FILE",
        help="Sequenza proteica da file (raw o FASTA)",
    )
    parser.add_argument(
        "--first",
        action="store_true",
        help="Con --seq: usa solo la prima sequenza del file FASTA invece di concatenarle tutte",
    )
    parser.add_argument(
        "--weights",
        default="best_panther_model_pro.pth",
        help="Percorso file .pth (default: best_panther_model_pro.pth)",
    )
    parser.add_argument(
        "--classes",
        default="classes_pro.npy",
        help="Percorso file .npy (default: classes_pro.npy)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Numero di risultati da mostrare (default: 5)",
    )
    args = parser.parse_args()

    console.print("[bold cyan]PANTHER Classifier CLI v6[/bold cyan]\n")

    # --- Sorgente della sequenza ---
    header = ""
    if args.seq:
        if not args.seq.exists():
            console.print(f"[bold red]Errore:[/bold red] file non trovato: {args.seq}")
            sys.exit(1)
        raw = args.seq.read_text()
        if args.first:
            header, seq = parse_fasta_first(raw)
            if header:
                console.print(f"[dim]Sequenza selezionata:[/dim] {header}\n")
        else:
            seq = parse_fasta(raw)
    elif not sys.stdin.isatty():
        raw = sys.stdin.read()
        if args.first:
            header, seq = parse_fasta_first(raw)
        else:
            seq = parse_fasta(raw)
    else:
        raw = read_multiline_input()
        seq = parse_fasta(raw)

    if not seq:
        console.print("[bold red]Sequenza non valida o vuota.[/bold red]")
        sys.exit(1)

    # --- Caricamento modello e predizione ---
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task(description="Caricamento modello...", total=None)
            model, label_classes = load_model(args.weights, args.classes)

            progress.add_task(description="Elaborazione predizione...", total=None)
            results = predict_top_k(model, label_classes, seq, k=args.topk)
    except FileNotFoundError as e:
        console.print(f"[bold red]Errore:[/bold red] file non trovato: {e.filename}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Errore durante il caricamento o la predizione:[/bold red] {e}")
        sys.exit(1)

    # --- Output ---
    console.print(
        f"[bold green]Analisi completata.[/bold green] "
        f"Lunghezza sequenza: [yellow]{len(seq)} aa[/yellow]\n"
    )

    table = Table(title="Top Predizioni PANTHER", title_style="bold magenta")
    table.add_column("Rank", justify="right", style="dim")
    table.add_column("Famiglia", style="cyan", no_wrap=True)
    table.add_column("Confidenza", justify="right", style="green")
    table.add_column("Barra", width=20)

    for i, (name, prob) in enumerate(results, start=1):
        bar = "|" * int(prob / 5)
        table.add_row(str(i), name, f"{prob:.2f}%", f"[green]{bar}[/green]")

    console.print(table)

    top_name, top_prob = results[0]
    console.print(Panel(
        f"La proteina appartiene molto probabilmente alla famiglia:\n"
        f"[bold cyan]{top_name}[/bold cyan] con il "
        f"[bold green]{top_prob:.2f}%[/bold green] di confidenza.",
        title="Risultato Principale",
        border_style="green",
    ))


if __name__ == "__main__":
    main()

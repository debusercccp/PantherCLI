"""
panther_tui.py — Interfaccia TUI per Panther Protein Classifier v6.
"""

import sys
from pathlib import Path

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, Static, Button, Label
from textual.containers import Container, Vertical

from model import load_model, predict_top_k

# Percorsi di default (nella stessa directory dello script)
BASE_DIR = Path(__file__).parent
DEFAULT_WEIGHTS = BASE_DIR / "best_panther_model_pro.pth"
DEFAULT_CLASSES = BASE_DIR / "classes_pro.npy"


class PantherApp(App):
    CSS = """
    Screen { background: $background; }

    Container {
        margin: 2 4;
    }

    #title-label {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }

    Input {
        margin-bottom: 1;
    }

    Button {
        margin-bottom: 1;
    }

    #results {
        background: $panel;
        border: solid $primary;
        padding: 1;
        margin-top: 1;
        height: 12;
        overflow-y: auto;
    }

    #error-msg {
        color: $error;
        text-style: bold;
        height: 1;
    }
    """

    def __init__(self, weights_path: Path = DEFAULT_WEIGHTS, classes_path: Path = DEFAULT_CLASSES):
        super().__init__()
        self.weights_path = weights_path
        self.classes_path = classes_path
        self._model = None
        self._label_classes = None

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Vertical(
                Label("Incolla la sequenza proteica (singola lettera, raw o FASTA):", id="title-label"),
                Input(placeholder="Es: MKVLVVLLS..."),
                Button("Predici Famiglia", variant="primary", id="predict-btn"),
                Static("", id="error-msg"),
                Static(id="results"),
            )
        )
        yield Footer()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_mount(self) -> None:
        self.title = "PANTHER Classifier v6"
        self._load_model_safe()

    def _load_model_safe(self) -> None:
        """Carica modello e classi; mostra errore inline se i file mancano."""
        try:
            self._model, self._label_classes = load_model(
                str(self.weights_path), str(self.classes_path)
            )
        except FileNotFoundError as e:
            self._show_error(f"File non trovato: {e.filename}")
        except Exception as e:
            self._show_error(f"Errore caricamento modello: {e}")

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "predict-btn":
            return
        self._run_prediction()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Permette di premere Invio per predire."""
        self._run_prediction()

    # ------------------------------------------------------------------
    # Prediction logic
    # ------------------------------------------------------------------

    def _run_prediction(self) -> None:
        self._clear_error()

        if self._model is None:
            self._show_error("Modello non caricato. Controlla i file .pth e .npy.")
            return

        raw = self.query_one(Input).value.strip().upper()
        seq = self._parse_input(raw)

        if not seq:
            self._show_error("Sequenza vuota o non valida.")
            return

        try:
            results = predict_top_k(self._model, self._label_classes, seq, k=5)
        except Exception as e:
            self._show_error(f"Errore durante la predizione: {e}")
            return

        self._display_results(seq, results)

    @staticmethod
    def _parse_input(text: str) -> str:
        """Rimuove header FASTA e whitespace dalla sequenza."""
        lines = text.splitlines()
        seq_lines = [l for l in lines if not l.startswith(">")]
        return "".join(seq_lines).replace(" ", "")

    def _display_results(self, seq: str, results: list[tuple[str, float]]) -> None:
        header = f"[b]Sequenza:[/b] {seq[:40]}{'...' if len(seq) > 40 else ''} ([i]{len(seq)} aa[/i])\n\n"
        header += "[b]TOP 5 PREDIZIONI:[/b]\n\n"
        rows = ""
        for i, (name, prob) in enumerate(results, start=1):
            bar = "█" * int(prob / 5)  # barra proporzionale (max 20 blocchi)
            rows += f"{i}. [cyan]{name}[/cyan]\n   [green]{prob:.2f}%[/green] {bar}\n\n"
        self.query_one("#results").update(header + rows)

    def _show_error(self, msg: str) -> None:
        self.query_one("#error-msg").update(f"⚠ {msg}")

    def _clear_error(self) -> None:
        self.query_one("#error-msg").update("")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Panther Protein Classifier TUI")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Percorso al file .pth")
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES, help="Percorso al file .npy")
    args = parser.parse_args()

    PantherApp(weights_path=args.weights, classes_path=args.classes).run()

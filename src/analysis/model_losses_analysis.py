"""
Analyse der Model Losses aus den Trainingsprotokollen.

Wertet die model_losses.txt Dateien aus und gibt tabellarisch alle 
interessanten Werte aus.
"""

from pathlib import Path
import re
from dataclasses import dataclass


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


@dataclass
class ModelStats:
    """Statistiken für ein trainiertes Modell."""
    name: str
    dataset: str
    epochs: int
    best_val_loss: float
    best_epoch: int
    final_train_loss: float
    final_val_loss: float
    initial_val_loss: float
    initial_train_loss: float
    total_time_mins: float
    lr_reductions: int
    final_lr: float
    # Parsed hyperparameters
    hidden_size: int | None
    num_layers: int | None
    dropout: float | None
    model_type: str


def parse_model_name(name: str) -> dict:
    """Extrahiert Hyperparameter aus dem Modellnamen."""
    result = {
        "hidden_size": None,
        "num_layers": None,
        "dropout": None,
        "type": "LSTM"
    }
    
    # MLP Modelle
    if name.startswith("mlp_"):
        result["type"] = "MLP"
        match = re.search(r"h(\d+)", name)
        if match:
            result["hidden_size"] = int(match.group(1))
        return result
    
    # Hidden size: h200, h400, etc.
    match = re.search(r"h(\d+)", name)
    if match:
        result["hidden_size"] = int(match.group(1))
    
    # Layers: l2, l3, l4
    match = re.search(r"l(\d+)", name)
    if match:
        result["num_layers"] = int(match.group(1))
    
    # Dropout: d0, d0.1
    match = re.search(r"d([\d.]+)", name)
    if match:
        result["dropout"] = float(match.group(1))
    
    return result


def parse_losses_file(filepath: Path, dataset_label: str) -> list[ModelStats]:
    """Parst eine model_losses.txt Datei und extrahiert Statistiken."""
    models = []
    
    if not filepath.exists():
        print(f"Datei nicht gefunden: {filepath}")
        return models
    
    content = filepath.read_text(encoding="utf-8")
    
    # Split nach "Starting" um einzelne Modelle zu trennen
    model_sections = re.split(r"^Starting ", content, flags=re.MULTILINE)
    
    for section in model_sections[1:]:
        lines = section.strip().split("\n")
        if not lines:
            continue
        
        model_name = lines[0].strip()
        
        epochs = []
        train_losses = []
        val_losses = []
        times = []
        learning_rates = []
        
        for line in lines[1:]:
            match = re.match(
                r"Epoch\s+(\d+)\s+\|\s+Train Loss:\s+([\d.]+)\s+\|\s+Validation Loss:\s+([\d.]+)\s+\|\s+LR:\s+([\d.e\-]+)\s+\|\s+Time:\s+([\d.]+)mins",
                line
            )
            if match:
                epochs.append(int(match.group(1)))
                train_losses.append(float(match.group(2)))
                val_losses.append(float(match.group(3)))
                learning_rates.append(float(match.group(4)))
                times.append(float(match.group(5)))
        
        if not epochs:
            continue
        
        best_val_loss = min(val_losses)
        best_epoch = epochs[val_losses.index(best_val_loss)]
        
        lr_changes = sum(1 for i in range(1, len(learning_rates)) 
                        if learning_rates[i] < learning_rates[i-1])
        
        params = parse_model_name(model_name)
        
        stats = ModelStats(
            name=model_name,
            dataset=dataset_label,
            epochs=max(epochs),
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
            final_train_loss=train_losses[-1],
            final_val_loss=val_losses[-1],
            initial_val_loss=val_losses[0],
            initial_train_loss=train_losses[0],
            total_time_mins=sum(times),
            lr_reductions=lr_changes,
            final_lr=learning_rates[-1],
            hidden_size=params["hidden_size"],
            num_layers=params["num_layers"],
            dropout=params["dropout"],
            model_type=params["type"]
        )
        models.append(stats)
    
    return models


def print_combined_table(all_models: list[ModelStats]):
    """Gibt eine kombinierte Tabelle aller Modelle aus."""
    
    print("\n" + "="*160)
    print(" ALLE MODELLE - ÜBERSICHT")
    print("="*160)
    
    # Header
    headers = [
        "Modell", "Datensatz", "Typ", "H", "L", "D",
        "Epochs", "Best Val", "@ Ep", "Final Val", "Final Train",
        "Improv.", "Overfit", "LR↓", "Zeit"
    ]
    
    widths = [18, 12, 5, 5, 3, 5, 7, 10, 5, 10, 11, 8, 8, 4, 8]
    
    header_line = " | ".join(h.center(w) for h, w in zip(headers, widths))
    print(header_line)
    print("-" * len(header_line))
    
    # Sortiere nach Best Val Loss
    sorted_models = sorted(all_models, key=lambda m: m.best_val_loss)
    
    for i, m in enumerate(sorted_models, 1):
        improvement = ((m.initial_val_loss - m.best_val_loss) / m.initial_val_loss) * 100
        
        # Overfitting: Differenz zwischen Train und Val Loss (positiv = overfitting)
        overfit = ((m.final_val_loss - m.final_train_loss) / m.final_train_loss) * 100
        
        dataset_short = "10.12 ✓" if "10.12" in m.dataset else "17.12"
        
        row = [
            m.name[:18],
            dataset_short,
            m.model_type,
            str(m.hidden_size or "-"),
            str(m.num_layers or "-"),
            f"{m.dropout:.1f}" if m.dropout else "-",
            str(m.epochs),
            f"{m.best_val_loss:.6f}",
            str(m.best_epoch),
            f"{m.final_val_loss:.6f}",
            f"{m.final_train_loss:.6f}",
            f"{improvement:.1f}%",
            f"{overfit:+.1f}%",
            str(m.lr_reductions),
            f"{m.total_time_mins:.1f}m"
        ]
        
        # Markiere Top 3
        prefix = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        
        print(prefix + " | ".join(str(v).ljust(w) for v, w in zip(row, widths)))
    
    print()
    
    # Legende
    print("Legende:")
    print("  H = Hidden Size | L = Layers | D = Dropout | @ Ep = Beste Epoche")
    print("  Improv. = Verbesserung von Initial zu Best | Overfit = (Val-Train)/Train")
    print("  LR↓ = Anzahl Learning Rate Reductions | 10.12 ✓ = Mit Wetterfilter")


def print_summary(all_models: list[ModelStats]):
    """Gibt eine Zusammenfassung aus."""
    print("\n" + "="*80)
    print(" ZUSAMMENFASSUNG")
    print("="*80)
    
    # Gruppiere nach Dataset
    with_filter = [m for m in all_models if "10.12" in m.dataset]
    without_filter = [m for m in all_models if "17.12" in m.dataset]
    
    # Beste LSTM Modelle pro Kategorie
    lstm_with = [m for m in with_filter if m.model_type == "LSTM"]
    lstm_without = [m for m in without_filter if m.model_type == "LSTM"]
    
    if lstm_with:
        best_with = min(lstm_with, key=lambda m: m.best_val_loss)
        print(f"\n🏆 Bestes LSTM (MIT Wetterfilter, 10.12):")
        print(f"   {best_with.name}")
        print(f"   Val Loss: {best_with.best_val_loss:.6f} @ Epoche {best_with.best_epoch}")
        print(f"   Config: hidden={best_with.hidden_size}, layers={best_with.num_layers}, dropout={best_with.dropout}")
    
    if lstm_without:
        best_without = min(lstm_without, key=lambda m: m.best_val_loss)
        print(f"\n🏆 Bestes LSTM (OHNE Wetterfilter, 17.12):")
        print(f"   {best_without.name}")
        print(f"   Val Loss: {best_without.best_val_loss:.6f} @ Epoche {best_without.best_epoch}")
        print(f"   Config: hidden={best_without.hidden_size}, layers={best_without.num_layers}, dropout={best_without.dropout}")
    
    if lstm_with and lstm_without:
        diff = best_without.best_val_loss - best_with.best_val_loss
        diff_pct = (diff / best_without.best_val_loss) * 100
        print(f"\n📊 Wetterfilter-Effekt:")
        print(f"   Verbesserung durch Wetterfilter: {diff:.6f} ({diff_pct:.1f}%)")
    
    # Gesamtstatistik
    print(f"\n📈 Gesamtstatistik:")
    print(f"   Trainierte Modelle: {len(all_models)}")
    print(f"   - Mit Wetterfilter: {len(with_filter)}")
    print(f"   - Ohne Wetterfilter: {len(without_filter)}")
    
    total_time = sum(m.total_time_mins for m in all_models)
    print(f"   Gesamte Trainingszeit: {total_time:.1f} min ({total_time/60:.1f} h)")
    
    total_epochs = sum(m.epochs for m in all_models)
    print(f"   Gesamte Epochen: {total_epochs}")


def print_hyperparameter_analysis(all_models: list[ModelStats]):
    """Analysiert den Einfluss verschiedener Hyperparameter."""
    print("\n" + "="*80)
    print(" HYPERPARAMETER-ANALYSE")
    print("="*80)
    
    lstm_models = [m for m in all_models if m.model_type == "LSTM" and m.dropout in (None, 0.0)]
    
    if not lstm_models:
        return
    
    # Hidden Size Analyse
    print("\n📐 Hidden Size (nur Modelle ohne Dropout):")
    by_hidden = {}
    for m in lstm_models:
        h = m.hidden_size
        if h:
            if h not in by_hidden:
                by_hidden[h] = []
            by_hidden[h].append(m)
    
    print(f"   {'Hidden':>6} | {'Beste Val Loss':>14} | Modell")
    print("   " + "-"*50)
    for h in sorted(by_hidden.keys()):
        best = min(by_hidden[h], key=lambda m: m.best_val_loss)
        print(f"   {h:>6} | {best.best_val_loss:>14.6f} | {best.name} ({best.dataset[:5]})")
    
    # Layers Analyse
    print("\n📚 Layer Anzahl (nur Modelle ohne Dropout):")
    by_layers = {}
    for m in lstm_models:
        l = m.num_layers
        if l:
            if l not in by_layers:
                by_layers[l] = []
            by_layers[l].append(m)
    
    print(f"   {'Layers':>6} | {'Beste Val Loss':>14} | Modell")
    print("   " + "-"*50)
    for l in sorted(by_layers.keys()):
        best = min(by_layers[l], key=lambda m: m.best_val_loss)
        print(f"   {l:>6} | {best.best_val_loss:>14.6f} | {best.name} ({best.dataset[:5]})")
    
    # Dropout Analyse
    print("\n💧 Dropout Effekt:")
    with_dropout = [m for m in all_models if m.model_type == "LSTM" and m.dropout and m.dropout > 0]
    without_dropout = [m for m in all_models if m.model_type == "LSTM" and (m.dropout is None or m.dropout == 0)]
    
    if with_dropout and without_dropout:
        best_with_d = min(with_dropout, key=lambda m: m.best_val_loss)
        best_without_d = min(without_dropout, key=lambda m: m.best_val_loss)
        print(f"   Ohne Dropout: {best_without_d.best_val_loss:.6f} ({best_without_d.name})")
        print(f"   Mit Dropout:  {best_with_d.best_val_loss:.6f} ({best_with_d.name})")
        print(f"   ⚠️  Dropout verschlechtert die Performance in diesem Datensatz!")


def main():
    print("\n" + "#"*80)
    print(" MODEL LOSSES ANALYSE")
    print(" Radar Simulation ML - Trainingsauswertung")
    print("#"*80)
    
    # Parse beide Dateien
    models_filtered = parse_losses_file(
        MODELS_DIR / "model_losses.txt",
        "10.12 (mit Wetterfilter)"
    )
    
    models_unfiltered = parse_losses_file(
        MODELS_DIR / "17.12" / "model_losses.txt",
        "17.12 (ohne Wetterfilter)"
    )
    
    all_models = models_filtered + models_unfiltered
    
    # Kombinierte Tabelle
    print_combined_table(all_models)
    
    # Zusammenfassung
    print_summary(all_models)
    
    # Hyperparameter-Analyse
    print_hyperparameter_analysis(all_models)


if __name__ == "__main__":
    main()

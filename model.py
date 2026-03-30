"""
model.py — Architettura del modello e feature extraction per Panther Classifier.
"""

import torch
import torch.nn as nn
import numpy as np

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
INPUT_DIM = len(AMINO_ACIDS) + 1  # 20 aa + lunghezza


def build_model(num_classes: int) -> nn.Sequential:
    """Costruisce il modello MLP con BatchNorm e LeakyReLU."""
    return nn.Sequential(
        nn.Linear(INPUT_DIM, 2048),
        nn.BatchNorm1d(2048),
        nn.LeakyReLU(0.1),
        nn.Dropout(0.2),
        nn.Linear(2048, 1024),
        nn.BatchNorm1d(1024),
        nn.LeakyReLU(0.1),
        nn.Dropout(0.2),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.LeakyReLU(0.1),
        nn.Linear(512, num_classes),
    )


def extract_features(sequence: str) -> list[float]:
    """
    Estrae 21 feature da una sequenza proteica:
    - Frequenza relativa di ciascuno dei 20 amminoacidi standard
    - Lunghezza della sequenza (normalizzata)
    """
    if not sequence:
        return [0.0] * INPUT_DIM
    seq = sequence.upper()
    n = len(seq)
    features = [seq.count(aa) / n for aa in AMINO_ACIDS]
    features.append(float(n))
    return features


def load_model(weights_path: str, classes_path: str) -> tuple[nn.Sequential, np.ndarray]:
    """
    Carica il modello e le classi da disco.

    Returns:
        model: modello in eval mode
        classes: array delle label PANTHER
    """
    classes = np.load(classes_path, allow_pickle=True)
    model = build_model(num_classes=len(classes))
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, classes


def predict_top_k(
    model: nn.Sequential,
    classes: np.ndarray,
    sequence: str,
    k: int = 5,
) -> list[tuple[str, float]]:
    """
    Esegue la predizione su una sequenza e restituisce le top-k famiglie.

    Returns:
        Lista di tuple (nome_classe, probabilità_percentuale)
    """
    features = torch.FloatTensor(extract_features(sequence)).unsqueeze(0)
    with torch.no_grad():
        outputs = model(features)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        top_probs, top_idxs = torch.topk(probs, k)

    return [
        (classes[top_idxs[0][i].item()], top_probs[0][i].item() * 100)
        for i in range(k)
    ]

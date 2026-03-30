import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# Utilizziamo la CPU per l'inferenza di massa (più stabile se non hai milioni di righe)
device = torch.device("cpu")

def get_model(input_dim, num_classes):
    return nn.Sequential(
        nn.Linear(input_dim, 2048),
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
        nn.Linear(512, num_classes)
    )

def batch_inference(input_csv, output_csv):
    # 1. Caricamento file necessari
    model_path = "best_panther_model_pro.pth"
    classes_path = "classes_pro.npy"

    if not os.path.exists(model_path) or not os.path.exists(classes_path):
        print("Errore: File del modello o delle classi non trovati.")
        return

    class_names = np.load(classes_path, allow_pickle=True)
    num_classes = len(class_names)

    # 2. Caricamento dati da predire
    print(f"Caricamento dati da {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Se il file ha la colonna target, la ignoriamo per la predizione
    X_df = df.drop(columns=['family_label'], errors='ignore')
    input_dim = X_df.shape[1]

    # 3. Preparazione Modello
    model = get_model(input_dim, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 4. Predizione a blocchi (per non saturare la RAM)
    X_tensor = torch.from_numpy(X_df.values.astype(np.float32))
    
    predictions = []
    confidences = []

    print("Inizio predizione massiva...")
    with torch.no_grad():
        # Processiamo 1024 righe alla volta
        for i in tqdm(range(0, len(X_tensor), 1024)):
            batch = X_tensor[i : i + 1024]
            outputs = model(batch)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            prob, idx = torch.max(probabilities, dim=1)
            
            # Convertiamo gli indici nei nomi delle famiglie
            batch_preds = [class_names[p.item()] for p in idx]
            batch_confs = [p.item() * 100 for p in prob]
            
            predictions.extend(batch_preds)
            confidences.extend(batch_confs)

    # 5. Salvataggio risultati
    df['predicted_family'] = predictions
    df['confidence_percent'] = confidences
    
    df.to_csv(output_csv, index=False)
    print(f"\n Predizioni completate! File salvato in: {output_csv}")

if __name__ == "__main__":
    # Esempio d'uso:
    # batch_inference("tuo_file_test.csv", "risultati_panther.csv")
    input_file = input("Inserisci il percorso del CSV da analizzare: ")
    output_file = "predizioni_finali.csv"
    batch_inference(input_file, output_file)

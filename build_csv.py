import os
import pandas as pd
from tqdm import tqdm

# Percorsi aggiornati per il tuo laptop 'orion'
INPUT_DIR = "/home/noya/datasets/PANTHER19.0_fasta"
OUTPUT_FILE = "/home/noya/datasets/dataset_proteine_libNN.csv"

# Crea la cartella se non esiste
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

def extract_features(sequence):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    if not sequence: return [0]*21
    counts = [sequence.count(aa) / len(sequence) for aa in amino_acids]
    counts.append(len(sequence))
    return counts

def build():
    if not os.path.exists(INPUT_DIR):
        print(f" Errore: Non trovo i FASTA in {INPUT_DIR}")
        return

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.fasta')]
    print(f" Elaborazione di {len(files)} file FASTA...")
    
    data = []
    cols = list('ACDEFGHIKLMNPQRSTVWY') + ['length', 'family_label']

    for file_name in tqdm(files):
        family_id = file_name.replace(".fasta", "")
        with open(os.path.join(INPUT_DIR, file_name), 'r') as f:
            seq = ""
            for line in f:
                if line.startswith(">"):
                    if seq: data.append(extract_features(seq) + [family_id])
                    seq = ""
                else:
                    seq += line.strip()
            if seq: data.append(extract_features(seq) + [family_id])

    df = pd.DataFrame(data, columns=cols)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f" CSV generato con successo in: {OUTPUT_FILE}")

if __name__ == "__main__":
    build()

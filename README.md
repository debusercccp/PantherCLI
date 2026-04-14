# Panther Protein Classifier 

Classificatore automatico di proteine nelle famiglie del database **PANTHER**, addestrato su PyTorch e progettato per girare su dispositivi edge come il **Raspberry Pi**.

---

## Il Modello

Il classificatore e un MLP a tre strati nascosti con regolarizzazione:

| Layer    | Neuroni    | Attivazione                               |
|----------|------------|-------------------------------------------|
| Input    | 21         | —                                         |
| Hidden 1 | 2048       | BatchNorm + LeakyReLU(0.1) + Dropout(0.2) |
| Hidden 2 | 1024       | BatchNorm + LeakyReLU(0.1) + Dropout(0.2) |
| Hidden 3 | 512        | BatchNorm + LeakyReLU(0.1)                |
| Output   | *N classi* | Softmax                                   |

**Feature di input (21):** frequenza relativa dei 20 amminoacidi standard + lunghezza della sequenza.
**Accuratezza:** ~56.46% su oltre 15.000 classi PANTHER.

> Il modello e addestrato su singole sequenze proteiche. Passare file FASTA multi-sequenza
> senza `--first` concatena tutte le sequenze in una sola, producendo risultati non significativi.

---

## Modello Pre-addestrato (Non Incluso)

Il file `best_panther_model_pro.pth` contiene i pesi di un modello pre-addestrato proprietario e **non e distribuito in questo repository**.

I pesi sono stati ottenuti tramite un training personalizzato su dati PANTHER e sono disponibili solo privatamente. Per addestrare un proprio modello da zero, utilizzare lo script `trainer_panther_GPU_FINALE.py`.

Per ottenere i pesi pre-addestrati, contattare direttamente l'autore.

---

## Il Database PANTHER

**PANTHER (Protein ANalysis THrough Evolutionary Relationships)** e un sistema di classificazione su larga scala che raggruppa le proteine in base a:

- **Relazioni evolutive** — analisi dei geni ancestrali comuni tra organismi.
- **Funzione molecolare** — i membri di una stessa famiglia condividono spesso attivita molecolari simili.
- **Ontologia genica** — collegamento diretto con Gene Ontology (GO).

Il modello e addestrato sull'ultima release di PANTHER, coprendo migliaia di famiglie proteiche attraverso svariati organismi.

---

## Struttura del Progetto

```
PantherCLI/
├── pantherCLI.py                   # Interfaccia CLI (Rich)
├── model.py                        # Architettura, feature extraction, inferenza
├── build_csv.py                    # Generazione dataset da file FASTA
├── trainer_panther_GPU_FINALE.py   # Script di training
├── best_panther_model_pro.pth      # Pesi del modello (non inclusi, solo privati)
├── classes_pro.npy                 # Mappa indice -> famiglia PANTHER
├── requirements.txt                # Dipendenze Python
└── NOTICE                          # Note sulla proprieta del modello
```

---

## Installazione

### 1. Clona il repository

```bash
git clone https://github.com/debusercccp/PantherCLI.git
cd PantherCLI
```

### 2. Crea l'ambiente virtuale

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Installa le dipendenze

```bash
pip install -r requirements.txt
```

### 4. Posiziona i file del modello

Copia `best_panther_model_pro.pth` e `classes_pro.npy` nella root del progetto, oppure passa i percorsi come argomenti CLI.

---

## Utilizzo

### Modo interattivo (input multi-riga)

```bash
python3 pantherCLI.py
```

Incolla la sequenza riga per riga, poi premi Invio su una riga vuota (oppure Ctrl+D) per confermare.

### Da file

```bash
python3 pantherCLI.py --seq proteina.fasta
```

Se il file contiene piu sequenze (file di famiglia PANTHER), usa `--first` per prendere solo la prima:

```bash
python3 pantherCLI.py --seq PTHR31133.fasta --first
```

### Da pipe

```bash
cat proteina.fasta | python3 pantherCLI.py
cat PTHR31133.fasta | python3 pantherCLI.py --first
```

### Opzioni

| Flag | Default | Descrizione |
|------|---------|-------------|
| `--seq FILE` | — | File FASTA o sequenza raw da disco |
| `--first` | off | Usa solo la prima sequenza del file FASTA |
| `--weights FILE` | `best_panther_model_pro.pth` | Percorso pesi del modello |
| `--classes FILE` | `classes_pro.npy` | Percorso mappa classi |
| `--topk N` | 5 | Numero di famiglie da mostrare |

---

## Dipendenze

| Pacchetto    | Versione |
|--------------|----------|
| numpy        | 2.4.4    |
| pandas       | 3.0.1    |
| scikit-learn | 1.8.0    |
| torch        | 2.10.0   |
| tqdm         | 4.67.3   |
| rich         | —        |

> Su Raspberry Pi si consiglia `torch` nella versione CPU-only per ridurre l'overhead:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```

---

Sviluppato per bioinformatica e Deep Learning su sistemi embedded.

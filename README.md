#  Panther Protein Classifier v6

Classificatore automatico di proteine nelle famiglie del database **PANTHER**, addestrato su PyTorch e progettato per girare su dispositivi edge come il **Raspberry Pi**.

---

##  Il Modello

Il classificatore è un MLP a tre strati nascosti con regolarizzazione:

| Layer | Neuroni | Attivazione |
|-------|---------|-------------|
| Input | 21 | — |
| Hidden 1 | 2048 | BatchNorm + LeakyReLU(0.1) + Dropout(0.2) |
| Hidden 2 | 1024 | BatchNorm + LeakyReLU(0.1) + Dropout(0.2) |
| Hidden 3 | 512 | BatchNorm + LeakyReLU(0.1) |
| Output | *N classi* | Softmax |

**Feature di input (21):** frequenza relativa dei 20 amminoacidi standard + lunghezza della sequenza.  
**Accuratezza:** ~56.46% su oltre 15.000 classi PANTHER.

---

##  Il Database PANTHER

**PANTHER (Protein ANalysis THrough Evolutionary Relationships)** è un sistema di classificazione su larga scala che raggruppa le proteine in base a:

- **Relazioni evolutive** — analisi dei geni ancestrali comuni tra organismi.
- **Funzione molecolare** — i membri di una stessa famiglia condividono spesso attività molecolari simili.
- **Ontologia genica** — collegamento diretto con Gene Ontology (GO).

Il modello è addestrato sull'ultima release di PANTHER, coprendo migliaia di famiglie proteiche attraverso svariati organismi.

---

##  Struttura del Progetto

```
panther/
├── panther_cli.py          # Interfaccia TUI (Textual)
├── model.py                # Architettura, feature extraction, inferenza
├── build_csv.py            # Generazione dataset da file FASTA
├── best_panther_model_pro.pth  # Pesi del modello (da caricare manualmente)
├── classes_pro.npy         # Mappa indice → famiglia PANTHER
└── requirements.txt        # Dipendenze Python
```

---

##  Installazione

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

Copia `best_panther_model_pro.pth` e `classes_pro.npy` nella root del progetto (o passa i percorsi come argomenti).

---

##  Utilizzo

### Avvio base

```bash
python3 pantherCLI.py
```

### Con percorsi personalizzati

```bash
python3 pantherCLI.py --weights /path/to/model.pth --classes /path/to/classes.npy
```

### Nell'interfaccia

1. Incolla la sequenza proteica (formato raw o FASTA — l'header `>` viene ignorato automaticamente).
2. Premi **Predici Famiglia** oppure **Invio**.
3. Vengono mostrate le **Top 5 famiglie PANTHER** con probabilità e barra visiva.

---

### Utilizo
1. Modo interattivo:
   
```Bash
python3 pantherCLI.py
```
  Ti chiederà di incollare la sequenza e poi mostrerà la tabella.

2. Modo "veloce" (passando la sequenza come argomento):
   
```Bash
python3 pantherCLI.py MKVLVVLLS
```
3. Pipe (da un file):
  Se hai un file FASTA e vuoi processarlo al volo:

```Bash
cat proteina.fasta | xargs python3 panther_cli.py
```

##  requirements.txt

```
torch
numpy
textual
```

> Su Raspberry Pi si consiglia `torch` nella versione CPU-only per ridurre l'overhead:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```

---

Sviluppato per bioinformatica e Deep Learning su sistemi embedded.

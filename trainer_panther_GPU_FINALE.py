import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

# Configurazione driver per la tua RX 6600
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_v6():
    # 1. Caricamento Dati
    dataset_path = "/home/rocco/datasets/dataset_proteine_libNN.csv"
    df = pd.read_csv(dataset_path)
    X_df = df.drop(columns=['family_label'])
    y_df = df['family_label']

    # Codifica delle etichette (Stringhe -> Numeri)
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_df)
    num_classes = len(encoder.classes_)
    
    # SALVATAGGIO IMMEDIATO DELLE CLASSI (per non perderle se stoppi)
    np.save('classes_pro.npy', encoder.classes_)
    print(f"Mappa delle {num_classes} classi salvata in classes_pro.npy")

    X_tensor = torch.from_numpy(X_df.values.astype(np.float32))
    y_tensor = torch.from_numpy(y_encoded.astype(np.int64))
    
    loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=2048, shuffle=True)

    # 2. Architettura Potenziata (BatchNorm + LeakyReLU)
    model = nn.Sequential(
        nn.Linear(X_tensor.shape[1], 2048),
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
    ).to(device)

    # 3. Ottimizzatore e Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    best_loss = float('inf')
    epochs = 50

    print(f"Inizio training su {device}...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(loader, unit="batch", leave=False)
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Calcolo accuratezza real-time
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            running_loss += loss.item()
            acc = 100 * correct / total
            loop.set_description(f"Epoca [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.2f}%")

        avg_loss = running_loss / len(loader)
        final_acc = 100 * correct / total
        
        scheduler.step(avg_loss)
        
        print(f"Fine Epoca {epoch+1}: Loss = {avg_loss:.4f} | Acc = {final_acc:.2f}% | LR = {optimizer.param_groups[0]['lr']}")

        # Salvataggio del modello migliore
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_panther_model_pro.pth")
            print("Modello salvato!")

if __name__ == "__main__":
    train_v6()

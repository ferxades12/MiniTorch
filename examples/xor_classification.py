from src.tensor import Tensor
import numpy as np
import src.nn as nn
import src.nn.functional as F
from src.utils.data import * 
import time 

# python -m examples.xor_classification

def generate_data(n_samples):
    X = np.random.rand(n_samples, 2)
    y = np.logical_xor(X[:, 0] > 0.5, X[:, 1] > 0.5)
    y = np.where(y, 1, 0) # Mapea true/false a 1/0
    return X, y


def main():
    device = "cuda"
    epochs = 100
    batch_size = 512
    
    # Generar dataset
    x, y = generate_data(n_samples=10_000)
    dataset = Dataset(x, y, to_tensor=True, target_to_tensor=True, device=device)
    train_data, test_data = random_split(dataset, (0.8, 0.2))
    train_dataloader = Dataloader(train_data, batch_size=batch_size, shuffle=True, device=device)
    test_dataloader = Dataloader(test_data, batch_size=batch_size, shuffle=True, device=device)

    model = nn.Sequential(
        nn.Linear(2, 256, device=device),
        nn.ReLU(),
        nn.Linear(256, 256, device=device),
        nn.ReLU(),
        nn.Linear(256, 2, device=device)
    )
    model.train()
    optimizer = nn.Adam(model.parameters(), lr=0.01)

    # Entrenamiento
    start_time = time.time()  # Iniciar medición de tiempo

    for epoch in range(epochs):
        epoch_loss = 0
        for data, lbl in train_dataloader:

            # Forward pass
            out = model(data)
            loss_value = F.cross_entropy(out, lbl)
            # loss_value = F.l1(loss_value, model)
            loss_value = F.l2(loss_value, model)

            # Backward pass
            loss_value.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss_value.data

        if (epoch + 1) % 20 == 0:
            avg_loss = epoch_loss / len(train_dataloader)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    end_time = time.time()  # Finalizar medición de tiempo
    print(f"Tiempo total de entrenamiento: {end_time - start_time:.2f} segundos")

    # Evaluación en test
    model.eval()

    acc = 0
    for data, lbl in test_dataloader:
        preds = model.predict(data)
        
        # Convertir a numpy si están en GPU
        pred_data = preds.data if preds.device == "cpu" else preds.xp.asnumpy(preds.data)
        lbl_data = lbl if not isinstance(lbl, Tensor) else (lbl.data if lbl.device == "cpu" else lbl.xp.asnumpy(lbl.data))
        
        acc += np.mean(pred_data == lbl_data)
    
    print(f"Test accuracy: {acc/ len(test_dataloader)*100:.2f}%")
    

if __name__ == "__main__":
    for i in range(1):
        main()
    #print("Done")


# Benchamark 01. No optimizations
"""
65.54 segundos
65.53 segundos
65.49 segundos
65.50 segundos
65.52 segundos
65.36 segundos
65.19 segundos
65.14 segundos
65.35 segundos
65.24 segundos
65.17 segundos
65.26 segundos
65.35 segundos
65.42 segundos
65.31 segundos
65.22 segundos
65.30 segundos
65.20 segundos
65.37 segundos
65.24 segundos
65.40 segundos
65.40 segundos
65.19 segundos
65.47 segundos
65.23 segundos
65.26 segundos
65.15 segundos
65.28 segundos
65.19 segundos
65.20 segundos
"""

# PyTorch ~27s
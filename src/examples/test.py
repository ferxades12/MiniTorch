import numpy as np
from src.tensor import Tensor
import src.nn as nn
import src.nn.functional as F

def generate_data(n_samples):
    X = np.random.rand(n_samples, 2)
    y = np.logical_xor(X[:, 0] > 0.5, X[:, 1] > 0.5)
    y = np.where(y, 1, 0) # Mapea true/false a 1/0
    return X, y

class MLP(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super().__init__()

        self.linear1 = nn.Linear(num_inputs, num_hidden)
        self.act_fn_hidden = nn.ReLU()
        self.linear2 = nn.Linear(num_hidden, num_outputs)
  
    def forward(self, x):
        x = self.linear1.forward(x)
        x = self.act_fn_hidden.forward(x)
        x = self.linear2.forward(x)
        return x

    """ def parameters(self):
        params = []
        params.extend(self.linear1.parameters())
        params.extend(self.linear2.parameters())
        return params """

    def predict(self, x):
        out = self.forward(x)
        # Multiclase: salida shape (batch, clases)
        if out.data.ndim > 1 and out.data.shape[1] > 1:
            return np.argmax(out.data, axis=1)
        # Binaria: salida shape (batch, 1)
        return (out.data > 0.5).astype(int)


def main():
    # Generar dataset
    X_train, y_train = generate_data(n_samples=8000)
    X_test, y_test = generate_data(n_samples=2000)

    # Convertir a Tensor (solo datos, no gradientes)
    X_train, y_train = Tensor(X_train), Tensor(y_train.astype(np.int32))
    X_test, y_test = Tensor(X_test), Tensor(y_test.astype(np.int32))

    # Modelo
    model = MLP(num_inputs=2, num_hidden=10, num_outputs=2)
    optimizer = nn.Adam(model.parameters(), lr=0.01)

    # Entrenamiento
    epochs = 100
    batch_size = 8
    n_samples = X_train.shape()[0]
    
    for epoch in range(epochs):
        # Shuffle data
        idx = np.random.permutation(n_samples)
        X_shuffled = Tensor(X_train.data[idx])
        y_shuffled = Tensor(y_train.data[idx])

        epoch_loss = 0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            xb = Tensor(X_shuffled.data[i:end_idx], requires_grad=True)
            yb = Tensor(y_shuffled.data[i:end_idx])

            # Forward pass
            out = model(xb)
            loss_value = F.CrossEntropy(out, yb)
            #loss_value = l1(loss_value, model)
            loss_value = F.L2(loss_value, model)

            # Backward pass
            loss_value.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss_value.data
            n_batches += 1

        if (epoch+1) % 20 == 0:
            avg_loss = epoch_loss / n_batches
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    # Evaluación en test
    preds = model.predict(X_test)
    acc = np.mean(preds == y_test.data)
    print(f"Test accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    main()
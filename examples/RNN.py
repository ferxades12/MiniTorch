import numpy as np
from src.tensor import Tensor
import src.nn as nn
import src.nn.functional as F
from src.utils.data import * 
import time 

def generate_sequence_data(n_samples=1000, seq_len=10):
    """
    Generate simple sequence classification task:
    - Class 0: sequences where the sum of elements is negative
    - Class 1: sequences where the sum of elements is positive
    """
    X = []
    y = []
    
    for _ in range(n_samples):
        # Generate random sequence with shape (seq_len, input_size)
        seq = np.random.randn(seq_len, 2) * 2  # 2 features per timestep
        X.append(seq)
        
        # Label based on whether sum is positive or negative
        label = 1 if seq.sum() > 0 else 0
        y.append(label)
    
    # Stack sequences: (n_samples, seq_len, input_size)
    X = np.array(X)
    y = np.array(y)
    
    return X, y


def main():
    print("=" * 50)
    print("RNN Sequence Classification Example")
    print("=" * 50)
    
    # Hyperparameters
    input_size = 2
    hidden_size = 16
    seq_len = 10
    n_samples = 5000
    
    # Generate dataset
    print(f"\nGenerating {n_samples} sequences of length {seq_len}...")
    x, y = generate_sequence_data(n_samples=n_samples, seq_len=seq_len)
    print(f"Data shape: X={x.shape}, y={y.shape}")

    # Create dataset and dataloaders
    dataset = Dataset(x, y, to_tensor=True)
    train_data, test_data = random_split(dataset, (0.8, 0.2))
    train_dataloader = Dataloader(train_data, batch_size=32, shuffle=True)
    test_dataloader = Dataloader(test_data, batch_size=32, shuffle=False)
    
    # Build model with RNN layer
    print(f"\nBuilding RNN model:")
    print(f"  - Input size: {input_size}")
    print(f"  - Hidden size: {hidden_size}")
    print(f"  - Output classes: 2")
    
    # Model: RNN -> Linear classifier
    class RNNClassifier(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super().__init__()
            self.rnn = nn.RNN(input_size, hidden_size, nonlinearity="tanh")
            self.classifier = nn.Linear(hidden_size, num_classes)
        
        def forward(self, x):
            # x shape: (batch_size, seq_len, input_size)
            # RNN output: (seq_len, batch_size, hidden_size), hn: (batch_size, hidden_size)
            output, hn = self.rnn(x)
            
            # Use final hidden state for classification
            logits = self.classifier(hn)
            return logits
    
    model = RNNClassifier(input_size, hidden_size, num_classes=2)
    model.train()
    
    # Optimizer
    optimizer = nn.Adam(model.parameters(), lr=0.01)
    
    # Training
    epochs = 50
    print(f"\nTraining for {epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0
        for data, lbl in train_dataloader:
            # Forward pass
            out = model(data)
            loss_value = F.cross_entropy(out, lbl)
            
            # Backward pass
            loss_value.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss_value.data
        
        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(train_dataloader)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    
    end_time = time.time()
    print(f"\nTraining completed in {end_time - start_time:.2f} seconds")
    
    # Evaluation
    print("\nEvaluating on test set...")
    model.eval()
    
    correct = 0
    total = 0
    
    for data, lbl in test_dataloader:
        preds = model.predict(data)
        correct += np.sum(preds.data == lbl)
        total += len(lbl)
    
    accuracy = (correct / total) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    # Show some predictions
    print("\n" + "=" * 50)
    print("Sample Predictions:")
    print("=" * 50)
    data_sample, lbl_sample = next(iter(test_dataloader))
    
    # Take first 5 samples from the batch
    n_samples_to_show = min(5, len(lbl_sample))
    data_to_predict = Tensor(data_sample.data[:n_samples_to_show])
    preds_sample = model.predict(data_to_predict)
    
    for i in range(n_samples_to_show):
        seq_sum = data_sample.data[i].sum()
        true_label = lbl_sample[i]
        pred_label = preds_sample.data[i]
        status = "✓" if pred_label == true_label else "✗"
        print(f"{status} Sequence sum: {seq_sum:7.2f} | True: {true_label} | Pred: {pred_label}")
    
    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()

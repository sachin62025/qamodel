import torch
from torch.utils.data import DataLoader
from qamodel.model import SimpleRNN
from qamodel.dataset import QADataset
from qamodel.tokenizer import Vocabulary
from qamodel.data_loader import load_data

def train_model(filepath, epochs=20, learning_rate=0.001, batch_size=1):
    df = load_data(filepath)
    
    # Build vocabulary
    vocab = Vocabulary()
    vocab.build_vocab(df)

    # Prepare dataset and dataloader
    dataset = QADataset(df, vocab)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss, optimizer
    model = SimpleRNN(len(vocab.vocab))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0  # Count batches

        for question, answer in dataloader:
            optimizer.zero_grad()
            output = model(question)
            loss = criterion(output, answer[0])
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1  # Increment batch count

        avg_loss = total_loss / num_batches  # Compute average loss
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

    return model, vocab

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
            if answer.numel() == 0:  # Check if answer is empty
                print("Empty answer found! Debugging data:")
                print(f"Question: {question}")
                print(f"Answer: {answer}")  # Print answer tensor
                print(f"Answer shape: {answer.shape}")
                raise ValueError("Target 'answer' is empty. Check data preprocessing.")
            optimizer.zero_grad()
            output = model(question)
            if len(answer) == 0:
                answer = torch.tensor([0])  # Prevent empty target issues

            # loss = criterion(output, answer[0])
            if answer.numel() == 0:  # Check if answer is empty
                raise ValueError("Target 'answer' is empty. Check data preprocessing.")
            loss = criterion(output, answer.squeeze(1))  # Ensure correct shape

            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1  # Increment batch count

        avg_loss = total_loss / num_batches  # Compute average loss
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

    return model, vocab

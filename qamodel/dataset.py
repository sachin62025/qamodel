import torch
from torch.utils.data import Dataset

class QADataset(Dataset):
    def __init__(self, df, vocab):
        self.df = df
        self.vocab = vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        question = self.df.iloc[index]['question']
        answer = self.df.iloc[index]['answer']
        return (
            torch.tensor(self.vocab.text_to_indices(question)),
            torch.tensor(self.vocab.text_to_indices(answer))
        )

# import torch
# from torch.utils.data import Dataset

# class QADataset(Dataset):
#     def __init__(self, df, vocab):
#         self.df = df
#         self.vocab = vocab

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, index):
#         question = self.df.iloc[index]['question']
#         answer = self.df.iloc[index]['answer']
#         return (
#             torch.tensor(self.vocab.text_to_indices(question)),
#             torch.tensor(self.vocab.text_to_indices(answer))
#         )

import torch
from torch.utils.data import Dataset

class QADataset(Dataset):
    def __init__(self, df, vocab):
        self.df = df.fillna("")  # Replace NaN with empty string
        self.vocab = vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        question = str(self.df.iloc[index]['question'])  # Ensure it's a string
        answer = str(self.df.iloc[index]['answer'])  # Ensure it's a string

        # Convert text to indices
        question_tensor = torch.tensor(self.vocab.text_to_indices(question), dtype=torch.long)
        answer_tensor = torch.tensor(self.vocab.text_to_indices(answer), dtype=torch.long)

        #  Fix: If answer is empty, assign a dummy token (e.g., 0)
        if answer_tensor.numel() == 0:
            # print(f"⚠ Warning: Empty answer at index {index}. Assigning a dummy token.")
            answer_tensor = torch.tensor([0], dtype=torch.long)

        return question_tensor, answer_tensor



# def tokenize(text):
#     """Tokenizes and cleans the text."""
#     text = text.lower().replace('?', '').replace("'", "")
#     return text.split()
def tokenize(text):
    """Tokenizes and cleans the text safely."""
    if not isinstance(text, str):  # Ensure text is a string
        text = str(text) if text is not None else ""
    text = text.lower().replace('?', '').replace("'", "")
    return text.split()

class Vocabulary:
    def __init__(self):
        self.vocab = {'<UNK>': 0}

    def build_vocab(self, df):
        """Builds vocabulary from dataset."""
        for _, row in df.iterrows():
            tokens = tokenize(row['question']) + tokenize(row['answer'])
            for token in tokens:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)

    def text_to_indices(self, text):
        """Converts text to indices."""
        return [self.vocab.get(token, 0) for token in tokenize(text)]

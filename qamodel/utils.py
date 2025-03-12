import torch
import torch.nn.functional as F

def predict(model, vocab, question, threshold=0.5):
    numerical_question = vocab.text_to_indices(question)
    question_tensor = torch.tensor(numerical_question).unsqueeze(0)
    
    output = model(question_tensor)
    probs = F.softmax(output, dim=1)
    
    value, index = torch.max(probs, dim=1)
    if value < threshold:
        return "I don't know"
    return list(vocab.vocab.keys())[index]

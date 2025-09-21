import torch

def get_accuracy(logits, y):
    preds = logits.argmax(dim=1)
    return torch.eq(preds, y).float().sum() / y.size(0)

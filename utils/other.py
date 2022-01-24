import math

def calc_accuracy(y_hat, y):
    return( ((y_hat.argmax(dim=1) == y).sum() / len(y)).item() )
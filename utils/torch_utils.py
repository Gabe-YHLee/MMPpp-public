import torch

def b_mul(x, y):
    leny = len(y.size())
    lenx = len(x.size())
    if lenx > leny:
        for _ in range(lenx - leny):
            y = y.unsqueeze(-1)
        return x*y 
    else:
        for _ in range(leny - lenx):
            x = x.unsqueeze(-1)
        return x*y 
    
def b_div(x, y):
    leny = len(y.size())
    lenx = len(x.size())
    assert lenx >= leny
    for _ in range(lenx - leny):
        y = y.unsqueeze(-1)
    return x/y 
class Module:
    def __init__(self):
        pass
    def parameters(self):
        return []
class Linear(Module):
    def __init__(self, in_f, out_f):
        pass
    def __call__(self, x):
        return x

def functional_mse_loss(x, y):
    return 0.0

class functional:
    mse_loss = staticmethod(functional_mse_loss)

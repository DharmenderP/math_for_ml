import numpy as np
from value import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):
    def __init__(self, n_input, nonlin=True):
        self.w = [Value(np.random.uniform(-1, 1)) for i in range(n_input)]
        self.b = Value(np.random.uniform(-1, 1))
        self.nonlin = nonlin

    def __call__(self, x):

        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)

        out = act.relu() if self.nonlin else act
        return out
    
    def parameters(self):
        return self.w + [self.b]
    
    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"
    
class Layer(Module):
    def __init__(self, n_input, n_output):
        self.neurons = [Neuron(n_input) for _ in range(n_output)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters() ]
    
    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"
    
class MLP(Module):
    def __init__(self, nin, nouts):#nouts an array having sizes of output layer[]
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers: #calling sequentailly
            x = layer(x)

        return x
    
    def parameters(self):
        return [p for l in self.layers for p in l.parameters() ]
    
    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
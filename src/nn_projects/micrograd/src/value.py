import math
import numpy as np

class Value:
    def __init__(self, data, children=(), op='', label = ''):
        self.data = data
        self.grad = 0.0 #zero means no effect
        self._prev = set(children)
        self._op = op
        self.label = label
        self._backward = lambda : None
    
    def __repr__(self):
        return f'Value(data={self.data})'
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, children=(self, other), op='+')

        def  bc():
            self.grad += 1.0 * out.grad #+= multivariate case
            other.grad += 1.0 * out.grad

        out._backward = bc  
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, children=(self, other), op='*')

        def  bc():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = bc  
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int,float powers for now"
        out = Value(self.data ** other, (self, ), f'**{other}')

        def  bc():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = bc  
        return out
        
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other): # other - self
        return other + (-self)

    
    def __radd__(self, other): # other + self
        return self + other
    
    def __rmul__(self, other): #other * self 2 * v
        return self * other
    
    def __truediv__(self, other): #self / other
        return self * (other ** -1)
    
    def __rtruediv__(self, other): # other / self
        return other * self**-1
    
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')

        def bc():
            self.grad += out.data * out.grad

        out._backward = bc
        return out

    def tanh(self):
        n = self.data
        t = (math.exp(2 * n) - 1) / (math.exp(2 * n) + 1)
        out = Value(t, (self, ), 'tanh')

        def  bc():
            self.grad += (1 - t ** 2) * out.grad

        out._backward = bc
        return out
    
    def backward(self):
        topo = []
        vis = set()

        def build_topo(u):
            if u not in vis:
                vis.add(u)
                for i in u._prev:
                    build_topo(i)    
                topo.append(u)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
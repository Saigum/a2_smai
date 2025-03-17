import cupy as cp
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from loss_functions import *
class layer:
    def __init__(self,input_dim,output_dim,activation: function_wrapper):
        self.weights = {}
        self.grads = {}
        self.grads['W'] = None
        self.grads['b'] = None
        self.weights['W'] = cp.random.randn(input_dim,output_dim) *cp.sqrt(2/input_dim)
        self.weights['b'] = cp.random.randn(1,output_dim) *cp.sqrt(2/input_dim)
        self.activation = activation
        self.x = None
    def __call__(self, x):
        self.x = x
        self.z = cp.dot(x,self.weights['W']) + self.weights['b']
        return self.activation(self.z)
    def backward(self,upstream_grad):
        ## upstream_grad is dl/da
        #dl/da *da/dz  dz/dw
        # dl/da da/dz dz/db 
        # print(f"Shape of upstream grad = {upstream_grad.shape}")
        # print(f"Shape of W = {self.weights['W'].shape}")
        dldz = upstream_grad*self.activation.grad(self.z)
        self.grads["W"] = cp.dot(self.x.T,dldz)
        self.grads["b"] = cp.mean(dldz,axis=0,keepdims=True)
        
        # Log if gradients are NaN
        if cp.isnan(self.grads["W"]).any():
            print("Warning: W gradients contain NaN values")
        if cp.isnan(self.grads["b"]).any():
            print("Warning: b gradients contain NaN values")
        if cp.isnan(dldz).any():
            print("Warning: dldz contains NaN values")
        ## now returning upstream grad
        ## w (100,)
        # print(f"Shape of dldz = {dldz.shape} and shape of W = {self.weights['W'].shape}")
        return (dldz@self.weights["W"].T)
    def update(self,lr):
        # if(cp.linalg.norm(self.grads["W"]) ==0 ):
        #     raise ValueError("Gradients are zero")
        # if(cp.isnan(cp.linalg.norm(self.grads["W"]))):
        #     raise ValueError("Gradients are nan")
        # if(self.weights["W"].shape != self.grads["W"].shape):
        #     print(f"Shape of weights = {self.weights['W'].shape} and shape of grads = {self.grads['W'].shape}")
        #     raise ValueError("Gradients and weights shape mismatch")
        # if(self.weights["b"].shape != self.grads["b"].shape):
        #     print(f"Shape of weights = {self.weights['b'].shape} and shape of grads = {self.grads['b'].shape}")
        #     raise ValueError("Bias Gradients and weights shape mismatch")
        
        self.weights["W"] = self.weights["W"] - lr*self.grads["W"]
        self.weights["b"] =  self.weights["b"] - lr*self.grads["b"]
        self.grads["W"]  = cp.zeros_like(self.grads["W"])
        self.grads["b"] = cp.zeros_like(self.grads["b"])
    
class mlp():
    def __init__(self,input_dim,hidden_layers,output_dim
                 ,activation_function=None,apply_softmax=False,apply_sigmoid=False):
        layers = [input_dim] + hidden_layers +[output_dim]
        if(activation_function is None):
            activation_function = function_wrapper(lambda x:x,lambda x:1)
        self.layers = [layer(layers[i],layers[i+1],activation_function) for i in range(len(layers)-1)]
        if apply_softmax:
            self.layers[-1].activation = softmax()
        if apply_sigmoid:
            self.layers[-1].activation = sigmoid()
    def __call__(self,x):
        for i in range(len(self.layers)):
            x  = self.layers[i](x)
        return(x)
    def backward(self,grad):
        for i in range(len(self.layers)-1,-1,-1):
            grad = self.layers[i].backward(grad)
    def update(self,lr):
        for i in range(len(self.layers)):
            self.layers[i].update(lr)
    





        



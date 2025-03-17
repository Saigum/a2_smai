import cupy as cp

class function_wrapper:
    def __init__(self, f, f_prime):
        self.f = f
        self.f_prime = f_prime
    def __call__(self,x):
        return self.f(x)
    def grad(self,x):
        return self.f_prime(x)
    

def sigmoid_(x:cp.ndarray):
    return(1/(1+cp.exp(-x)))
def sigmoid_prime(x:cp.ndarray):
    return(sigmoid_(x)*(1-sigmoid_(x)))

class sigmoid(function_wrapper):
    def __init__(self):
        super().__init__(sigmoid_, sigmoid_prime)

def relu_(x:cp.ndarray):
    return(cp.maximum(0,x))
def relu_prime(x:cp.ndarray):
    return(x>0)

class relu(function_wrapper):
    def __init__(self):
        super().__init__(relu_,relu_prime)

def tanh_(x:cp.ndarray):
    return(cp.exp(x)-cp.exp(-x))/(cp.exp(x)+cp.exp(-x))

def tanh_prime(x:cp.ndarray):
    ## derivative of tanh is sec^2(x)
    return(4/(cp.exp(x)+cp.exp(-x))**2)

class tanh(function_wrapper):
    def __init__(self):
        super().__init__(tanh_,tanh_prime)


def softmax_(x:cp.ndarray):
    exp_max_scaled = cp.exp(x-cp.max(x,axis=1,keepdims=True))
    return(exp_max_scaled/cp.sum(exp_max_scaled))

def softmax_prime(x:cp.ndarray):
    return(cp.ones_like(x))

class softmax(function_wrapper):
    def __init__(self):
        super().__init__(softmax_,softmax_prime)

class loss_function(function_wrapper):
    def __call__(self,yhat,y):
        return self.f(yhat,y)
    def grad(self, yhat,y):
        return self.f_prime(yhat,y)
    
def mse_(yhat:cp.ndarray,y:cp.ndarray):
    return(0.5*cp.mean((yhat-y)**2,axis=0))
def mse_prime(yhat:cp.ndarray,y:cp.ndarray):
    return((yhat-y))

class mse(loss_function):
    def __init__(self):
        super().__init__(mse_,mse_prime)

def cross_entropy_(logits,y):
    return(-cp.sum(y*cp.log(softmax_(logits))))

def cross_entropy_prime(logits,y):
    ##derivative of softmax is 1.
    yhat = softmax_(logits)
    return(yhat-y/y.shape[0])

class cross_entropy(loss_function):
    def __init__(self):
        super().__init__(cross_entropy_,cross_entropy_prime)



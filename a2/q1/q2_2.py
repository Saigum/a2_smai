import pandas as pd
import cupy as cp
from loss_functions import *
from neural_networks import *

dataset = pd.read_csv(r"Bengaluru_House_Data.csv")
dataset = dataset.dropna(axis=0, how='any')
prices = cp.array(dataset['price'].to_numpy())
dataset.drop(columns="price", inplace=True)
dataset = dataset.dropna(axis=0)
dataset = dataset.select_dtypes(include=['float64', 'int64'])
dataset = cp.array(dataset.to_numpy().astype('float32'))
dataset = (dataset - dataset.mean()) / dataset.std()
print(dataset)

## normalizing the prices
prices = (prices - prices.mean()) / prices.std()

activations= [relu(),sigmoid(), tanh()]
activation_names = ["relu","sigmoid", "tanh"]
for activation in activations:
    print("--------------------------------------------")
    print(f"Training with activation function: {activation}")
    print("--------------------------------------------")
    regression_model= mlp(dataset.shape[1], [10,5,2], 1, activation_function=activation)
    lr = 0.0001
    batch_size = 100
    epochs = 10
    training_losses = []
    criterion = mse()

    from tqdm import tqdm
    with tqdm(total=epochs * (len(dataset)/batch_size)) as pbar:
        for epoch in range(epochs):
            acc = 0
            for i in range(0, len(dataset), batch_size):
                conditions = dataset[i:i + batch_size]
                y_hat = regression_model(conditions)
                # Ensure prices are the same shape as y_hat (i.e., (batch_size, 1))
                target = cp.reshape(prices[i:i + batch_size], (-1, 1))
                # print(f"y_hat shape = {y_hat.shape} and target shape = {target.shape}")
                loss = criterion(y_hat, target)
                loss_grad = criterion.grad(y_hat, target)
                regression_model.backward(loss_grad)
                regression_model.update(lr)
                training_losses.append(loss)
                pbar.update(1)
                pbar.set_description(f"Epoch {epoch} | Training Loss: {loss}")
            print(f"Epoch {epoch} | Training Loss: {loss}")
        import matplotlib.pyplot as plt
        plt.plot(training_losses)


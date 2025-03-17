from loss_functions import *
from neural_networks import *
from PIL import Image

import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle as pk

# Custom dataset class
class imgdataset():
    def __init__(self, df: pd.DataFrame,encoder:OneHotEncoder):
        self.encoder = encoder
        self.df = df
    def __getitem__(self, id):
        if isinstance(id, slice):
            rows = self.df.iloc[id]
            filepaths = [path.lstrip("../../") for path in rows["path"]]
            images = [Image.open(filepath).convert('L') for filepath in filepaths]
            # Convert images to CuPy arrays (float32) and normalize
            images = cp.array([cp.array(image).reshape(-1) for image in images], dtype=cp.float32) / 255.0
            labels = cp.array(self.encoder.transform(rows[["latex"]]).toarray(), dtype=cp.float32)
            return images, labels
        else:
            row = self.df.iloc[id]
            filepath = row["path"].lstrip("../../")
            image = Image.open(filepath)
            image = cp.array(image).reshape(1, -1) / 255.0
            label = cp.array(self.encoder.transform([[row["latex"]]]).toarray()[0], dtype=cp.float32)
            return image, label
    def __len__(self):
        return len(self.df)


def main():
    ## data preprocessing
    data_path = r"C:\Users\saira\OneDrive\Desktop\smai-main\smai-main\a2\q1\classification-task\fold-1\train.csv"
    df = pd.read_csv(data_path)
    df["path"] = df["path"].str.replace("../../", "", regex=False)
    encoder = OneHotEncoder()
    encoder.fit(df[['latex']])      

    test_data_path = r"C:\Users\saira\OneDrive\Desktop\smai-main\smai-main\a2\q1\classification-task\fold-1\test.csv"
    test_df = pd.read_csv(test_data_path)
    test_df["path"] = test_df["path"].str.replace("../../", "", regex=False)
    test_encoder = OneHotEncoder()
    test_encoder.fit(test_df[['latex']])
    ###  loaders
    test_data = imgdataset(test_df,test_encoder)
    data = imgdataset(df,encoder)


    activations=[tanh(),sigmoid(),relu()]
    activation_names=["relu","sigmoid","tanh"]
    criterion_names=["mse","cross_entropy"]
    criterions=[mse(),cross_entropy()]
    models=[]
    
    for i,criterion in enumerate(criterions):
        for j,activation in enumerate(activations):
            model = mlp(32*32, [100,100], 369, activation_function=activation)
            lr = 0.0001
            batch_size = 100
            epochs = 10
            training_accs = []
            with tqdm(total=epochs * (len(data)/batch_size)) as pbar:
                for epoch in range(epochs):
                    acc = 0
                    for i in range(0, len(data), batch_size):
                        images, latex = data[i:i + batch_size]
                        y_hat = model(images)
                        # print(f"Sum of preds = {y_hat.sum()}")
                        loss = criterion(y_hat, latex)
                        loss_grad = criterion.grad(y_hat,latex)
                        # print(loss_grad)
                        model.backward(loss_grad)
                        model.update(lr)
                        preds = cp.argmax(y_hat, axis=1)
                        true_labels = cp.argmax(latex, axis=1)
                        acc += int(cp.sum(preds == true_labels))
                        pbar.update(1)
                        samples_seen = min(i + batch_size, len(data))
                        pbar.set_description(f"Epoch {epoch}| Running accuracy: {acc / samples_seen:.4f}| Loss: {cp.mean(loss)}")
                    training_accs.append(acc / len(data))
            plt.plot(training_accs)
            plt.title(f"Training accuracy with {activation} activation")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.savefig(f"training_acc_{activation}.png")
            plt.show()
            test_acc=0
            for i in range(0, len(test_data), batch_size):
                images, latex = test_data[i:i + batch_size]
                y_hat = model(images)
                preds = cp.argmax(y_hat, axis=1)
                true_labels = cp.argmax(latex, axis=1)
                test_acc += cp.sum(preds == true_labels)
            print(f"Test accuracy for {activation_names[i]} is : {test_acc / len(test_data):.4f}")
            pk.dump(model, open(f"model_{activation_names[i]}.pkl", "wb"))

if __name__ == "__main__":
    main()
        
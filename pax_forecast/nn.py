from preprocessing import preprocessing, IMG_FOLDER
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(13, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


def nn_train(data, Y):
    X = data.values
    y = Y.values.reshape(-1, 1)

    # Scale inputs
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

    # Convert to PyTorch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    model = Net()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    # Track error per epoch
    epochs = 200
    train_mse_list = []
    test_mse_list = []
    test_r2_list = []
    test_acc_list = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = criterion(output, y_train_t)
        loss.backward()
        optimizer.step()

        # Track training and test MSE
        model.eval()
        with torch.no_grad():
            y_train_pred = model(X_train_t)
            y_test_pred = model(X_test_t)

            train_mse = criterion(y_train_pred, y_train_t).item()
            test_mse = criterion(y_test_pred, y_test_t).item()

            train_mse_list.append(train_mse)
            test_mse_list.append(test_mse)

            test_r2 = r2_score(y_test, y_test_pred.numpy())
            test_acc = np.mean(np.abs(y_test - y_test_pred.numpy()) <= 100)

            train_mse_list.append(train_mse)
            test_mse_list.append(test_mse)
            test_r2_list.append(test_r2)
            test_acc_list.append(test_acc)

    # Plot Test vs train MSE
    plt.figure(figsize=(10, 6))
    plt.plot(train_mse_list, label='Training MSE')
    plt.plot(test_mse_list, label='Test MSE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.title('Training vs Test MSE over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_FOLDER,'nn-test-train-mse.png'))
    plt.show()

    return 1


if __name__ == '__main__':
    df = preprocessing()
    X = df.drop('PAX', axis=1)
    Y = df['PAX']
    print(X.columns, len(X.columns))
    nn_train(X, Y)
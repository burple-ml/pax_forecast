from preprocessing import preprocessing, IMG_FOLDER
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(13, 64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, 64),
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
    X = scaler_X.fit_transform(X)

    # Convert to tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # Parameters
    epochs = 300
    k_folds = 5
    kf = KFold(n_splits=k_folds)

    # Track average values across folds
    train_mse_all = np.zeros(epochs)
    test_mse_all = np.zeros(epochs)
    acc_all = []
    r2_all = []

    # K-fold loop
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        print(f"Fold {fold}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        batch_size = 64  # typical value: 32–256

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size)

        model = Net()
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)
        criterion = nn.MSELoss()

        train_mse_fold = []
        test_mse_fold = []

        for epoch in range(epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                y_pred = model(batch_X)
                loss = criterion(y_pred, batch_y)
                loss.backward()
                optimizer.step()

            # Evaluation
            model.eval()
            with torch.no_grad():
                y_train_pred = model(X_train).detach().numpy()
                y_test_pred = model(X_test).detach().numpy()

                train_mse = mean_squared_error(y_train.numpy(), y_train_pred)
                test_mse = mean_squared_error(y_test.numpy(), y_test_pred)

                train_mse_fold.append(train_mse)
                test_mse_fold.append(test_mse)

        # Add epoch-wise errors for averaging later
        train_mse_all += np.array(train_mse_fold)
        test_mse_all += np.array(test_mse_fold)

        # Final predictions on this fold (for accuracy and R²)
        final_y_pred = model(X_test).detach().numpy()
        final_y_true = y_test.numpy()

        acc = np.mean(np.abs(final_y_true - final_y_pred) <= 150)
        r2 = r2_score(final_y_true, final_y_pred)

        acc_all.append(acc)
        r2_all.append(r2)

    # Average epoch-wise MSE across all folds
    train_mse_avg = train_mse_all / k_folds
    test_mse_avg = test_mse_all / k_folds

    # Plot hockey stick
    plt.figure(figsize=(10, 6))
    plt.plot(train_mse_avg, label='Avg Training MSE')
    plt.plot(test_mse_avg, label='Avg Test MSE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.title('K-Fold CV: Average Training vs Test MSE over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_FOLDER, 'nn-test-train-mse.png'))
    plt.show()

    # Final averaged scores
    print("\n=== Final Cross-Validated Performance ===")
    print(f"Average R² Score        : {np.mean(r2_all):.4f}")
    print(f"Average Accuracy ±150   : {np.mean(acc_all) * 100:.2f}%")
    return 1


if __name__ == '__main__':
    df = preprocessing()
    X = df.drop('PAX', axis=1)
    Y = df['PAX']
    print(X.columns, len(X.columns))
    nn_train(X, Y)
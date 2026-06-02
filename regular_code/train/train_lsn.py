import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from regular_code.config import LSN_CFG
from regular_code.models.lsn import LSN
from regular_code.utils.device import get_device


def K_to_onehot(K_tensor, N: int):
    return F.one_hot(K_tensor.long() - 1, num_classes=N - 1).float()


class LSNDataset(Dataset):
    def __init__(self, X, Y_onehot):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y_onehot, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx].permute(2, 0, 1)
        y = self.Y[idx]
        return x, y


def train_lsn(model, X, K_array, N: int, batch_size=256, epochs=40, lr=1e-3, device=None):
    device = get_device() if device is None else torch.device(device)
    model = model.to(device)

    K_tensor = torch.tensor(K_array)
    Y_onehot = K_to_onehot(K_tensor, N)
    dataset = LSNDataset(X, Y_onehot)

    total = len(dataset)
    train_len = int(0.7 * total)
    val_len = total - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(">>> 开始训练 LSN")
    print("训练数据量:", train_len, "验证数据量:", val_len)
    print("输入维度:", X.shape, "输出维度:", Y_onehot.shape)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        for inputs, labels_onehot in train_loader:
            inputs = inputs.to(device)
            labels = torch.argmax(labels_onehot.to(device), dim=1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_acc += (preds == labels).float().sum().item()

        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for inputs, labels_onehot in val_loader:
                inputs = inputs.to(device)
                labels = torch.argmax(labels_onehot.to(device), dim=1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_acc += (preds == labels).float().sum().item()

        print(
            f"[Epoch {epoch + 1}/{epochs}] "
            f"Train Loss={train_loss / train_len:.4f} Acc={train_acc / train_len:.4f} | "
            f"Val Loss={val_loss / val_len:.4f} Acc={val_acc / val_len:.4f}"
        )

    print(">>> LSN 训练结束！")
    return model


def main():
    data = np.load(LSN_CFG.dataset_path)
    X = data["X"]
    K_array = data["K"]
    N = X.shape[1]

    print(">>> 数据加载成功")
    print("X shape:", X.shape)
    print("K shape:", K_array.shape)
    print("N:", N)

    model = LSN(num_sources=N - 1)
    model = train_lsn(model, X, K_array, N, batch_size=256, epochs=40, lr=1e-3)
    torch.save(model.state_dict(), LSN_CFG.checkpoint_path)
    print(f">>> 模型权重已保存到 {LSN_CFG.checkpoint_path}")


if __name__ == "__main__":
    main()

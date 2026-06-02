import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from regular_code.config import LCN_CFG
from regular_code.data.lcn_data import OmegaPhiDataset
from regular_code.models.lcn import LCN
from regular_code.utils.device import get_device


SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)


def train_lcn(num_train=200_000, num_val=20_000, batch_size=256, lr=1e-3, epochs=20, save_path=None, cfg=LCN_CFG):
    assert cfg.N == 5, "当前脚本按 N=5（p=2 => u=12）对齐实现。"
    save_path = cfg.checkpoint_path if save_path is None else save_path

    device = get_device()
    print(f"[INFO] device: {device}")

    train_ds = OmegaPhiDataset(num_train, cfg)
    val_ds = OmegaPhiDataset(num_val, cfg)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = LCN(cfg.N).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    for ep in range(epochs):
        model.train()
        tr_loss = 0.0
        for x, u in train_loader:
            x = x.to(device)
            u = u.to(device)

            pred = model(x)
            loss = crit(pred, u)

            opt.zero_grad()
            loss.backward()
            opt.step()

            tr_loss += loss.item()

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for x, u in val_loader:
                x = x.to(device)
                u = u.to(device)
                va_loss += crit(model(x), u).item()

        print(
            f"Epoch [{ep + 1:02d}/{epochs}] "
            f"Train MSE={tr_loss / len(train_loader):.3e} | "
            f"Val MSE={va_loss / len(val_loader):.3e}"
        )

    torch.save(model.state_dict(), save_path)
    print(f"[INFO] saved to {save_path}")


if __name__ == "__main__":
    train_lcn()

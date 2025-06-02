import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import tqdm

class Trainer:
    """
    Trainer for video classification models with tqdm progress bars.
    """
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=3)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        pbar = tqdm.tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", unit="batch", leave=False)
        for videos, labels in pbar:
            videos = videos.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(videos)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * videos.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += videos.size(0)

            running_loss = total_loss / total
            running_acc = correct / total
            pbar.set_postfix(loss=f"{running_loss:.4f}", acc=f"{running_acc:.4f}")

        avg_loss = total_loss / total
        acc = correct / total
        return avg_loss, acc

    def eval_epoch(self, epoch):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        pbar = tqdm.tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]  ", unit="batch", leave=False)
        with torch.no_grad():
            for videos, labels in pbar:
                videos = videos.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(videos)
                loss = self.criterion(logits, labels)

                total_loss += loss.item() * videos.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += videos.size(0)

                running_loss = total_loss / total
                running_acc = correct / total
                pbar.set_postfix(loss=f"{running_loss:.4f}", acc=f"{running_acc:.4f}")

        avg_loss = total_loss / total
        acc = correct / total
        return avg_loss, acc

    def fit(self, epochs):
        train_loss_history, val_loss_history = [], []
        train_acc_history, val_acc_history = [], []
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.eval_epoch(epoch)
            self.scheduler.step(val_loss)
            print(f"Epoch {epoch}:")
            print(f"  Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")
            print(f"  Val   loss: {val_loss:.4f}, acc: {val_acc:.4f}")
            torch.save(self.model.state_dict(), "./output/r2plus1d_18_latest.pt")
            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)
        return train_loss_history, val_loss_history, train_acc_history, val_acc_history
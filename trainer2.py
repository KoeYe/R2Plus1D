import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.cuda.amp import autocast, GradScaler          # ← correct import
import tqdm
from models.r2plus1d_attn_v5 import TemporalSelfAttention


def split_tsa_params(model: nn.Module, tsa_cls):
    """Return (tsa_params, base_params) without any duplicates."""
    # 1) grab everything that lives *inside* a TemporalSelfAttention
    tsa_params = [p
                  for m in model.modules() if isinstance(m, tsa_cls)
                  for p in m.parameters()]
    tsa_ids = {id(p) for p in tsa_params}

    # 2) everything else is 'base'
    base_params = [p for p in model.parameters() if id(p) not in tsa_ids]

    assert not (set(tsa_ids) & {id(p) for p in base_params})
    return tsa_params, base_params

class Trainer:
    """
    Trainer for video‑classification models – now with TWO optimisers:
      • optim_tsa   – only TemporalSelfAttention parameters
      • optim_base  – all remaining parameters
    """
    def __init__(self, model, train_loader, val_loader,
                 device, tsa_cls=TemporalSelfAttention,           # pass the class *object* of TSA
                 lr_tsa=1e-3, lr_base=1e-4):
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device

        # ---- split params -------------------------------------------------- #
        tsa_params, base_params = split_tsa_params(self.model, tsa_cls)
        self.optim_tsa   = optim.Adam(tsa_params,   lr=lr_tsa,  weight_decay=1e-4)
        self.optim_base  = optim.Adam(base_params,  lr=lr_base, weight_decay=1e-4)

        # optional LR schedulers (one per optim)
        self.sched_tsa  = lr_scheduler.ReduceLROnPlateau(self.optim_tsa,  mode='min', patience=3)
        self.sched_base = lr_scheduler.ReduceLROnPlateau(self.optim_base, mode='min', patience=3)

        self.criterion = nn.CrossEntropyLoss()
        self.scaler    = GradScaler()        # single scaler is fine

        os.makedirs('./output', exist_ok=True)

    # --------------------------------------------------------------------- #
    def _zero_grads(self):
        self.optim_tsa.zero_grad(set_to_none=True)
        self.optim_base.zero_grad(set_to_none=True)

    # --------------------------------------------------------------------- #
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = correct = total = 0
        pbar = tqdm.tqdm(self.train_loader,
                         desc=f"Epoch {epoch} [Train]", unit="batch", leave=False)

        for videos, labels in pbar:
            videos, labels = videos.to(self.device), labels.to(self.device)
            self._zero_grads()

            with autocast():                       # mixed precision forward
                logits = self.model(videos)
                loss   = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim_tsa)
            self.scaler.step(self.optim_base)
            self.scaler.update()

            total_loss += loss.item() * videos.size(0)
            preds       = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += videos.size(0)
            pbar.set_postfix(
                loss=f"{total_loss/total:.4f}",
                acc =f"{correct/total:.4f}"
            )

        return total_loss / total, correct / total

    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def eval_epoch(self, epoch):
        self.model.eval()
        total_loss = correct = total = 0
        pbar = tqdm.tqdm(self.val_loader,
                         desc=f"Epoch {epoch} [Val]  ", unit="batch", leave=False)

        for videos, labels in pbar:
            videos, labels = videos.to(self.device), labels.to(self.device)

            with autocast():
                logits = self.model(videos)
                loss   = self.criterion(logits, labels)

            total_loss += loss.item() * videos.size(0)
            preds       = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += videos.size(0)
            pbar.set_postfix(
                loss=f"{total_loss/total:.4f}",
                acc =f"{correct/total:.4f}"
            )

        return total_loss / total, correct / total

    # --------------------------------------------------------------------- #
    def fit(self, epochs):
        best_val_loss = float('inf')
        history = {"train_loss": [], "val_loss": [],
                   "train_acc":  [], "val_acc":  []}

        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc = self.train_epoch(epoch)
            vl_loss, vl_acc = self.eval_epoch(epoch)

            self.sched_tsa.step(vl_loss)
            self.sched_base.step(vl_loss)

            print(f"Epoch {epoch:03d} | "
                  f"train loss {tr_loss:.4f} acc {tr_acc:.4f} || "
                  f"val loss {vl_loss:.4f} acc {vl_acc:.4f}")

            torch.save(self.model.state_dict(), "output/r2plus1d_18_latest.pt")
            if vl_loss < best_val_loss:
                torch.save(self.model.state_dict(), "output/r2plus1d_18_best.pt")
                best_val_loss = vl_loss

            history["train_loss"].append(tr_loss)
            history["val_loss"].append(vl_loss)
            history["train_acc"].append(tr_acc)
            history["val_acc"].append(vl_acc)

        return history


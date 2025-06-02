import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from video_datasets import HuggingFaceSSV2Dataset
# from models.r2plus1d import R2Plus1DClassifier
from models.r2plus1d_attn import R2Plus1DClassifier
import tqdm
from torch.utils.data import DataLoader
from trainer import Trainer
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = R2Plus1DClassifier(num_classes=174, pretrained=True)
data_root = "./data/something-something-v2"
train_set = HuggingFaceSSV2Dataset(data_root)
val_set = HuggingFaceSSV2Dataset(data_root, data_split='validation')
num_cls = len(train_set.idx2templates)

train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=4)

trainer = Trainer(model, train_loader, val_loader, device)
train_loss_history, val_loss_history, train_acc_history, val_acc_history = trainer.fit(epochs=30)
torch.save(model.state_dict(), "./output/r2plus1d_18.pt")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot Loss curves on the left
ax1.plot(train_loss_history, label='Train Loss')
ax1.plot(val_loss_history,   label='Val Loss')
ax1.set_title('Loss over Epochs')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True)

# Plot Accuracy curves on the right
ax2.plot(train_acc_history, label='Train Acc')
ax2.plot(val_acc_history,   label='Val Acc')
ax2.set_title('Accuracy over Epochs')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('results/ssv2_r2plus1d_attn.png', dpi=200)
plt.show()



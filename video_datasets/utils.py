from .ssv2_dataset import *
from .har_dataset import *
from orig_utils import *

def make_data_loader(dataset_name="ssv2", batch_size=8):
    if dataset_name == "ssv2":
        data_root = "/home/leo/PycharmProjects/ece285-video-project/data/something-something-v2"
        train_set = HuggingFaceSSV2Dataset(data_root, data_split="train", clip_len=16)
        val_set = HuggingFaceSSV2Dataset(data_root, data_split="validation", clip_len=16)
        test_set = HuggingFaceSSV2Dataset(data_root, data_split="test", clip_len=16)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    elif dataset_name == "har":
        data_root = "/home/leo/PycharmProjects/ece285-video-project/data/HAR"
        train_set = HARVideoDataset(data_root, data_split="train", clip_len=16)
        val_set = HARVideoDataset(data_root, data_split="validation", clip_len=16)
        test_set = HARVideoDataset(data_root, data_split="test", clip_len=16)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    else:
        train_loader, val_loader, test_loader = load_data(dataset_name, batch_size)
    return train_loader, val_loader, test_loader
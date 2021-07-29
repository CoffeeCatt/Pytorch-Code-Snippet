from torch.utils.data import Dataset
class SimpleDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.data = x
        self.targets = y
        self.transform = transform

    def __getitem__(self, index):
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        target = self.targets[index]
        return img, target

    def __len__(self):
        return len(self.data)

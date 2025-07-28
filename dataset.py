import torch
from torch.utils.data import DataLoader, Dataset

class ASTEDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def load_data(config, tokenizer):
    # Placeholder for loading actual data
    train_encodings, train_labels = {}, []
    val_encodings, val_labels = {}, []
    test_encodings, test_labels = {}, []
    return (ASTEDataset(train_encodings, train_labels),
            ASTEDataset(val_encodings, val_labels),
            ASTEDataset(test_encodings, test_labels))

def create_dataloader(dataset, config, shuffle=False):
    return DataLoader(dataset, batch_size=config['batch_size'], shuffle=shuffle)

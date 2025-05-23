from torch.utils.data import Dataset

class DatasetBase(Dataset):  
    def __init__(self, args):
        self.args = args
        self.labels = None
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.shuffle = args.shuffle

        self.dataset = None
        self.train_dataset = None
        self.valid_dataset = None

    def __updata__(self, tag):
        if tag == "train":
            self.dataset = self.train_dataset
        elif tag == "valid":
            self.dataset = self.valid_dataset
        else:
            raise ValueError("tag must be 'train' or 'valid'")

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset[index]
    



registered_datasets = {
    "base": DatasetBase,
}
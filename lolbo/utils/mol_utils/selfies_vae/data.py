import torch
import selfies as sf
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from pathlib import Path

class SELFIESDataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

        # self.train = SELFIESDataset("./selfies_data/guacamol_v1_train_selfies2.csv")
        # self.val   = SELFIESDataset("./selfies_data/guacamol_v1_test_selfies2.csv")
        self.train = SELFIESDataset(Path(Path.home() / "lolbo/data/guacamol_data/guacamol_v1_train_selfies2.csv"))
        self.val   = SELFIESDataset(Path(Path.home() / "lolbo/data/guacamol_data/guacamol_v1_test_selfies2.csv"))

        self.val.vocab     = self.train.vocab
        self.val.vocab2idx = self.train.vocab2idx

        # Drop data from val that we have no tokens for
        self.val.data = [
            smile for smile in self.val.data
            if False not in [tok in self.train.vocab for tok in smile]
        ]

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, pin_memory=True, shuffle=True, collate_fn=collate_fn, num_workers=10)

    def val_dataloader(self):
        return DataLoader(self.val,   batch_size=self.batch_size, pin_memory=True, shuffle=False, collate_fn=collate_fn, num_workers=10)

class SELFIESDataset(Dataset):
    def __init__(self, fname):
        with open(fname, 'r') as f:
            selfie_strings = [x.strip() for x in f.readlines()]

        self.data = []
        for string in selfie_strings:
            self.data.append(list(sf.split_selfies(string)))

        self.vocab = set((token for selfie in self.data for token in selfie))
        self.vocab.discard(".")
        self.vocab = ['<start>', '<stop>', *sorted(list(self.vocab))]
        self.vocab2idx = {
            v:i
            for i, v in enumerate(self.vocab)
        }

    def tokenize_selfies(self, selfies_list):   
        tokenized_selfies = []
        for string in selfies_list: 
            tokenized_selfies.append(list(sf.split_selfies(string)))
        return tokenized_selfies 

    def encode(self, smiles):
        return torch.tensor([self.vocab2idx[s] for s in [*smiles, '<stop>']])

    def decode(self, tokens):
        dec = [self.vocab[t] for t in tokens]

        # Chop out start token and everything past (and including) first stop token
        stop = dec.index("<stop>") if "<stop>" in dec else None # want first stop token
        selfie = dec[0:stop] # cut off stop tokens
        while "<start>" in selfie: # start at last start token (I've seen one case where it started w/ 2 start tokens)
            start = (1+dec.index("<start>")) 
            selfie = selfie[start:]

        selfie = "".join(selfie)
        return selfie

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.encode(self.data[idx])

    @property
    def vocab_size(self):
        return len(self.vocab)

def collate_fn(data):
    # Length of longest molecule in batch 
    max_size = max([x.shape[-1] for x in data])
    return torch.vstack(
        # Pad with stop token
        [F.pad(x, (0, max_size - x.shape[-1]), value=1) for x in data]
    )
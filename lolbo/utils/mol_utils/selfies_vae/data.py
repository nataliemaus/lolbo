import torch
import selfies as sf
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from pathlib import Path

class SELFIESDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        batch_size,
        train_data_path,
        validation_data_path,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.train = SELFIESDataset(Path(Path.home() / train_data_path))
        self.val   = SELFIESDataset(Path(Path.home() / validation_data_path))

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


DEFAULT_SELFIES_VOCAB = ['<start>', '<stop>', '[#Branch1]', '[#Branch2]', 
    '[#C-1]', '[#C]', '[#N+1]', '[#N]', '[#O+1]', '[=B]', '[=Branch1]', 
    '[=Branch2]', '[=C-1]', '[=C]', '[=N+1]', '[=N-1]', '[=NH1+1]', 
    '[=NH2+1]', '[=N]', '[=O+1]', '[=OH1+1]', '[=O]', '[=PH1]', '[=P]', 
    '[=Ring1]', '[=Ring2]', '[=S+1]', '[=SH1]', '[=S]', '[=Se+1]', '[=Se]', 
    '[=Si]', '[B-1]', '[BH0]', '[BH1-1]', '[BH2-1]', '[BH3-1]', '[B]', '[Br+2]', 
    '[Br-1]', '[Br]', '[Branch1]', '[Branch2]', '[C+1]', '[C-1]', '[CH1+1]', 
    '[CH1-1]', '[CH1]', '[CH2+1]', '[CH2]', '[C]', '[Cl+1]', '[Cl+2]', '[Cl+3]', 
    '[Cl-1]', '[Cl]', '[F+1]', '[F-1]', '[F]', '[H]', '[I+1]', '[I+2]', '[I+3]', 
    '[I]', '[N+1]', '[N-1]', '[NH0]', '[NH1+1]', '[NH1-1]', '[NH1]', '[NH2+1]', 
    '[NH3+1]', '[N]', '[O+1]', '[O-1]', '[OH0]', '[O]', '[P+1]', '[PH1]', '[PH2+1]', 
    '[P]', '[Ring1]', '[Ring2]', '[S+1]', '[S-1]', '[SH1]', '[S]', '[Se+1]', '[Se-1]', 
    '[SeH1]', '[SeH2]', '[Se]', '[Si-1]', '[SiH1-1]', '[SiH1]', '[SiH2]', '[Si]'
]


class SELFIESDataset(Dataset):
    def __init__(
        self,
        fname=None,
        load_data=False,
    ):
        self.data = []
        if load_data:
            assert fname is not None
            with open(fname, 'r') as f:
                selfie_strings = [x.strip() for x in f.readlines()]
            for string in selfie_strings:
                self.data.append(list(sf.split_selfies(string)))
            self.vocab = set((token for selfie in self.data for token in selfie))
            self.vocab.discard(".")
            self.vocab = ['<start>', '<stop>', *sorted(list(self.vocab))]
        else:
            self.vocab = DEFAULT_SELFIES_VOCAB

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
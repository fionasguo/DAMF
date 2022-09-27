import torch
from torch.utils.data import Dataset


######################### Dataset object #########################

class MFData(Dataset):
    """
    multiple labels in different classes
    eg text: "gobsmacked that this horrific death does not deserve prosecution", label = [harm,cheating]
    """

    def __init__(self, encodings, mf_labels=None, domain_labels=None):

        self.encodings = encodings
        self.mf_labels = mf_labels
        self.domain_labels = domain_labels
        self.feat_embed = None

    def __getitem__(self, idx):
        # item = {key: torch.tensor(val[idx],dtype=torch.long) for key, val in self.encodings.items()}
        item = {key: torch.tensor(val.squeeze(), dtype=torch.long) for key, val in self.encodings[idx].items()}
        if self.mf_labels is not None:
            item['mf_labels'] = torch.tensor(self.mf_labels[idx],dtype=torch.float)
        if self.domain_labels is not None:
            item['domain_labels'] = torch.tensor(self.domain_labels[idx],dtype=torch.long)

        if self.feat_embed is not None:
            item['feat_embed'] = torch.tensor(self.feat_embed[idx],dtype=torch.float)

        return item, idx

    def __len__(self):
        return len(self.encodings)

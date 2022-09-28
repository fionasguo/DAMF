"""
Define the Dataset object - MFData
"""

import torch
from torch.utils.data import Dataset


class MFData(Dataset):
    """
    Dataset object used for model training.

    Multiple MF labels possible for one data example,
    eg. text: "gobsmacked that this horrific death does not deserve prosecution", label = [harm,cheating]

    Attr:
        encodings: the output from pretrained tokenizer, must include input_ids, and attention_mask
        mf_labels: one-hot encoding for mf labels, e.g. [1,0,0,1,0] for 5 class MFs.
        domain_labels: indicates which domain this example comes from
        feat_embed: feature embeddings generated when no domain adversarial training is involved.
    """

    def __init__(self, encodings, mf_labels=None, domain_labels=None):

        self.encodings = encodings
        self.mf_labels = mf_labels
        self.domain_labels = domain_labels
        self.feat_embed = None

    def __getitem__(self, idx):
        item = {key: torch.tensor(val, dtype=torch.long)
                for key, val in self.encodings[idx].items()}
        if self.mf_labels is not None:
            item['mf_labels'] = torch.tensor(
                self.mf_labels[idx], dtype=torch.float)
        if self.domain_labels is not None:
            item['domain_labels'] = torch.tensor(
                self.domain_labels[idx], dtype=torch.long)

        if self.feat_embed is not None:
            item['feat_embed'] = torch.tensor(
                self.feat_embed[idx], dtype=torch.float, requires_grad=False)

        return item, idx

    def __len__(self):
        return len(self.encodings)

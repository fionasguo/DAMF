import torch
from torch.autograd import Function

######################### Feed Forward Classifier #########################

class FFClassifier(torch.nn.Module):
    """
    feedforward classifier to be used by mf or domain classifier
    no logit layer, because mf and domain classifiers require different: sigmoid and softmax
    """

    def __init__(self,input_dim,hidden_dim,n_classes,dropout=0.3):
        super(FFClassifier,self).__init__()

        self.model = torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim),
                                         torch.nn.BatchNorm1d(hidden_dim),
                                         torch.nn.ReLU(True),
                                         torch.nn.Dropout(dropout),
                                         torch.nn.Linear(hidden_dim, n_classes)
                                         )

    def forward(self, input):

        return self.model(input)

######################### Reconstruction layer #########################

class Reconstruction(torch.nn.Module):
    """
    to balance the corrupting effect from domain adversary, to retain some document information in the features from pretrained model
    nonlinear transformation: tanh(WH+b)
    W: weights (768,768)
    b: bias (768,1)
    H: last_hidden_state from pretrained model (batch_size, seq_len, 768)
    """

    def __init__(self, embed_dim, device):
        super(Reconstruction, self).__init__()

        self.l = torch.nn.Linear(embed_dim,embed_dim)

        # self.W = torch.empty((embed_dim, embed_dim)).to(device)
        # self.W = torch.nn.Parameter(torch.nn.init.xavier_normal_(self.W))
        #
        # self.b = torch.empty((embed_dim, 1)).to(device)
        # self.b = torch.nn.Parameter(torch.nn.init.xavier_normal_(self.b)).squeeze(1)

        self.tanh = torch.nn.Tanh()

    def forward(self, input):
        # input: (batch_size, seq_len, embed_dim); output: (batch_size,seq_len,embed_dim)
        # return self.tanh(torch.einsum('bld,dd->bld', input, self.W) + self.b)
        return self.tanh(self.l(input))

class ReconstructionLoss(torch.nn.Module):
    """
    make sure that reconstructed embeddings are not too far away from the original embeddings
    original embeddings are calculated when no adversary is being trained (at the end of epoch num_no_adv)
    """

    def __init__(self):
        super(ReconstructionLoss, self).__init__()

        self.tanh = torch.nn.Tanh()

    def forward(self, orig_embed, rec_embed, seq_len):
        # embed: (batch_size, seq_len, embed_dim)
        tmp = torch.norm(rec_embed - self.tanh(orig_embed), dim=2) ** 2 # (batch_size, seq_len)
        loss = tmp.sum(1) / seq_len
        return loss.mean()

######################### Transformation layer #########################

class Transformation(torch.nn.Module):
    """
    domain-invariant transformation
    linear trans - W*H
    H: pooler_output (representation for [cls]) from pretrained model

    https://github.com/MalavikaSrikanth16/adversarial-learning-for-stance/blob/e21a85b96994306ca7a146a3d0085b9cb84c024b/src/modeling/baseline_model_layers.py#L104
    """
    def __init__(self, input_dim, device):
        super(Transformation, self).__init__()

        self.l = torch.nn.Linear(input_dim,input_dim,bias=False)

        # self.W = torch.empty((input_dim,input_dim)).to(device)
        # self.W = torch.nn.Parameter(torch.nn.init.xavier_normal_(self.W)) # (D, D)

    def forward(self, input):
        # input: (batch_size, embed_dim)
        # return torch.einsum('bd,dd->bd', input, self.W)
        return self.l(input),self.l.weight

class TransformationLoss(torch.nn.Module):
    """
    Loss for the domain-invariant transformation layer
    regularize this to the identity.
    """
    def __init__(self, dim, device):
        super(TransformationLoss, self).__init__()

        self.device = device

        self.eye = torch.eye(dim,requires_grad=False).to(device)

    def forward(self, W):
        # if multiple gpu, W is concatenated together - need to check size first
        n_gpus = W.shape[0]/768

        W_blocks = torch.split(W,768,0)

        loss = torch.tensor(0.0,dtype=torch.float).to(self.device)

        for w in W_blocks:
            loss += torch.norm(w - self.eye) ** 2

        loss /= n_gpus

        return loss
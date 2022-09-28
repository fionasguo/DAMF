import torch.nn as nn
from grad_rev_fn import ReverseLayerF
import transformers
from transformers import AutoModel,AutoConfig
transformers.logging.set_verbosity_error()

from modules import *

class MFBasic(nn.Module):
    """
    Model for moral foundation prediction with domain adversarial training
    """

    def __init__(self,
                 pretrained_path,
                 n_mf_classes,
                 dropout_rate):

        super(MFBasic,self).__init__()

        self.feature = AutoModel.from_pretrained(pretrained_path)

        self.mf_classifier = FFClassifier(768,100,n_mf_classes,dropout_rate) # hidden_dim = 100

    def forward(self, input_ids, att_mask):

        feature = self.feature(input_ids=input_ids,attention_mask=att_mask)
        pooler_output = feature.pooler_output

        class_output = self.mf_classifier(pooler_output)

        return {'class_output':class_output}

    def gen_feature_embeddings(self, input_ids, att_mask):

        feature = self.feature(input_ids=input_ids,attention_mask=att_mask)
        last_hidden_state = feature.last_hidden_state
        pooler_output = feature.pooler_output

        return last_hidden_state, pooler_output


class MFDomainAdapt(nn.Module):
    """
    Model for moral foundation prediction with domain adversarial training
    """

    def __init__(self,
                 pretrained_path,
                 n_mf_classes,n_domain_classes,
                 dropout_rate,
                 device,
                 transformation=False,
                 reconstruction=False):

        super(MFDomainAdapt,self).__init__()

        # if is_training:
        #     self.feature = AutoModel.from_pretrained(pretrained_path)
        # else:
        #     self.pretrained_config = AutoConfig.from_pretrained(pretrained_path, local_files_only=True)
        #     self.feature = AutoModel.from_config(self.pretrained_config)

        self.feature = AutoModel.from_pretrained(pretrained_path)

        self.reconstruction = reconstruction
        self.rec_module = Reconstruction(768, device) if reconstruction else None

        self.transformation = transformation
        self.trans_module = Transformation(768,device) if transformation else None

        self.mf_classifier = FFClassifier(768+n_domain_classes,100,n_mf_classes,dropout_rate) # hidden_dim = 100

        self.n_domain_classes = n_domain_classes
        self.domain_classifier = FFClassifier(768,100,n_domain_classes,dropout_rate)

    def gen_feature_embeddings(self, input_ids, att_mask):

        feature = self.feature(input_ids=input_ids,attention_mask=att_mask)
        last_hidden_state = feature.last_hidden_state
        pooler_output = feature.pooler_output

        return last_hidden_state, pooler_output

    def forward(self, input_ids, att_mask, domain_labels, lambda_domain, adv=True):

        last_hidden_state, pooler_output = self.gen_feature_embeddings(input_ids, att_mask)

        rec_embeddings = None
        if self.reconstruction and adv:
            rec_embeddings = self.rec_module(last_hidden_state)

        trans_W = None
        if self.transformation and adv:
            pooler_output,trans_W = self.trans_module(pooler_output)

        # concat domain features onto pretrained LM embeddings before sending to MF classifier
        domain_feature = nn.functional.one_hot(domain_labels,num_classes=self.n_domain_classes).squeeze(1)
        class_input = torch.cat((pooler_output,domain_feature),dim=1)

        # connect to mf classifier
        class_output = self.mf_classifier(class_input)

        # connect to domain classifier with gradient reversal
        domain_output = None
        if adv:
            reverse_pooler_output = ReverseLayerF.apply(pooler_output, lambda_domain)
            domain_output = self.domain_classifier(reverse_pooler_output)

        return {'class_output':class_output, 'domain_output':domain_output, 'rec_embed':rec_embeddings, 'trans_W':trans_W}

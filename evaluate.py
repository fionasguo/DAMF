import os
import torch.backends.cudnn as cudnn
import torch
from torch.utils.data import DataLoader
from sklearn import metrics
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from model import MFBasic, MFDomainAdapt


def predict(model, dataset, batch_size, device, domain_adapt=True, is_adv=True):
    model.eval()

    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=False)
    len_dataloader = len(dataloader)
    data_target_iter = iter(dataloader)

    i = 0

    mf_preds = []
    mf_labels = []
    domain_preds = []
    domain_labels = []

    while i < len_dataloader:
        # test model using target data
        data_target, _ = data_target_iter.next()
        for k, v in data_target.items():
            data_target[k] = data_target[k].to(device)

        with torch.no_grad():
            if not domain_adapt:
                outputs = model(
                    data_target['input_ids'], data_target['attention_mask'])
            else:
                outputs = model(data_target['input_ids'], data_target['attention_mask'],
                                data_target['domain_labels'], lambda_domain=0, adv=is_adv)

        mf_pred_confidence = torch.sigmoid(outputs['class_output'])
        mf_pred = ((mf_pred_confidence) >= 0.5).long()
        mf_preds.extend(mf_pred.to('cpu').tolist())
        mf_labels.extend(data_target['mf_labels'].to('cpu').tolist())

        if domain_adapt and is_adv:
            domain_pred = outputs['domain_output'].data.max(1, keepdim=True)[1]
            domain_preds.extend(domain_pred.to('cpu').tolist())
            domain_labels.extend(
                data_target['domain_labels'].to('cpu').tolist())

        i += 1

    return mf_preds, mf_labels, domain_preds, domain_labels


def evaluate(dataset, batch_size, model=None, model_path=None, domain_adapt=True, is_adv=True, test=False):
    if test:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    assert (model != None or model_path !=
            None), 'Provide a model object or a model path.'
    if model == None:
        model = torch.load(model_path, map_location=torch.device(device))

    # for evaluation, don't use nn.DataParallel (multiple gpus)
    try:
        model = model.module.to(device)
    except:
        model = model.to(device)

    mf_preds, mf_labels, domain_preds, domain_labels = predict(
        model, dataset, batch_size, device, domain_adapt, is_adv)

    # print
    mf_report = metrics.classification_report(
        mf_labels, mf_preds, zero_division=0)
    logging.debug('MF classification report:')
    logging.debug(mf_report)
    conf_matrix = metrics.multilabel_confusion_matrix(mf_labels, mf_preds)
    logging.debug('MF classification confusion matrix:')
    logging.debug(conf_matrix)
    mf_report = metrics.classification_report(
        mf_labels, mf_preds, zero_division=0, output_dict=True)
    macro_f1 = mf_report['weighted avg']['f1-score']

    if domain_adapt and is_adv:
        domain_report = metrics.classification_report(
            domain_labels, domain_preds, zero_division=0)
        conf_matrix = metrics.confusion_matrix(domain_labels, domain_preds)
        logging.debug('Domain classification report:')
        logging.debug(domain_report)
        logging.debug('Domain classification confusion matrix:')
        logging.debug(conf_matrix)

    return macro_f1

################################ feature embedding analysis ##################################


def compute_feat(model, dataset, batch_size, device):
    # get feature embeddings
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=False)

    len_dataloader = len(dataloader)
    data_iter = iter(dataloader)

    model.eval()

    i = 0

    feats = []
    last_hidden_states = []
    domain_labels = []

    while i < len_dataloader:
        data, _ = data_iter.next()
        for k, v in data.items():
            data[k] = data[k].to(device)

        with torch.no_grad():
            try:
                last_hidden_state, feat = model.module.gen_feature_embeddings(
                    data['input_ids'], data['attention_mask'])
            except:
                last_hidden_state, feat = model.gen_feature_embeddings(
                    data['input_ids'], data['attention_mask'])

        feat = feat.detach()
        feats.extend(feat.data.tolist())

        last_hidden_state = last_hidden_state.detach()
        last_hidden_states.extend(last_hidden_state.to('cpu').tolist())

        domain_labels.extend(data['domain_labels'].tolist())

        i += 1

    dataset.feat_embed = last_hidden_states

    return feats, domain_labels


def plot_heatmap(feats, domain_labels, fig_save_path):
    # heat map
    plt.figure(figsize=[10, 8])
    fig = sns.heatmap(feats, yticklabels=domain_labels).get_figure()

    fig.savefig(fig_save_path + '/heatmap.png', format='png')


def plot_tsne(feats, domain_labels, fig_save_path):

    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(feats)

    plt.figure(figsize=(8, 8))
    plt.xlim((-40, 40))
    plt.ylim((-40, 40))
    fig = sns.scatterplot(
        x=tsne_results[:, 0], y=tsne_results[:, 1],
        hue=domain_labels,
        palette=sns.color_palette("hls", len(np.unique(domain_labels))),
        legend="full",
        alpha=0.3
    ).get_figure()

    fig.savefig(fig_save_path + '/tsne.png', format='png')


def feature_embedding_analysis(source_dataset, target_dataset, batch_size, fig_save_path, model=None, model_path=None):

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    assert (model != None or model_path !=
            None), 'Provide a model object or a model path.'
    if model == None:
        model = torch.load(model_path, map_location=torch.device(device))

    model = model.eval()

    model = model.to(device)

    # get feature embeddings of source and target data from the best model (adv model)
    s_feats, s_domain_labels = compute_feat(
        model, source_dataset, batch_size, device)
    t_feats, t_domain_labels = compute_feat(
        model, target_dataset, batch_size, device)

    s_feats.extend(t_feats)
    s_domain_labels.extend(t_domain_labels)
    feats = np.array(s_feats)
    domain_labels = np.array(s_domain_labels).squeeze()

    np.savetxt(fig_save_path + '/feat_embeddings.tsv', feats)
    np.savetxt(fig_save_path + '/domain_labels.tsv', domain_labels)

    plot_heatmap(feats, domain_labels, fig_save_path)
    plot_tsne(feats, domain_labels, fig_save_path)
